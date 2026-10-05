"""Caption construction: tag shuffling, dropout, and tags/NL mixing.

Ported from diffusion-pipe (models/trainer.py:72-520), which is the best-developed part of that
trainer. Both target datasets use the same layout: `<stem>.txt` holds comma-separated tags and
`<stem>_nl.txt` an optional natural-language description.

Live augmentation requires encoding each caption. Persistent caption-variation caching instead
precomputes a finite sequence of augmented captions, preserving duplicate slots while sharing
exact embedding matches. Latent caching remains independent of either text-encoding mode.
"""

from __future__ import annotations

import functools
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CaptionMode = str  # "tags" | "nl" | "tags_nl" | "nl_tags" | "mixed"

_VARIANTS = ("tags", "nl", "tags_nl", "nl_tags")

# Fields added by the attribution feature. Kept out of a caption config's identity while they sit at
# their defaults, so turning the feature on is the only thing that invalidates existing
# caption-variation caches -- see `caption_identity`.
_ATTRIBUTION_DEFAULTS = {
    "attribution_patterns": [],
    "attribution_position": "fixed",
    "attribution_dropout_immune": True,
    "attribution_dedupe_on_combine": True,
}


@dataclass
class CaptionConfig:
    caption_mode: CaptionMode = "tags"
    # Percentages; normalised automatically, need not sum to 100.
    mixed_weights: dict[str, float] = field(
        default_factory=lambda: {"tags": 50, "nl": 10, "tags_nl": 20, "nl_tags": 20}
    )

    shuffle_tags: bool = False
    tag_delimiter: str = ", "
    shuffle_keep_first_n: int = 0     # keep N leading tags in place (trigger words)
    tag_dropout_percent: float = 0.0  # fraction of tags dropped per sample
    min_tags_kept: int = 3            # never drop below this many
    protected_tags: set[str] = field(default_factory=set)

    caption_dropout_percent: float = 0.0  # fraction of samples trained unconditional

    nl_shuffle_sentences: bool = False
    nl_keep_first_sentence: bool = False

    # Artist/trigger attribution, e.g. `Drawn by emily (pure dream)`. Each pattern is a
    # case-insensitive regex matched against every tag and every NL sentence; a match is an
    # attribution entry wherever it sits, and a post crediting several artists has several. Empty
    # = feature off, and off is byte-identical to the feature not existing.
    attribution_patterns: list[str] = field(default_factory=list)
    # "fixed" pins attribution entries to the front of their field (tags or NL), in their original
    # order, whatever shuffling does. "random" lets each land anywhere in its field.
    attribution_position: str = "fixed"
    # Pull attribution entries out before tag dropout / shuffling so they can never be dropped.
    # False leaves them in the ordinary pool (dropout and shuffle apply; "fixed" still pins
    # whichever survive).
    attribution_dropout_immune: bool = True
    # tags_nl / nl_tags join both fields, so an attribution present in both would appear twice.
    # True strips the trailing field's copy of any attribution the leading field already carries.
    attribution_dedupe_on_combine: bool = True

    def __post_init__(self):
        if self.caption_mode not in (*_VARIANTS, "mixed"):
            raise ValueError(f"unknown caption_mode: {self.caption_mode}")
        for k in self.mixed_weights:
            if k not in _VARIANTS:
                raise ValueError(f"unknown mixed_weights key: {k}")
        if not 0.0 <= self.tag_dropout_percent <= 1.0:
            raise ValueError("tag_dropout_percent must be in [0,1]")
        if not 0.0 <= self.caption_dropout_percent <= 1.0:
            raise ValueError("caption_dropout_percent must be in [0,1]")
        if isinstance(self.attribution_patterns, str):
            self.attribution_patterns = [self.attribution_patterns]
        self.attribution_patterns = [str(p) for p in self.attribution_patterns if str(p).strip()]
        for p in self.attribution_patterns:
            try:
                re.compile(p, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"dataset.caption.attribution_patterns: invalid regex {p!r}: {exc}")
        if self.attribution_position not in ("fixed", "random"):
            raise ValueError("dataset.caption.attribution_position must be 'fixed' or 'random', "
                             f"got {self.attribution_position!r}")

    @property
    def attribution_enabled(self) -> bool:
        return bool(self.attribution_patterns)

    @classmethod
    def from_dict(cls, d: dict) -> CaptionConfig:
        d = dict(d)
        path = d.pop("protected_tags_file", None)
        cfg = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if path:
            cfg.protected_tags = load_protected_tags(path)
        return cfg


def load_protected_tags(path: str | Path) -> set[str]:
    """One tag per line; blank lines and `#` comments ignored."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"protected tags file not found: {p}")
    tags = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            tags.add(line.lower())
    return tags


def split_tags(caption: str, delimiter: str = ", ") -> list[str]:
    sep = delimiter.strip() or ","
    return [t.strip() for t in caption.split(sep) if t.strip()]


def caption_identity(cfg: CaptionConfig) -> dict:
    """The caption config as a plain dict, for cache keys.

    The attribution fields are omitted while the feature is off, so a config that does not use it
    keys exactly as it did before the fields existed and existing caption-variation caches stay
    valid. With patterns set, all four are part of the identity.
    """
    data = asdict(cfg)
    if not cfg.attribution_enabled:
        for key in _ATTRIBUTION_DEFAULTS:
            data.pop(key, None)
    return data


def process_tags(tags: list[str], cfg: CaptionConfig, rng: random.Random,
                 extra_kept: int = 0) -> list[str]:
    """Apply dropout then shuffling, respecting protected and pinned-leading tags.

    `extra_kept` counts tags held out of this list that will be in the final caption anyway --
    attribution entries pulled out before dropout -- so `min_tags_kept` floors the caption, not
    just the part of it this function sees.
    """
    if not tags:
        return tags

    keep_n = min(cfg.shuffle_keep_first_n, len(tags))
    head, tail = tags[:keep_n], tags[keep_n:]

    if cfg.tag_dropout_percent > 0 and tail:
        # Leading pinned tags and protected tags are exempt from dropout.
        keepable = [t for t in tail if t.lower() in cfg.protected_tags]
        droppable = [t for t in tail if t.lower() not in cfg.protected_tags]

        n_drop = int(len(droppable) * cfg.tag_dropout_percent + 0.5)
        # Enforce the floor across the whole caption, not just the droppable subset.
        max_droppable = max(0, extra_kept + len(head) + len(keepable) + len(droppable)
                            - cfg.min_tags_kept)
        n_drop = min(n_drop, max_droppable)

        if n_drop > 0:
            survivors = set(rng.sample(range(len(droppable)), len(droppable) - n_drop))
            droppable = [t for i, t in enumerate(droppable) if i in survivors]

        # Rebuild preserving original relative order.
        kept = set(keepable) | set(droppable)
        tail = [t for t in tail if t in kept]

    if cfg.shuffle_tags:
        rng.shuffle(tail)

    return head + tail


def process_nl(nl: str, cfg: CaptionConfig, rng: random.Random) -> str:
    """Optionally shuffle sentences; the first often carries framing/subject and can be pinned."""
    if not nl or not cfg.nl_shuffle_sentences:
        return nl

    parts = [s.strip() for s in nl.split(". ") if s.strip()]
    if len(parts) < 2:
        return nl

    if cfg.nl_keep_first_sentence:
        head, rest = parts[:1], parts[1:]
        rng.shuffle(rest)
        parts = head + rest
    else:
        rng.shuffle(parts)

    return _join_sentences(parts)


def _split_sentences(nl: str) -> list[str]:
    return [s.strip() for s in nl.split(". ") if s.strip()]


def _join_sentences(parts: list[str]) -> str:
    # Only the original last sentence carries its full stop, so after a shuffle it can land in the
    # middle; stripping before joining keeps that from producing "...stands.. The girl".
    out = ". ".join(p.rstrip(".") for p in parts)
    return out if not out or out.endswith(".") else out + "."


def select_variant(cfg: CaptionConfig, has_nl: bool, rng: random.Random) -> str:
    """Pick a caption form. Falls back to tags-only when the sample has no NL caption."""
    if cfg.caption_mode != "mixed":
        variant = cfg.caption_mode
    else:
        weights = {k: v for k, v in cfg.mixed_weights.items() if v > 0}
        if not weights:
            return "tags"
        keys = list(weights)
        variant = rng.choices(keys, weights=[weights[k] for k in keys], k=1)[0]

    if not has_nl and variant in ("nl", "tags_nl", "nl_tags"):
        return "tags"
    return variant


def build_caption(
    tags_text: str,
    nl_text: str | None,
    cfg: CaptionConfig,
    rng: random.Random | None = None,
) -> str:
    """Produce the final caption string for one sample.

    Returns "" when caption dropout fires — the unconditional sample the model needs for CFG.
    """
    rng = rng or random

    if cfg.caption_dropout_percent > 0 and rng.random() < cfg.caption_dropout_percent:
        return ""

    variant = select_variant(cfg, bool(nl_text), rng)

    if cfg.attribution_enabled:
        return _build_attributed(tags_text, nl_text, variant, cfg, rng)

    tags = process_tags(split_tags(tags_text, cfg.tag_delimiter), cfg, rng)
    tags_str = cfg.tag_delimiter.join(tags)

    if variant == "tags":
        return tags_str
    nl_str = process_nl(nl_text or "", cfg, rng)
    if variant == "nl":
        return nl_str
    if variant == "tags_nl":
        return f"{tags_str}. {nl_str}" if tags_str else nl_str
    if variant == "nl_tags":
        return f"{nl_str} {tags_str}" if tags_str else nl_str
    raise AssertionError(f"unreachable variant {variant}")


# --------------------------------------------------------------------------- attribution
#
# An attribution entry is one tag, or one NL sentence, matching `attribution_patterns` -- usually
# `Drawn by <artist>`. Datasets carry the same phrase in both the tags and the NL caption so the
# trigger survives whichever variant a step picks, which creates the two problems handled here:
# shuffling and dropout treat a trigger like decoration, and the combined variants repeat it.
# Ported from the diffusion-pipe-mageflow-ft fork, generalised to captions crediting several
# artists (that fork pulled out only the first match, so a second artist was still dropped).


@functools.lru_cache(maxsize=32)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


def is_attribution(entry: str, cfg: CaptionConfig) -> bool:
    text = entry.strip()
    return any(p.search(text) for p in _compiled(tuple(cfg.attribution_patterns)))


def _attribution_key(entry: str) -> str:
    """Comparable form, so `Drawn by Emily` (a tag) and `drawn by emily.` (a sentence) match."""
    return " ".join(entry.strip().rstrip(".").split()).casefold()


def _split_attribution(entries: list[str], cfg: CaptionConfig) -> tuple[list[str], list[str]]:
    attrs = [e for e in entries if is_attribution(e, cfg)]
    rest = [e for e in entries if not is_attribution(e, cfg)]
    return attrs, rest


def _place(entries: list[str], attrs: list[str], cfg: CaptionConfig,
           rng: random.Random) -> list[str]:
    if not attrs:
        return entries
    if cfg.attribution_position == "random":
        out = list(entries)
        for a in attrs:
            out.insert(rng.randint(0, len(out)), a)
        return out
    return attrs + entries


def _process_tags_attributed(tags: list[str], cfg: CaptionConfig,
                             rng: random.Random) -> list[str]:
    if cfg.attribution_dropout_immune:
        attrs, rest = _split_attribution(tags, cfg)
        return _place(process_tags(rest, cfg, rng, extra_kept=len(attrs)), attrs, cfg, rng)
    out = process_tags(tags, cfg, rng)
    if cfg.attribution_position == "fixed":
        attrs, rest = _split_attribution(out, cfg)
        return attrs + rest
    return out


def _process_nl_attributed(nl: str, cfg: CaptionConfig, rng: random.Random) -> str:
    parts = _split_sentences(nl)
    attrs, rest = _split_attribution(parts, cfg)
    if not attrs:
        return process_nl(nl, cfg, rng)
    if not cfg.attribution_dropout_immune:
        # In the ordinary pool: shuffled like any other sentence, then pinned if "fixed".
        rest = parts
        attrs = []
    if cfg.nl_shuffle_sentences and len(rest) > 1:
        if cfg.nl_keep_first_sentence:
            head, tail = rest[:1], rest[1:]
            rng.shuffle(tail)
            rest = head + tail
        else:
            rest = list(rest)
            rng.shuffle(rest)
    if attrs:
        rest = _place(rest, attrs, cfg, rng)
    elif cfg.attribution_position == "fixed":
        pinned, others = _split_attribution(rest, cfg)
        rest = pinned + others
    return _join_sentences(rest)


def _drop_duplicate_attribution(entries: list[str], leading: list[str],
                                cfg: CaptionConfig) -> list[str]:
    """Remove attribution entries the leading field already carries. Others are kept: a tag list
    crediting two artists and an NL caption naming one should still name both once."""
    keys = {_attribution_key(e) for e in leading if is_attribution(e, cfg)}
    if not keys:
        return entries
    return [e for e in entries if not (is_attribution(e, cfg) and _attribution_key(e) in keys)]


def _build_attributed(tags_text: str, nl_text: str | None, variant: str, cfg: CaptionConfig,
                      rng: random.Random) -> str:
    # Same draw order as the plain path: tags are always processed, the NL only when used.
    tags = _process_tags_attributed(split_tags(tags_text, cfg.tag_delimiter), cfg, rng)
    if variant == "tags":
        return cfg.tag_delimiter.join(tags)
    nl_str = _process_nl_attributed(nl_text or "", cfg, rng)
    if variant == "nl":
        return nl_str
    dedupe = cfg.attribution_dedupe_on_combine
    if variant == "tags_nl":
        if dedupe:
            sentences = _split_sentences(nl_str)
            kept = _drop_duplicate_attribution(sentences, tags, cfg)
            if len(kept) != len(sentences):
                nl_str = _join_sentences(kept)
        tags_str = cfg.tag_delimiter.join(tags)
        if not nl_str:
            return tags_str
        return f"{tags_str}. {nl_str}" if tags_str else nl_str
    if variant == "nl_tags":
        if dedupe:
            tags = _drop_duplicate_attribution(tags, _split_sentences(nl_str), cfg)
        tags_str = cfg.tag_delimiter.join(tags)
        return f"{nl_str} {tags_str}" if tags_str else nl_str
    raise AssertionError(f"unreachable variant {variant}")


def read_caption_files(image_path: str | Path) -> tuple[str, str | None]:
    """`<stem>.txt` -> tags, `<stem>_nl.txt` -> NL caption (None if absent)."""
    p = Path(image_path)
    tags_path = p.with_suffix(".txt")
    nl_path = p.with_name(p.stem + "_nl.txt")

    tags = tags_path.read_text(encoding="utf-8").strip() if tags_path.exists() else ""
    nl = nl_path.read_text(encoding="utf-8").strip() if nl_path.exists() else None
    return tags, nl
