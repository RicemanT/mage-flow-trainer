"""MOSS-Image annotation captions: short / medium / long / layout variants and the MOSS tag order."""
import json
import random
from pathlib import Path

from trainer.data.caption import CaptionConfig, build_caption, caption_identity
from trainer.data.moss_captions import Annotations, annotations_for, compile_layout, crop_transform

LAYOUT = {
    "summary": {"meta": ["anime illustration"], "canvas": "single", "caption": "A girl under a title."},
    "items": [
        {"label": "character", "caption": "Hatsune Miku standing.", "tags": ["hatsune miku"], "box_2d": [100, 100, 1000, 900]},
        {"label": "text", "caption": "Bold title.", "text": "HELLO", "box_2d": [0, 0, 80, 1000]},
        {"label": "object", "caption": "A leek at the far left edge.", "box_2d": [500, 0, 600, 60]},
    ],
}
MOSS = {"caption": {"long": "A long caption. " * 10, "medium": "A medium caption.", "short": "A short caption."}}


def write_annotations(tmp_path, stem="img", moss=MOSS, layout=LAYOUT):
    base = tmp_path / stem
    if moss is not None:
        (tmp_path / f"{stem}_moss.json").write_text(json.dumps(moss), encoding="utf-8")
    if layout is not None:
        (tmp_path / f"{stem}_layout.json").write_text(json.dumps(layout), encoding="utf-8")
    return base


def test_layout_without_crop_matches_moss_compiler():
    # 1000x1500 into a 512x768 bucket: same aspect, no crop, scale 0.512.
    out = json.loads(compile_layout(LAYOUT, (1000, 1500), (512, 768)))
    assert out["bbox_coord_space"] == "abs_centralized" and out["bbox_format"] == "yxyx"
    character = out["items"][0]
    # y: 0.1*1500*0.512 = 76.8 -> floor 76 - 384 = -308; x: 0.1*1000*0.512 = 51.2 -> 51 - 256 = -205
    assert character["bbox"] == [-308, -205, 384, 205] and character["tags"] == ["hatsune miku"]
    assert list(character) == ["label", "caption", "bbox", "tags"]
    assert out["items"][1]["text"] == "HELLO" and list(out["items"][1]) == ["label", "caption", "text", "bbox"]
    assert out["summary"] == {"meta": ["anime illustration"], "canvas": "single", "caption": "A girl under a title."}


def test_layout_drops_items_cut_off_by_the_crop():
    # A 2:1 image into a 3:2 bucket loses 1/8 of its width on each side.
    scale, left, top = crop_transform((2000, 1000), (768, 512))
    assert (round(scale, 3), round(left), round(top)) == (0.512, 128, 0)
    out = json.loads(compile_layout(LAYOUT, (2000, 1000), (768, 512)))
    labels = [item["label"] for item in out["items"]]
    assert labels == ["character"], "the full-width title loses >10% and the left-edge object is cropped out"
    assert compile_layout({"summary": {"caption": "x"}, "items": [LAYOUT["items"][2]]}, (2000, 1000), (768, 512)) is None


def test_variants_fall_back_without_files(tmp_path):
    cfg = CaptionConfig(caption_mode="mixed", mixed_weights={"short": 1})
    with_files = Annotations(write_annotations(tmp_path), (1000, 1500), (512, 768))
    without = Annotations(tmp_path / "missing", (1000, 1500), (512, 768))
    assert build_caption("1girl, smile", "An NL caption.", cfg, random.Random(0), with_files) == "A short caption."
    assert build_caption("1girl, smile", "An NL caption.", cfg, random.Random(0), without) == "An NL caption."
    assert build_caption("1girl, smile", None, cfg, random.Random(0), without) == "1girl, smile"


def test_artist_template_and_layout_summary(tmp_path):
    notes = Annotations(write_annotations(tmp_path), (1000, 1500), (512, 768))
    cfg = CaptionConfig(caption_mode="medium", attribution_patterns=["^@"])
    tags = "best quality, hatsune miku, vocaloid, @foo bar, 1girl"
    assert build_caption(tags, None, cfg, random.Random(0), notes) == "@foo bar. A medium caption."
    layout = json.loads(build_caption(tags, None, CaptionConfig(caption_mode="layout", attribution_patterns=["^@"]),
                                      random.Random(0), notes))
    assert layout["summary"]["caption"] == "@foo bar. A girl under a title."
    off = CaptionConfig(caption_mode="medium", attribution_patterns=["^@"], annotation_attribution_template="")
    assert build_caption(tags, None, off, random.Random(0), notes) == "A medium caption."


def test_moss_tag_order_keeps_the_head():
    cfg = CaptionConfig(caption_mode="tags", attribution_patterns=["^@"], keep_order_through_attribution=True,
                        shuffle_tags=True, tag_dropout_percent=0.5, min_tags_kept=1)
    head = ["masterpiece", "year 2024", "hatsune miku", "vocaloid", "@foo bar"]
    general = ["1girl", "smile", "blue hair", "twintails", "outdoors", "sky"]
    seen = set()
    for seed in range(30):
        out = build_caption(", ".join(head + general), None, cfg, random.Random(seed)).split(", ")
        assert out[:5] == head
        assert set(out[5:]) <= set(general) and len(out[5:]) < len(general)
        seen.add(tuple(out[5:]))
    assert len(seen) > 5, "the general tags are shuffled and dropped"


def test_unused_annotation_weights_do_not_change_the_cache_key():
    old = CaptionConfig(caption_mode="mixed", mixed_weights={"tags": 50, "nl": 50})
    gui = CaptionConfig(caption_mode="mixed", mixed_weights={"tags": 50, "nl": 50, "short": 0, "layout": 0})
    assert caption_identity(old) == caption_identity(gui)


def test_annotations_for_latent_entries(tmp_path):
    class Entry:
        path = tmp_path / "img_0512x0768_trainer.safetensors"
        bucket = (512, 768)
        source_size = (1000, 1500)
    write_annotations(tmp_path)
    assert annotations_for(Entry, CaptionConfig(caption_mode="tags")) is None, "no file reads when unused"
    notes = annotations_for(Entry, CaptionConfig(caption_mode="mixed", mixed_weights={"layout": 1}))
    assert notes.base == Path(tmp_path / "img") and notes.available() == {"short", "medium", "long", "layout"}


def test_layout_item_dropout_varies_granularity(tmp_path):
    notes = Annotations(write_annotations(tmp_path), (1000, 1500), (512, 768))
    full = json.loads(notes.layout())["items"]
    assert len(full) == 3
    cfg = CaptionConfig(caption_mode="layout", layout_item_dropout=0.7)
    counts = set()
    for seed in range(60):
        items = json.loads(build_caption("1girl", None, cfg, random.Random(seed), notes))["items"]
        assert 1 <= len(items) <= 3
        assert [full.index(i) for i in items] == sorted(full.index(i) for i in items), "order kept"
        counts.add(len(items))
    assert counts == {1, 2, 3}, counts
    off = CaptionConfig(caption_mode="layout")
    assert all(len(json.loads(build_caption("1girl", None, off, random.Random(s), notes))["items"]) == 3 for s in range(20))
    assert "layout_item_dropout" not in caption_identity(off)
    assert caption_identity(cfg)["layout_item_dropout"] == 0.7


def test_panel_items(tmp_path):
    comic = {"summary": {"canvas": "multi_panel", "caption": "Two panels."}, "items": [
        {"label": "panel", "caption": "Left panel: Miku waves.", "box_2d": [0, 0, 1000, 500]},
        {"label": "character", "caption": "Miku waving.", "box_2d": [100, 50, 900, 450]},
        {"label": "panel", "caption": "Right panel: Miku sleeps.", "box_2d": [0, 500, 1000, 1000]},
        {"label": "character", "caption": "Miku asleep.", "box_2d": [300, 550, 900, 950]}]}
    write_annotations(tmp_path, layout=comic)
    notes = Annotations(tmp_path / "img", (1000, 1000), (512, 512))
    items = json.loads(build_caption("1girl", None, CaptionConfig(caption_mode="layout"), random.Random(0), notes))["items"]
    assert [i["label"] for i in items] == ["panel", "character", "panel", "character"]
    assert items[0] == {"label": "panel", "caption": "Left panel: Miku waves.", "bbox": [-256, -256, 256, 0]}
    off = CaptionConfig(caption_mode="layout", layout_panels=False)
    items = json.loads(build_caption("1girl", None, off, random.Random(0), notes))["items"]
    assert [i["label"] for i in items] == ["character", "character"]
    assert "layout_panels" not in caption_identity(CaptionConfig()) and caption_identity(off)["layout_panels"] is False
    # A layout of panels only has no layout variant without panels: it falls back instead of an empty caption.
    write_annotations(tmp_path, stem="only", layout={"summary": {"caption": "x"}, "items": comic["items"][::2]})
    only = Annotations(tmp_path / "only", (1000, 1000), (512, 512))
    assert "layout" in only.available() and "layout" not in only.available(panels=False)
    assert build_caption("1girl", "An NL caption.", off, random.Random(0), only) == "An NL caption."
