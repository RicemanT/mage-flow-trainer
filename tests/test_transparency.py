"""Transparent images load onto a solid palette colour, never onto the colours hidden under their alpha."""
from PIL import Image

from trainer.data.bucket import BucketManager
from trainer.data.cache import FLATTEN_PALETTE, load_and_crop, to_rgb


def sprite(size=(64, 64)):
    image = Image.new("RGBA", size, (255, 0, 255, 0))          # magenta garbage under full transparency
    image.paste((20, 120, 40, 255), (16, 16, 48, 48))
    return image


def test_transparent_image_gets_a_palette_background(tmp_path):
    path = tmp_path / "sprite.png"
    sprite().save(path)
    buckets = BucketManager(False, (64, 64), 32, 128, 16)
    buckets.make_buckets()
    img, _ = load_and_crop(path, buckets)
    corner, centre = img.getpixel((0, 0)), img.getpixel((img.width // 2, img.height // 2))
    assert img.mode == "RGB" and corner in FLATTEN_PALETTE and corner != (255, 0, 255)
    assert centre == (20, 120, 40)
    assert to_rgb(sprite(), "sprite").getpixel((0, 0)) == corner          # the same file name, the same colour


def test_opaque_and_palette_images():
    opaque = Image.new("RGBA", (8, 8), (1, 2, 3, 255))
    assert to_rgb(opaque, "x").getpixel((0, 0)) == (1, 2, 3)
    indexed = Image.new("P", (8, 8), 0)
    indexed.putpalette([255, 0, 255] + [0] * 765)
    indexed.info["transparency"] = 0
    assert to_rgb(indexed, "y").getpixel((0, 0)) in FLATTEN_PALETTE
