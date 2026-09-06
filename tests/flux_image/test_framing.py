from pathlib import Path

from PIL import Image, ImageDraw

from autotransition.flux_image.framing import frame_avatar_image


def test_frame_avatar_image_outputs_canonical_portrait_and_reduces_subject(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "framed.png"
    image = Image.new("RGB", (1000, 1000), (24, 160, 210))
    ImageDraw.Draw(image).rectangle((380, 120, 620, 880), fill=(230, 220, 180))
    image.save(source)

    report = frame_avatar_image(source, output, width=480, height=832, subject_scale=0.70)

    with Image.open(output) as framed:
        assert framed.size == (480, 832)
        assert framed.getpixel((0, 0)) == (24, 160, 210)
        pixels = framed.load()
        subject = [
            (x, y)
            for y in range(framed.height)
            for x in range(framed.width)
            if pixels[x, y] == (230, 220, 180)
        ]
        assert subject
        assert max(x for x, _ in subject) - min(x for x, _ in subject) < 480 * 0.70
        assert max(y for _, y in subject) - min(y for _, y in subject) < 832 * 0.70
    assert report.fallback_used is False
    assert report.detected_subject_bounds is not None

