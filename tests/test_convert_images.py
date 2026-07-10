"""CPU tests for scripts/convert_images_to_jpg.py."""

import tempfile
from pathlib import Path

from click.testing import CliRunner
from PIL import Image, features

from scripts import convert_images_to_jpg as converter


def test_recursive_conversion():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        rgba = root / "transparent.png"
        nested = root / "nested" / "image.webp"
        extensionless = root / "camera_export"
        existing = root / "existing.jpg"
        nested.parent.mkdir()
        Image.new("RGBA", (8, 8), (255, 0, 0, 0)).save(rgba)
        Image.new("RGB", (9, 7), "green").save(nested)
        Image.new("RGB", (6, 5), "blue").save(extensionless, format="JPEG")
        Image.new("RGB", (4, 4), "black").save(existing)
        if features.check("avif"):
            Image.new("RGB", (7, 6), "purple").save(root / "image.avif")

        result = CliRunner().invoke(converter.main, [str(root)])
        assert result.exit_code == 0, result.output
        expected = [
            root / "transparent.jpg",
            root / "nested" / "image.jpg",
            root / "camera_export.jpg",
        ]
        if features.check("avif"):
            expected.append(root / "image.jpg")
        for path in expected:
            with Image.open(path) as image:
                assert image.format == "JPEG"
                assert image.mode == "RGB"
        with Image.open(root / "transparent.jpg") as image:
            red, green, blue = image.getpixel((0, 0))
            assert red > 245 and green > 245 and blue > 245
        assert rgba.exists() and nested.exists() and extensionless.exists()
        assert "were already .jpg" in result.output
    print("ok  recursive conversion, transparency, AVIF, and extensionless input")


def test_delete_dry_run_and_output_dir():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "input" / "nested" / "source.png"
        output = root / "output"
        source.parent.mkdir(parents=True)
        Image.new("RGB", (8, 8), "orange").save(source)

        dry = CliRunner().invoke(
            converter.main,
            [str(root / "input"), "--output-dir", str(output), "--dry-run"],
        )
        assert dry.exit_code == 0, dry.output
        assert not (output / "nested" / "source.jpg").exists()

        result = CliRunner().invoke(
            converter.main,
            [
                str(root / "input"),
                "--output-dir",
                str(output),
                "--delete-originals",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (output / "nested" / "source.jpg").is_file()
        assert not source.exists()
    print("ok  dry run, separate output tree, and delayed source deletion")


def test_collision_and_overwrite():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        Image.new("RGB", (8, 8), "blue").save(root / "same.png")
        Image.new("RGB", (8, 8), "red").save(root / "same.webp")
        collision = CliRunner().invoke(converter.main, [str(root)])
        assert collision.exit_code != 0
        assert "same JPEG" in collision.output

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source.png"
        target = root / "source.jpg"
        Image.new("RGB", (8, 8), "blue").save(source)
        target.write_bytes(b"not a jpeg")
        blocked = CliRunner().invoke(converter.main, [str(root / "source.png")])
        assert blocked.exit_code != 0

        # File input is intentionally unsupported: the public interface is a folder.
        result = CliRunner().invoke(converter.main, [str(root), "--overwrite"])
        assert result.exit_code == 0, result.output
        with Image.open(target) as image:
            assert image.format == "JPEG"
    print("ok  target collisions fail clearly and overwrite is explicit")
