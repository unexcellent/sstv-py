"""Tests for sstv.encode_to_mp3() and sstv.encode_to_mp3_file()."""

from pathlib import Path

import pytest

import sstv
from tests.helpers import assert_matches


def test_round_trip(image):
    data = sstv.encode_to_mp3(image, sstv.Mode.ROBOT_36, sample_rate=22_050)
    decoded = sstv.decode_from_mp3(data)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_unsupported_sample_rate_raises(image):
    with pytest.raises(ValueError, match="sample_rate"):
        sstv.encode_to_mp3(image, sstv.Mode.ROBOT_36, sample_rate=12_345)


def test_file_round_trip(tmp_path: Path, image):
    path = tmp_path / "transmission.mp3"
    sstv.encode_to_mp3_file(image, path, sstv.Mode.ROBOT_36, sample_rate=22_050)
    decoded = sstv.decode_from_mp3(path)
    assert len(decoded) == 1
    assert_matches(decoded[0], image)


def test_file_matches_bytes(tmp_path: Path, image):
    path = tmp_path / "transmission.mp3"
    sstv.encode_to_mp3_file(image, str(path), sstv.Mode.ROBOT_36, sample_rate=22_050)
    assert path.read_bytes() == sstv.encode_to_mp3(
        image, sstv.Mode.ROBOT_36, sample_rate=22_050
    )


def test_file_unwritable_path_raises(tmp_path: Path, image):
    with pytest.raises(OSError):
        sstv.encode_to_mp3_file(
            image, tmp_path / "missing-dir" / "out.mp3", sstv.Mode.ROBOT_36
        )


def test_file_unsupported_sample_rate_raises(tmp_path: Path, image):
    with pytest.raises(ValueError, match="sample_rate"):
        sstv.encode_to_mp3_file(
            image, tmp_path / "out.mp3", sstv.Mode.ROBOT_36, sample_rate=12_345
        )
