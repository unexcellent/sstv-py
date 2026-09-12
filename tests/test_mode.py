"""Tests for sstv.Mode."""

import sstv


def test_dimensions():
    assert sstv.Mode.ROBOT_36.image_width == 320
    assert sstv.Mode.ROBOT_36.image_height == 240
    assert sstv.Mode.PD_290.image_width == 800
    assert sstv.Mode.PD_290.image_height == 616


def test_equality_and_hash():
    assert sstv.Mode.ROBOT_36 == sstv.Mode.ROBOT_36
    assert sstv.Mode.ROBOT_36 != sstv.Mode.ROBOT_72
    assert len({sstv.Mode.ROBOT_36, sstv.Mode.ROBOT_36, sstv.Mode.MARTIN_1}) == 2
