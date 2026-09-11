# sstv

Decode slow-scan television (SSTV) transmissions in Python. This package
wraps the Rust [`sstv`](https://crates.io/crates/sstv) crate.

## Usage

`sstv.decode()` takes raw audio samples and returns every SSTV image found
in them as a Pillow image:

```python
import sstv
import soundfile

samples, sample_rate = soundfile.read("recording.wav")
for image in sstv.decode(samples, sample_rate):
    image.save("decoded.png")
```

Samples are a one-dimensional numpy array: `int16`, or `float32`/`float64`
in `[-1.0, 1.0]`. For multi-channel audio pass a single channel, e.g.
`samples[:, 0]`.

By default the mode of each transmission is detected from the VIS code in
its header. Both can be overridden:

```python
# The transmission is known to be Robot 36
images = sstv.decode(samples, sample_rate, mode=sstv.Mode.ROBOT_36)

# The recording starts directly at the first scanline (no header)
images = sstv.decode(samples, sample_rate, mode=sstv.Mode.ROBOT_36, header=False)
```

Decode metadata is stored in each image's `info` dict:

```python
image.info["sstv_mode"]      # e.g. sstv.Mode.ROBOT_36 — which mode was detected
image.info["sstv_complete"]  # False if the signal cut off mid-image;
                             # the rows the signal did not carry are black
```

Note that Pillow does not carry `info` through operations like `crop` or
`resize`, so read the metadata before transforming the image.

Supported modes: Scottie 1/2/DX, Martin 1/2, Robot 36/72, Wrasse SC2-180,
Pasokon P3/P5/P7, PD 50/90/120/160/180/240/290.

## Installation

```sh
pip install sstv
```

## Development

Built with [maturin](https://maturin.rs). To build and test locally:

```sh
uv venv && uv pip install -e '.[dev]'
uv run pytest
```
