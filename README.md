# sstv

Encode and decode slow-scan television (SSTV) transmissions in Python. This
package wraps the Rust [`sstv`](https://crates.io/crates/sstv) crate.

## Usage

`sstv.decode_from_wav()` takes a WAV recording and returns every SSTV image
found in it as a Pillow image:

```python
import sstv

for image in sstv.decode_from_wav("recording.wav"):
    image.save("decoded.png")
```

It accepts a path, in-memory WAV data (`bytes`), or a binary file-like
object. The sample rate is read from the WAV header, only the first channel
of multi-channel audio is used, and integer samples of any bit depth as
well as float samples are converted to 16 bit. `sstv.decode_from_mp3()` works
the same way for MP3 recordings.

For audio from other sources, `sstv.decode()` takes raw samples as a
one-dimensional numpy array — `int16`, or `float32`/`float64` in
`[-1.0, 1.0]` — plus the sample rate:

```python
import soundfile

samples, sample_rate = soundfile.read("recording.flac")
images = sstv.decode(samples[:, 0], sample_rate)
```

By default the mode of each transmission is detected from the VIS code in
its header. Both can be overridden:

```python
# The transmission is known to be Robot 36
images = sstv.decode(samples, sample_rate, mode=sstv.Mode.ROBOT_36)

# The recording starts directly at the first scanline (no header)
images = sstv.decode(samples, sample_rate, mode=sstv.Mode.ROBOT_36, header=False)
```

Both keyword arguments work the same on `decode_from_wav()` and `decode_from_mp3()`.

Decode metadata is stored in each image's `info` dict:

```python
image.info["sstv_mode"]      # e.g. sstv.Mode.ROBOT_36 — which mode was detected
image.info["sstv_complete"]  # False if the signal cut off mid-image;
                             # the rows the signal did not carry are black
```

Note that Pillow does not carry `info` through operations like `crop` or
`resize`, so read the metadata before transforming the image.

### Encoding

`sstv.encode_to_wav_file()` turns an image into a WAV file;
`sstv.encode_to_wav()` returns the WAV data as `bytes` instead, and
`sstv.encode()` the raw int16 samples, e.g. for feeding an audio device:

```python
from PIL import Image

photo = Image.open("photo.png").resize((320, 240))
sstv.encode_to_wav_file(photo, "out.wav", sstv.Mode.ROBOT_36)

data = sstv.encode_to_wav(photo, sstv.Mode.ROBOT_36)
samples = sstv.encode(photo, sstv.Mode.ROBOT_36, sample_rate=44_100)
```

`sstv.encode_to_mp3()` / `sstv.encode_to_mp3_file()` produce the
transmission as mono 128 kbit/s MP3 instead; their sample rate must be one
MP3 supports (8000-48000, see the docstring).

The image can be a `PIL.Image` (converted to RGB internally) or a
`(height, width, 3)` uint8 numpy array, and its dimensions must match the
mode's resolution — resize with
`image.resize((mode.image_width, mode.image_height))`. The transmission
includes the calibration header, so decoders can detect the mode.

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
