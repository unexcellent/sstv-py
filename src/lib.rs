//! Python bindings for the `sstv` crate's decoder.

use numpy::{PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

/// An SSTV transmission mode.
///
/// Pass a specific mode to `decode` when the transmission is known (or its
/// header is missing), or `Mode.AUTO` to detect each image's mode from the
/// VIS code in its header.
///
/// The `image_width` and `image_height` properties give the fixed resolution
/// of every image transmitted in that mode.
#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum Mode {
    /// Detect the mode from the transmission's header.
    AUTO,
    /// A 320x256 colour image in a 110 second transmission.
    SCOTTIE_1,
    /// A 320x256 colour image in a 71 second transmission.
    SCOTTIE_2,
    /// A 320x256 colour image in a 269 second transmission.
    SCOTTIE_DX,
    /// A 320x256 colour image in a 114 second transmission.
    MARTIN_1,
    /// A 320x256 colour image in a 58 second transmission.
    MARTIN_2,
    /// A 320x240 colour image in a 36 second transmission.
    ROBOT_36,
    /// A 320x240 colour image in a 72 second transmission.
    ROBOT_72,
    /// A 320x256 colour image in a 182 second transmission.
    WRASSE_SC2_180,
    /// A 640x496 colour image in a 203 second transmission.
    PASOKON_P3,
    /// A 640x496 colour image in a 305 second transmission.
    PASOKON_P5,
    /// A 640x496 colour image in a 406 second transmission.
    PASOKON_P7,
    /// A 320x256 colour image in a 50 second transmission.
    PD_50,
    /// A 320x256 colour image in a 90 second transmission.
    PD_90,
    /// A 640x496 colour image in a 126 second transmission.
    PD_120,
    /// A 512x400 colour image in a 161 second transmission.
    PD_160,
    /// A 640x496 colour image in a 187 second transmission.
    PD_180,
    /// A 640x496 colour image in a 248 second transmission.
    PD_240,
    /// An 800x616 colour image in a 289 second transmission.
    PD_290,
}

impl From<Mode> for sstv::Mode {
    fn from(mode: Mode) -> Self {
        match mode {
            Mode::AUTO => Self::Auto,
            Mode::SCOTTIE_1 => Self::Scottie1,
            Mode::SCOTTIE_2 => Self::Scottie2,
            Mode::SCOTTIE_DX => Self::ScottieDx,
            Mode::MARTIN_1 => Self::Martin1,
            Mode::MARTIN_2 => Self::Martin2,
            Mode::ROBOT_36 => Self::Robot36,
            Mode::ROBOT_72 => Self::Robot72,
            Mode::WRASSE_SC2_180 => Self::WrasseSc2180,
            Mode::PASOKON_P3 => Self::PasokonP3,
            Mode::PASOKON_P5 => Self::PasokonP5,
            Mode::PASOKON_P7 => Self::PasokonP7,
            Mode::PD_50 => Self::Pd50,
            Mode::PD_90 => Self::Pd90,
            Mode::PD_120 => Self::Pd120,
            Mode::PD_160 => Self::Pd160,
            Mode::PD_180 => Self::Pd180,
            Mode::PD_240 => Self::Pd240,
            Mode::PD_290 => Self::Pd290,
        }
    }
}

impl From<sstv::Mode> for Mode {
    fn from(mode: sstv::Mode) -> Self {
        match mode {
            sstv::Mode::Auto => Self::AUTO,
            sstv::Mode::Scottie1 => Self::SCOTTIE_1,
            sstv::Mode::Scottie2 => Self::SCOTTIE_2,
            sstv::Mode::ScottieDx => Self::SCOTTIE_DX,
            sstv::Mode::Martin1 => Self::MARTIN_1,
            sstv::Mode::Martin2 => Self::MARTIN_2,
            sstv::Mode::Robot36 => Self::ROBOT_36,
            sstv::Mode::Robot72 => Self::ROBOT_72,
            sstv::Mode::WrasseSc2180 => Self::WRASSE_SC2_180,
            sstv::Mode::PasokonP3 => Self::PASOKON_P3,
            sstv::Mode::PasokonP5 => Self::PASOKON_P5,
            sstv::Mode::PasokonP7 => Self::PASOKON_P7,
            sstv::Mode::Pd50 => Self::PD_50,
            sstv::Mode::Pd90 => Self::PD_90,
            sstv::Mode::Pd120 => Self::PD_120,
            sstv::Mode::Pd160 => Self::PD_160,
            sstv::Mode::Pd180 => Self::PD_180,
            sstv::Mode::Pd240 => Self::PD_240,
            sstv::Mode::Pd290 => Self::PD_290,
            // `sstv::Mode` is non-exhaustive; modes added to the crate before
            // a binding exists here fall back to AUTO.
            _ => Self::AUTO,
        }
    }
}

#[pymethods]
impl Mode {
    /// The horizontal resolution in pixels of images transmitted in this mode.
    ///
    /// Raises `ValueError` for `Mode.AUTO`, which has no fixed resolution.
    #[getter]
    fn image_width(&self) -> PyResult<u32> {
        if matches!(self, Self::AUTO) {
            return Err(PyValueError::new_err(
                "Mode.AUTO has no fixed resolution; the mode is detected per image while decoding",
            ));
        }
        Ok(sstv::Mode::from(*self).image_width())
    }

    /// The vertical resolution in pixels of images transmitted in this mode.
    ///
    /// Raises `ValueError` for `Mode.AUTO`, which has no fixed resolution.
    #[getter]
    fn image_height(&self) -> PyResult<u32> {
        if matches!(self, Self::AUTO) {
            return Err(PyValueError::new_err(
                "Mode.AUTO has no fixed resolution; the mode is detected per image while decoding",
            ));
        }
        Ok(sstv::Mode::from(*self).image_height())
    }
}

/// One decoded image, held until the GIL is re-acquired and it can be turned
/// into a Pillow image.
struct Decoded {
    mode: Mode,
    width: usize,
    height: usize,
    complete: bool,
    /// RGB bytes in row-major order, `width * height * 3` of them.
    pixels: Vec<u8>,
}

impl From<sstv::DecodedImage> for Decoded {
    fn from(decoded: sstv::DecodedImage) -> Self {
        let mut pixels = Vec::with_capacity(decoded.pixels().len() * 3);
        for pixel in decoded.pixels() {
            pixels.extend_from_slice(&[pixel.red(), pixel.green(), pixel.blue()]);
        }
        Self {
            mode: decoded.mode().into(),
            width: decoded.width(),
            height: decoded.height(),
            complete: decoded.complete(),
            pixels,
        }
    }
}

impl Decoded {
    /// The image as a Pillow `Image` in RGB mode, with the decode metadata
    /// stored under the `sstv_mode` and `sstv_complete` keys of its `info`
    /// dict.
    fn into_pil<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let image = py.import("PIL.Image")?.call_method1(
            "frombytes",
            ("RGB", (self.width, self.height), PyBytes::new(py, &self.pixels)),
        )?;
        let info = image.getattr("info")?;
        info.set_item("sstv_mode", self.mode)?;
        info.set_item("sstv_complete", self.complete)?;
        Ok(image)
    }
}

/// Scale a float sample in [-1.0, 1.0] to the i16 range, clamping outliers.
fn float_sample(sample: f64) -> i16 {
    (sample.clamp(-1.0, 1.0) * f64::from(i16::MAX)) as i16
}

/// Convert the accepted Python sample representations to owned i16 samples.
fn samples_to_vec(samples: &Bound<'_, PyAny>) -> PyResult<Vec<i16>> {
    if let Ok(array) = samples.cast::<PyUntypedArray>() {
        if array.ndim() != 1 {
            return Err(PyTypeError::new_err(format!(
                "samples must be one-dimensional, got a {}-dimensional array; \
                 pass a single channel, e.g. samples[:, 0]",
                array.ndim()
            )));
        }
        if let Ok(array) = samples.extract::<PyReadonlyArray1<i16>>() {
            return Ok(array.as_array().to_vec());
        }
        if let Ok(array) = samples.extract::<PyReadonlyArray1<f32>>() {
            return Ok(array
                .as_array()
                .iter()
                .map(|&sample| float_sample(f64::from(sample)))
                .collect());
        }
        if let Ok(array) = samples.extract::<PyReadonlyArray1<f64>>() {
            return Ok(array.as_array().iter().map(|&sample| float_sample(sample)).collect());
        }
        return Err(PyTypeError::new_err(format!(
            "unsupported samples dtype {}; expected int16, float32, or float64",
            array.dtype()
        )));
    }
    samples.extract::<Vec<i16>>().map_err(|_| {
        PyTypeError::new_err(
            "samples must be a one-dimensional numpy array of int16, float32, or float64, \
             or a sequence of ints within the int16 range",
        )
    })
}

/// Decode every SSTV image contained in a stream of audio samples.
///
/// Args:
///     samples: The audio as a one-dimensional numpy array — dtype int16, or
///         float32/float64 with values in [-1.0, 1.0] (scaled to 16-bit
///         internally). A plain sequence of ints in the int16 range is also
///         accepted. Multi-channel audio must be reduced to one channel
///         first, e.g. ``samples[:, 0]``.
///     sample_rate: The sample rate in Hz, greater than zero.
///     mode: The transmission's mode. With ``Mode.AUTO`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. ``Mode.AUTO`` cannot be detected
///         without a header and decodes as ``Mode.ROBOT_36``.
///
/// Returns:
///     One RGB ``PIL.Image`` per image found, in order of appearance. Audio
///     that contains no SSTV transmission yields an empty list; audio with
///     several transmissions — with or without gaps between them — yields all
///     of them.
///
///     Decode metadata is stored in each image's ``info`` dict:
///
///     - ``info["sstv_mode"]``: the ``Mode`` the image was transmitted in
///       (useful with ``Mode.AUTO`` to learn what was detected).
///     - ``info["sstv_complete"]``: ``False`` if the signal ended before the
///       image's last scanline; the rows the signal did not carry are black.
///
///     Note that Pillow does not carry ``info`` through operations like
///     ``crop`` or ``resize`` — read the metadata before transforming the
///     image.
///
/// Raises:
///     TypeError: If ``samples`` is not an accepted array or sequence type.
///     ValueError: If ``sample_rate`` is zero.
///
/// Example:
///     >>> import sstv
///     >>> images = sstv.decode(samples, sample_rate=48000)
///     >>> [(img.info["sstv_mode"], img.info["sstv_complete"]) for img in images]
///     [(Mode.ROBOT_36, True)]
///     >>> images[0].size
///     (320, 240)
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, *, mode = Mode::AUTO, header = true))]
fn decode<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    sample_rate: u32,
    mode: Mode,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if sample_rate == 0 {
        return Err(PyValueError::new_err("sample_rate must be greater than zero"));
    }
    let samples = samples_to_vec(samples)?;
    let mode = sstv::Mode::from(mode);

    // Decoding minutes of audio is CPU-bound; let other Python threads run.
    let decoded = py.detach(move || {
        run_decoder(
            sstv::Decoder::from_samples(mode, samples.into_iter(), sample_rate),
            header,
        )
    });
    decoded.into_iter().map(|image| image.into_pil(py)).collect()
}

fn run_decoder<I: Iterator<Item = i16>>(decoder: sstv::Decoder<I>, header: bool) -> Vec<Decoded> {
    let decoder = if header { decoder } else { decoder.without_header() };
    decoder.images().map(Decoded::from).collect()
}

/// Read an argument accepted as audio data — a path, bytes, or a binary
/// file-like object — into raw bytes. `name` is the parameter name used in
/// error messages.
fn audio_to_bytes(audio: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = audio.cast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(bytes) = audio.cast::<PyByteArray>() {
        return Ok(bytes.to_vec());
    }
    if let Ok(path) = audio.extract::<std::path::PathBuf>() {
        return Ok(std::fs::read(path)?);
    }
    if let Ok(read) = audio.getattr("read") {
        return read.call0()?.extract().map_err(|_| {
            PyTypeError::new_err(format!(
                "{name}.read() must return bytes; open the file in binary mode"
            ))
        });
    }
    Err(PyTypeError::new_err(format!(
        "{name} must be a path, bytes, or a binary file-like object",
    )))
}

/// Decode every SSTV image contained in a WAV recording.
///
/// The sample rate is read from the WAV header. Only the first channel of
/// multi-channel audio is used; integer samples of any bit depth and float
/// samples are converted to 16 bit. Data cut short relative to the length
/// declared in the header — common in recordings whose writer was
/// interrupted — is decoded up to the cut.
///
/// Args:
///     wav: The recording as a path (``str`` or ``os.PathLike``), in-memory
///         WAV data (``bytes`` or ``bytearray``), or a binary file-like
///         object with a ``read()`` method.
///     mode: The transmission's mode. With ``Mode.AUTO`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. ``Mode.AUTO`` cannot be detected
///         without a header and decodes as ``Mode.ROBOT_36``.
///
/// Returns:
///     One RGB ``PIL.Image`` per image found, in order of appearance, with
///     the decode metadata in each image's ``info`` dict — exactly as
///     described for ``decode``.
///
/// Raises:
///     TypeError: If ``wav`` is not an accepted type.
///     ValueError: If the WAV data is malformed.
///     OSError: If ``wav`` is a path that cannot be read.
///
/// Example:
///     >>> import sstv
///     >>> images = sstv.decode_wav("recording.wav")
///     >>> [img.info["sstv_mode"] for img in images]
///     [Mode.ROBOT_36]
#[pyfunction]
#[pyo3(signature = (wav, *, mode = Mode::AUTO, header = true))]
fn decode_wav<'py>(
    py: Python<'py>,
    wav: &Bound<'py, PyAny>,
    mode: Mode,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let wav = audio_to_bytes(wav, "wav")?;
    let mode = sstv::Mode::from(mode);

    let decoded = py.detach(move || {
        sstv::Decoder::from_wav(mode, &wav)
            .map(|decoder| run_decoder(decoder, header))
            .map_err(|error| PyValueError::new_err(format!("malformed WAV data: {error}")))
    })?;
    decoded.into_iter().map(|image| image.into_pil(py)).collect()
}

/// Decode every SSTV image contained in an MP3 recording.
///
/// The sample rate is read from the MP3 frames. Only the first channel of
/// multi-channel audio is used.
///
/// Args:
///     mp3: The recording as a path (``str`` or ``os.PathLike``), in-memory
///         MP3 data (``bytes`` or ``bytearray``), or a binary file-like
///         object with a ``read()`` method.
///     mode: The transmission's mode. With ``Mode.AUTO`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. ``Mode.AUTO`` cannot be detected
///         without a header and decodes as ``Mode.ROBOT_36``.
///
/// Returns:
///     One RGB ``PIL.Image`` per image found, in order of appearance, with
///     the decode metadata in each image's ``info`` dict — exactly as
///     described for ``decode``.
///
/// Raises:
///     TypeError: If ``mp3`` is not an accepted type.
///     ValueError: If the data contains no decodable MP3 frames.
///     OSError: If ``mp3`` is a path that cannot be read.
///
/// Example:
///     >>> import sstv
///     >>> images = sstv.decode_mp3("recording.mp3")
///     >>> [img.info["sstv_mode"] for img in images]
///     [Mode.ROBOT_36]
#[pyfunction]
#[pyo3(signature = (mp3, *, mode = Mode::AUTO, header = true))]
fn decode_mp3<'py>(
    py: Python<'py>,
    mp3: &Bound<'py, PyAny>,
    mode: Mode,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mp3 = audio_to_bytes(mp3, "mp3")?;
    let mode = sstv::Mode::from(mode);

    // This mirrors the sstv crate's `Decoder::from_mp3`, kept local so the
    // crate's `mp3` feature (and with it the LAME encoder) stays out of the
    // dependency tree.
    let decoded = py.detach(move || -> PyResult<Vec<Decoded>> {
        let mut frames = minimp3::Decoder::new(std::io::Cursor::new(mp3));
        let mut samples: Vec<i16> = Vec::new();
        let mut sample_rate = 0u32;
        loop {
            match frames.next_frame() {
                Ok(frame) => {
                    sample_rate = frame.sample_rate as u32;
                    let channels = frame.channels.max(1);
                    samples.extend(frame.data.iter().step_by(channels));
                }
                Err(minimp3::Error::Eof) => break,
                Err(error) => {
                    return Err(PyValueError::new_err(format!("malformed MP3 data: {error}")));
                }
            }
        }
        if sample_rate == 0 {
            return Err(PyValueError::new_err(
                "no MP3 frames found; is this an MP3 recording?",
            ));
        }
        Ok(run_decoder(
            sstv::Decoder::from_samples(mode, samples.into_iter(), sample_rate),
            header,
        ))
    })?;
    decoded.into_iter().map(|image| image.into_pil(py)).collect()
}

#[pymodule]
fn _sstv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mode>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_wav, m)?)?;
    m.add_function(wrap_pyfunction!(decode_mp3, m)?)?;
    Ok(())
}
