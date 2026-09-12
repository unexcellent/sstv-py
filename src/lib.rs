//! Python bindings for the `sstv` crate's decoder.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

/// An SSTV transmission mode.
///
/// Pass a specific mode to `decode` when the transmission is known (or its
/// header is missing); with the default of ``None``, each image's mode is
/// detected from the VIS code in its header.
///
/// The `image_width` and `image_height` properties give the fixed resolution
/// of every image transmitted in that mode.
#[pyclass(frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Mode {
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
            // `sstv::Mode` is non-exhaustive, and decoded images never carry
            // `Auto`; the crate detects `Auto` as `Robot36`, so modes added
            // to the crate before a binding exists here fall back to that.
            sstv::Mode::Auto | _ => Self::ROBOT_36,
        }
    }
}

#[pymethods]
impl Mode {
    /// The horizontal resolution in pixels of images transmitted in this mode.
    #[getter]
    fn image_width(&self) -> u32 {
        sstv::Mode::from(*self).image_width()
    }

    /// The vertical resolution in pixels of images transmitted in this mode.
    #[getter]
    fn image_height(&self) -> u32 {
        sstv::Mode::from(*self).image_height()
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
///     mode: The transmission's mode. With ``None`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. The mode cannot be detected without
///         a header; if ``mode`` is ``None``, ``Mode.ROBOT_36`` is assumed.
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
///       (useful with ``mode=None`` to learn what was detected).
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
#[pyo3(signature = (samples, sample_rate, *, mode = None, header = true))]
fn decode<'py>(
    py: Python<'py>,
    samples: &Bound<'py, PyAny>,
    sample_rate: u32,
    mode: Option<Mode>,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if sample_rate == 0 {
        return Err(PyValueError::new_err("sample_rate must be greater than zero"));
    }
    let samples = samples_to_vec(samples)?;
    let mode = mode.map_or(sstv::Mode::Auto, sstv::Mode::from);

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
///     mode: The transmission's mode. With ``None`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. The mode cannot be detected without
///         a header; if ``mode`` is ``None``, ``Mode.ROBOT_36`` is assumed.
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
///     >>> images = sstv.decode_from_wav("recording.wav")
///     >>> [img.info["sstv_mode"] for img in images]
///     [Mode.ROBOT_36]
#[pyfunction]
#[pyo3(signature = (wav, *, mode = None, header = true))]
fn decode_from_wav<'py>(
    py: Python<'py>,
    wav: &Bound<'py, PyAny>,
    mode: Option<Mode>,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let wav = audio_to_bytes(wav, "wav")?;
    let mode = mode.map_or(sstv::Mode::Auto, sstv::Mode::from);

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
///     mode: The transmission's mode. With ``None`` (the default), each
///         image's mode is detected from the VIS code in its header.
///     header: If ``False``, assume the samples begin directly at the first
///         scanline and skip searching for a header. Use this when the signal
///         carries no detectable header. The mode cannot be detected without
///         a header; if ``mode`` is ``None``, ``Mode.ROBOT_36`` is assumed.
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
///     >>> images = sstv.decode_from_mp3("recording.mp3")
///     >>> [img.info["sstv_mode"] for img in images]
///     [Mode.ROBOT_36]
#[pyfunction]
#[pyo3(signature = (mp3, *, mode = None, header = true))]
fn decode_from_mp3<'py>(
    py: Python<'py>,
    mp3: &Bound<'py, PyAny>,
    mode: Option<Mode>,
    header: bool,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mp3 = audio_to_bytes(mp3, "mp3")?;
    let mode = mode.map_or(sstv::Mode::Auto, sstv::Mode::from);

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

/// Read the accepted image argument types — a PIL image or a
/// `(height, width, 3)` uint8 numpy array — into pixels, validating the
/// dimensions against the mode's resolution.
fn pixels_from_image(
    image: &Bound<'_, PyAny>,
    mode: sstv::Mode,
) -> PyResult<Vec<sstv::RgbPixel>> {
    let expected = (mode.image_width() as usize, mode.image_height() as usize);

    if image.cast::<PyUntypedArray>().is_ok() {
        let array = image.extract::<PyReadonlyArray3<u8>>().map_err(|_| {
            PyTypeError::new_err("image array must be uint8 with shape (height, width, 3)")
        })?;
        let shape = array.shape().to_vec();
        if shape != [expected.1, expected.0, 3] {
            return Err(PyValueError::new_err(format!(
                "image shape ({}, {}, {}) does not match the mode's resolution \
                 ({}, {}, 3)",
                shape[0], shape[1], shape[2], expected.1, expected.0,
            )));
        }
        let array = array.as_array();
        return Ok(array
            .rows()
            .into_iter()
            .map(|rgb| sstv::RgbPixel::new(rgb[0], rgb[1], rgb[2]))
            .collect());
    }

    let py = image.py();
    if image.is_instance(&py.import("PIL.Image")?.getattr("Image")?)? {
        let size: (usize, usize) = image.getattr("size")?.extract()?;
        if size != expected {
            return Err(PyValueError::new_err(format!(
                "image size {}x{} does not match the mode's resolution {}x{}; \
                 resize with image.resize((mode.image_width, mode.image_height))",
                size.0, size.1, expected.0, expected.1,
            )));
        }
        let rgb = image.call_method1("convert", ("RGB",))?;
        let bytes: Vec<u8> = rgb.call_method0("tobytes")?.extract()?;
        return Ok(bytes
            .chunks_exact(3)
            .map(|rgb| sstv::RgbPixel::new(rgb[0], rgb[1], rgb[2]))
            .collect());
    }

    Err(PyTypeError::new_err(
        "image must be a PIL.Image or a (height, width, 3) uint8 numpy array",
    ))
}

/// Encode an image into the raw audio samples of an SSTV transmission.
///
/// The transmission includes the calibration header carrying the mode's VIS
/// code, so the result decodes with ``decode(samples, sample_rate)`` without
/// specifying the mode.
///
/// Args:
///     image: The image to transmit, as a ``PIL.Image`` (converted to RGB
///         internally) or a ``(height, width, 3)`` uint8 numpy array of RGB
///         values. The dimensions must match the mode's resolution exactly;
///         resize beforehand with
///         ``image.resize((mode.image_width, mode.image_height))``.
///     mode: The SSTV mode to transmit in.
///     sample_rate: The sample rate of the produced audio in Hz, greater
///         than zero.
///
/// Returns:
///     The transmission as a one-dimensional int16 numpy array of PCM
///     samples, ready for an audio device or further processing.
///
/// Raises:
///     TypeError: If ``image`` is not an accepted type.
///     ValueError: If the image dimensions do not match the mode's
///         resolution, or ``sample_rate`` is zero.
///
/// Example:
///     >>> import sstv
///     >>> from PIL import Image
///     >>> image = Image.open("photo.png").resize((320, 240))
///     >>> samples = sstv.encode(image, sstv.Mode.ROBOT_36)
///     >>> sstv.decode(samples, 48000)[0].info["sstv_mode"]
///     Mode.ROBOT_36
#[pyfunction]
#[pyo3(signature = (image, mode, sample_rate = 48_000))]
fn encode<'py>(
    py: Python<'py>,
    image: &Bound<'py, PyAny>,
    mode: Mode,
    sample_rate: u32,
) -> PyResult<Bound<'py, PyArray1<i16>>> {
    if sample_rate == 0 {
        return Err(PyValueError::new_err("sample_rate must be greater than zero"));
    }
    let mode = sstv::Mode::from(mode);
    let pixels = pixels_from_image(image, mode)?;

    // The crate's `Encoder` is not `Send`, so it must be built inside detach.
    let samples: Vec<i16> = py.detach(move || {
        // expect: `pixels_from_image` guarantees a full image of pixels, so
        // the encoder's only error, `EmptyImage`, cannot occur.
        #[allow(clippy::expect_used)]
        let encoder = sstv::Encoder::new(mode, pixels.into_iter())
            .expect("a dimension-checked image is never empty");
        sstv::Synthesizer::new(encoder, sample_rate).collect()
    });
    Ok(PyArray1::from_vec(py, samples))
}

#[pymodule]
fn _sstv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Mode>()?;
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(decode_from_wav, m)?)?;
    m.add_function(wrap_pyfunction!(decode_from_mp3, m)?)?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    Ok(())
}
