use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, AtomicU32, Ordering}};
use super::MAX_SECONDS;

pub(crate) struct Audio {
    /// Only the audio callback touches this until the stream is dropped.
    samples: Mutex<Vec<f32>>,
    // Outside the lock: the capture loop polls these every 10 ms and must
    // never make the real-time callback's `try_lock` drop a buffer.
    failed: AtomicBool,
    full: AtomicBool,
}
impl Audio {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: Mutex::new(Vec::with_capacity(capacity)),
            failed: AtomicBool::new(false),
            full: AtomicBool::new(false),
        }
    }
}
/// Loudest RMS since the UI last read it. Non-negative `f32` bit patterns sort
/// like their values, so `fetch_max` keeps the peak without a lock.
pub(crate) fn meter<T: cpal::Sample>(data: &[T], channels: usize, level: &AtomicU32)
where
    f32: cpal::FromSample<T>,
{
    let frames = data.len() / channels.max(1);
    if frames == 0 {
        return;
    }
    let energy = data
        .chunks_exact(channels)
        .map(|frame| {
            let s = frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32;
            s * s
        })
        .sum::<f32>();
    let rms = (energy / frames as f32).sqrt();
    if rms.is_finite() {
        level.fetch_max(rms.to_bits(), Ordering::Relaxed);
    }
}
pub(crate) fn append<T: cpal::Sample>(data: &[T], channels: usize, rate: u32, a: &Audio)
where
    f32: cpal::FromSample<T>,
{
    if let Ok(mut samples) = a.samples.try_lock() {
        for frame in data.chunks_exact(channels) {
            if samples.len() >= rate as usize * MAX_SECONDS {
                a.full.store(true, Ordering::Release);
                break;
            }
            samples.push(frame.iter().map(|s| s.to_sample::<f32>()).sum::<f32>() / channels as f32);
        }
    }
}
/// A microphone the user can choose: a stable identifier and a display name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDevice {
    pub id: String,
    pub name: String,
}

/// Every input device the default host reports. Enumeration can block on the
/// audio server, so call it off the UI thread.
pub fn input_devices() -> Vec<InputDevice> {
    let Ok(devices) = cpal::default_host().input_devices() else {
        return Vec::new();
    };
    let mut devices: Vec<_> = devices
        .filter_map(|device| {
            Some(InputDevice {
                id: device.id().ok()?.to_string(),
                name: device.description().ok()?.name().to_owned(),
            })
        })
        .collect();
    devices.dedup_by(|a, b| a.id == b.id);
    devices
}

/// The id of the device the system currently records from by default.
pub fn default_input_device() -> Option<String> {
    Some(
        cpal::default_host()
            .default_input_device()?
            .id()
            .ok()?
            .to_string(),
    )
}

/// The chosen device when it is connected, the system default otherwise.
fn input_device(id: Option<&str>) -> Option<cpal::Device> {
    let host = cpal::default_host();
    id.and_then(|id| id.parse::<cpal::DeviceId>().ok())
        .and_then(|id| host.device_by_id(&id))
        .or_else(|| host.default_input_device())
}

pub struct Capture {
    stream: Option<cpal::Stream>,
    audio: Arc<Audio>,
    rate: u32,
}
impl Capture {
    pub fn start(device: Option<&str>, level: Arc<AtomicU32>) -> Result<Self> {
        let device = input_device(device).context("No microphone available")?;
        let config = device.default_input_config()?;
        let rate = config.sample_rate();
        let channels = config.channels() as usize;
        let audio = Arc::new(Audio::with_capacity(rate as usize * MAX_SECONDS));
        let a = audio.clone();
        let e = audio.clone();
        let err = move |_| e.failed.store(true, Ordering::Release);
        // Preserve the device's native configuration; convert every CPAL
        // sample representation through the same bounded mono callback.
        macro_rules! stream {
            ($sample:ty) => {
                device.build_input_stream(
                    &config.into(),
                    move |d: &[$sample], _| {
                        meter(d, channels, &level);
                        append(d, channels, rate, &a)
                    },
                    err,
                    None,
                )?
            };
        }
        let stream = match config.sample_format() {
            cpal::SampleFormat::I8 => stream!(i8),
            cpal::SampleFormat::I16 => stream!(i16),
            cpal::SampleFormat::I24 => stream!(cpal::I24),
            cpal::SampleFormat::I32 => stream!(i32),
            cpal::SampleFormat::I64 => stream!(i64),
            cpal::SampleFormat::U8 => stream!(u8),
            cpal::SampleFormat::U16 => stream!(u16),
            cpal::SampleFormat::U24 => stream!(cpal::U24),
            cpal::SampleFormat::U32 => stream!(u32),
            cpal::SampleFormat::U64 => stream!(u64),
            cpal::SampleFormat::F32 => stream!(f32),
            cpal::SampleFormat::F64 => stream!(f64),
            _ => bail!("Unsupported microphone format"),
        };
        stream.play()?;
        Ok(Self {
            stream: Some(stream),
            audio,
            rate,
        })
    }
    pub fn ended(&self) -> bool {
        self.audio.full.load(Ordering::Acquire) || self.audio.failed.load(Ordering::Acquire)
    }
    pub fn finish(mut self) -> Result<(Vec<f32>, u32)> {
        self.stream.take();
        if self.audio.failed.load(Ordering::Acquire) {
            bail!("Microphone disconnected. Your draft is safe.")
        }
        let mut samples = self.audio.samples.lock().unwrap_or_else(|e| e.into_inner());
        Ok((std::mem::take(&mut *samples), self.rate))
    }
}
