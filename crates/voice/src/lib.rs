//! Desktop-local Parakeet v3. No engine, document, RPC or audio persistence.
use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
use parakeet_rs::{ParakeetTDT, Transcriber};
use sha2::{Digest, Sha256};
use std::{
    io::Write,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, Ordering},
    },
};

mod resample;

pub const MAX_SECONDS: usize = 60;
#[derive(serde::Deserialize)]
pub struct Artifact {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}
#[derive(serde::Deserialize)]
pub struct Manifest {
    pub repository: String,
    pub revision: String,
    pub files: Vec<Artifact>,
}
pub fn manifest() -> Manifest {
    serde_json::from_str(include_str!("../model.json")).expect("pinned model manifest")
}
pub fn download_size() -> u64 {
    manifest().files.iter().map(|f| f.size).sum()
}
pub fn installed(dir: &Path) -> bool {
    manifest()
        .files
        .iter()
        .all(|f| std::fs::metadata(dir.join(&f.name)).is_ok_and(|m| m.len() == f.size))
        && std::fs::read_to_string(dir.join("verified")).is_ok_and(|r| r == manifest().revision)
}

/// Called only on a worker. Temporary files never establish readiness.
pub fn download(dir: &Path, cancel: &AtomicBool, mut progress: impl FnMut(u64)) -> Result<()> {
    tokio::runtime::Builder::new_current_thread().enable_all().build()?.block_on(async {
        std::fs::create_dir_all(dir)?;
        let client = reqwest::Client::builder().connect_timeout(std::time::Duration::from_secs(20)).build()?;
        let m=manifest(); let mut total=0;
        for file in m.files {
            if cancel.load(Ordering::Acquire) { bail!("Download cancelled") }
            let staging=dir.join(format!("{}.part",file.name));
            let result=async {
                let mut response=tokio::select! {
                    _=cancelled(cancel)=>bail!("Download cancelled"),
                    response=tokio::time::timeout(std::time::Duration::from_secs(30),client.get(format!("https://huggingface.co/{}/resolve/{}/{}",m.repository,m.revision,file.name)).send())=>response??.error_for_status()?,
                };
                let mut out=std::fs::File::create(&staging)?;let mut hash=Sha256::new();let mut size=0;
                loop {
                    let chunk=tokio::select! {
                        _=cancelled(cancel)=>bail!("Download cancelled"),
                        chunk=tokio::time::timeout(std::time::Duration::from_secs(30),response.chunk())=>chunk??,
                    };
                    let Some(chunk)=chunk else {break};
                    size+=chunk.len() as u64;if size>file.size {bail!("Unexpected model size")}
                    hash.update(&chunk);out.write_all(&chunk)?;total+=chunk.len() as u64;progress(total);
                }
                if size!=file.size || format!("{:x}",hash.finalize())!=file.sha256 {bail!("Model checksum verification failed")}
                out.sync_all()?;std::fs::rename(&staging,dir.join(&file.name))?;Ok::<_,anyhow::Error>(())
            }.await;
            if result.is_err() {let _=std::fs::remove_file(staging);} result?;
        }
        if cancel.load(Ordering::Acquire) {bail!("Download cancelled")}
        std::fs::write(dir.join("verified"),manifest().revision)?;Ok(())
    })
}
async fn cancelled(cancel: &AtomicBool) {
    while !cancel.load(Ordering::Acquire) {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
    }
}

#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
pub struct Recognizer(ParakeetTDT);
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub struct Recognizer;
impl Recognizer {
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    pub fn load(dir: &Path) -> Result<Self> {
        // Verify before handing bytes to the native runtime, including after restart.
        for f in manifest().files {
            let mut input = std::fs::File::open(dir.join(&f.name))?;
            let mut hash = Sha256::new();
            std::io::copy(&mut input, &mut hash)?;
            if format!("{:x}", hash.finalize()) != f.sha256 {
                bail!("Model is damaged. Remove it in Settings and download again.")
            }
        }
        Ok(Self(ParakeetTDT::from_pretrained(dir, None).map_err(
            |_| anyhow::anyhow!("Could not load Parakeet v3"),
        )?))
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    pub fn load(_dir: &Path) -> Result<Self> {
        bail!(
            "Local dictation is unavailable on macOS Intel: ONNX Runtime has no prebuilt for x86_64-apple-darwin."
        )
    }
    pub fn transcribe(&mut self, samples: Vec<f32>, rate: u32) -> Result<String> {
        if !(8_000..=192_000).contains(&rate) {
            bail!("Unsupported microphone sample rate")
        }
        if samples.len() > rate as usize * MAX_SECONDS {
            bail!("Recording exceeds one minute")
        }
        if samples.len() < rate as usize / 5 || samples.iter().all(|s| s.abs() < 0.0001) {
            return Ok(String::new());
        }
        self.transcribe_model(samples, rate)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
    fn transcribe_model(&mut self, samples: Vec<f32>, rate: u32) -> Result<String> {
        let samples = resample::for_model(samples, rate)?;
        self.0
            .transcribe_samples(samples, resample::MODEL_RATE, 1, None)
            .map(|r| r.text)
            .map_err(|_| anyhow::anyhow!("Could not transcribe this recording"))
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    fn transcribe_model(&mut self, _samples: Vec<f32>, _rate: u32) -> Result<String> {
        bail!(
            "Local dictation is unavailable on macOS Intel: ONNX Runtime has no prebuilt for x86_64-apple-darwin."
        )
    }
}
