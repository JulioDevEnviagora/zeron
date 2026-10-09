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
