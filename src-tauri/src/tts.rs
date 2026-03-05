// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 NeuroSkill.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 only.
//! Calibration TTS — two backends behind a common Tauri-command façade.
//!
//! ## Backends
//!
//! | Backend   | Crate       | Thread name     | Model                                    |
//! |-----------|-------------|-----------------|------------------------------------------|
//! | KittenTTS | `kittentts` | `skill-tts`     | `KittenML/kitten-tts-mini-0.8`           |
//! | NeuTTS    | `neutts`    | `skill-neutts`  | `neuphonic/neutts-nano-q4-gguf` (config) |
//!
//! ## Architecture
//!
//! Both backends follow the same pattern: all blocking work (model download,
//! inference, audio playback) runs on a single dedicated OS thread.  Tokio
//! async commands send work items through an `mpsc::sync_channel` and await a
//! `tokio::sync::oneshot` for the result.
//!
//! ## Config statics
//!
//! NeuTTS config is mirrored into module-level statics so that async Tauri
//! commands do not need `tauri::State` (which would require returning `Result`
//! and break the existing two-arg `tts_speak` call in `ws_commands.rs`).
//! Call [`neutts_apply_config`] on startup and whenever settings change.
//!
//! ## Preset voices
//!
//! NeuTTS ships five bundled reference voices in `neutts-rs/samples/`.
//! Each is a pre-encoded `.npy` file so the encoder is never needed for them.
//!
//! | Name      | Language | Gender |
//! |-----------|----------|--------|
//! | jo        | en-us    | ♀      |
//! | dave      | en-us    | ♂      |
//! | greta     | de       | ♀      |
//! | juliette  | fr-fr    | ♀      |
//! | mateo     | es       | ♂      |
//!
//! ## Engine lifecycle
//!
//! `tts_init`   — pre-download + warm-up; emits `"tts-progress"` events.
//! `tts_unload` — drop the in-memory model (free GPU/RAM); engine goes idle.
//! `tts_speak`  — synthesise + play; lazy-inits if engine is unloaded.

use std::sync::{OnceLock, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};

use kittentts::{download::{self, LoadProgress}, KittenTTS};
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, MixerDeviceSink, Player};
use tauri::Emitter;
use tokio::sync::oneshot;

// ─── Logging ─────────────────────────────────────────────────────────────────

static TTS_LOGGING: AtomicBool = AtomicBool::new(false);

pub fn set_logging(enabled: bool) {
    TTS_LOGGING.store(enabled, Ordering::Relaxed);
}

#[inline]
fn tts_log(msg: &str) {
    if TTS_LOGGING.load(Ordering::Relaxed) {
        eprintln!("[tts] {msg}");
    }
}

// ─── Shared constants ─────────────────────────────────────────────────────────

const TAIL_SILENCE_SECS: f32 = 1.0;

/// Absolute path to the bundled NeuTTS sample voices directory.
/// Resolved at compile time from the crate manifest directory.
const NEUTTS_SAMPLES_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neutts-rs/samples");

// ─── Shared progress event ────────────────────────────────────────────────────

#[derive(Clone, serde::Serialize)]
struct TtsProgressEvent {
    phase: &'static str,
    step:  u32,
    total: u32,
    label: String,
}

impl TtsProgressEvent {
    fn step(step: u32, total: u32, label: impl Into<String>) -> Self {
        Self { phase: "step", step, total, label: label.into() }
    }
    fn ready() -> Self {
        Self { phase: "ready", step: 0, total: 0, label: String::new() }
    }
    fn unloaded() -> Self {
        Self { phase: "unloaded", step: 0, total: 0, label: String::new() }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ── KittenTTS backend ─────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

const HF_REPO:            &str = "KittenML/kitten-tts-mini-0.8";
const VOICE_DEFAULT:      &str = "Jasper";
const SPEED:              f32  = 1.0;
const KITTEN_SAMPLE_RATE: u32  = kittentts::SAMPLE_RATE;

// ─── KittenTTS state ──────────────────────────────────────────────────────────

/// Voice names from the loaded model.  Populated on first Init; cached forever.
static AVAILABLE_VOICES: OnceLock<Vec<String>> = OnceLock::new();
/// `true` while a KittenTTS model is resident in memory.
static KITTEN_LOADED: AtomicBool = AtomicBool::new(false);
static ACTIVE_VOICE: OnceLock<RwLock<String>> = OnceLock::new();

fn voice_lock() -> &'static RwLock<String> {
    ACTIVE_VOICE.get_or_init(|| RwLock::new(VOICE_DEFAULT.to_string()))
}
fn get_voice() -> String {
    voice_lock().read().map(|g| g.clone()).unwrap_or_else(|_| VOICE_DEFAULT.to_string())
}
fn set_voice_inner(voice: String) {
    if let Ok(mut g) = voice_lock().write() { *g = voice; }
}

// ─── KittenTTS worker commands ────────────────────────────────────────────────

enum TtsCmd {
    Init  { cb: Box<dyn FnMut(LoadProgress) + Send + 'static>, done: oneshot::Sender<Result<(), String>> },
    Speak { text: String, voice: String, done: oneshot::Sender<()> },
    Unload { done: oneshot::Sender<()> },
}

static TTS_TX: OnceLock<std::sync::mpsc::SyncSender<TtsCmd>> = OnceLock::new();

fn get_tx() -> &'static std::sync::mpsc::SyncSender<TtsCmd> {
    TTS_TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<TtsCmd>(16);
        std::thread::Builder::new()
            .name("skill-tts".into())
            .spawn(|| kitten_worker(rx))
            .expect("failed to spawn TTS worker thread");
        tx
    })
}

// ─── espeak-ng data path (macOS) ─────────────────────────────────────────────

/// Resolve the espeak-ng data directory and register it with both TTS crates.
///
/// Resolution order: `ESPEAK_DATA_PATH` env var → `.app` bundle →
/// `ESPEAK_DATA_PATH_DEV` baked in at compile time by `build.rs`.
#[cfg(target_os = "macos")]
fn init_espeak_data_path() {
    let resolved: Option<std::path::PathBuf> = (|| {
        if let Ok(p) = std::env::var("ESPEAK_DATA_PATH") {
            let path = std::path::Path::new(&p);
            if path.is_dir() {
                eprintln!("[tts] espeak-ng data path (env): {p}");
                return Some(path.to_path_buf());
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            let bundled = exe.parent().and_then(|p| p.parent())
                .map(|p| p.join("Resources").join("espeak-ng-data"));
            if let Some(ref path) = bundled {
                if path.is_dir() {
                    eprintln!("[tts] espeak-ng data path (bundle): {}", path.display());
                    return Some(path.clone());
                }
            }
        }
        if let Some(p) = option_env!("ESPEAK_DATA_PATH_DEV") {
            let path = std::path::Path::new(p);
            if path.is_dir() {
                eprintln!("[tts] espeak-ng data path (dev static): {p}");
                return Some(path.to_path_buf());
            }
        }
        None
    })();

    if let Some(path) = resolved {
        kittentts::phonemize::set_data_path(&path);
        neutts::phonemize::set_data_path(&path);
    } else {
        eprintln!(
            "[tts] WARNING: espeak-ng data path not resolved — phonemisation will likely fail."
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn init_espeak_data_path() {}

// ─── KittenTTS worker ────────────────────────────────────────────────────────

fn kitten_worker(rx: std::sync::mpsc::Receiver<TtsCmd>) {
    init_espeak_data_path();

    let mut stream: Option<MixerDeviceSink> = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| eprintln!("[tts] warning: could not open audio: {e}")).ok();
    let mut model: Option<KittenTTS> = None;

    for cmd in rx {
        match cmd {
            TtsCmd::Init { cb, done } => {
                if KITTEN_LOADED.load(Ordering::Relaxed) {
                    done.send(Ok(())).ok();
                    continue;
                }
                match download::load_from_hub_cb(HF_REPO, cb) {
                    Ok(m) => {
                        let voices = m.available_voices.clone();
                        let _ = AVAILABLE_VOICES.set(voices.clone());
                        eprintln!(
                            "[tts] KittenTTS ready (repo={HF_REPO} voices={voices:?})"
                        );
                        model = Some(m);
                        KITTEN_LOADED.store(true, Ordering::Relaxed);
                        done.send(Ok(())).ok();
                    }
                    Err(e) => {
                        done.send(Err(format!("kittentts: model load failed: {e}"))).ok();
                    }
                }
            }

            TtsCmd::Speak { text, voice, done } => {
                // Lazy-init if unloaded.
                if model.is_none() {
                    match download::load_from_hub_cb(HF_REPO, |_| {}) {
                        Ok(m) => {
                            let _ = AVAILABLE_VOICES.set(m.available_voices.clone());
                            KITTEN_LOADED.store(true, Ordering::Relaxed);
                            model = Some(m);
                        }
                        Err(e) => {
                            eprintln!("[tts] lazy init failed: {e}");
                            done.send(()).ok();
                            continue;
                        }
                    }
                }
                if stream.is_none() {
                    stream = DeviceSinkBuilder::open_default_sink()
                        .map_err(|e| eprintln!("[tts] could not open audio: {e}")).ok();
                }
                match (&model, &stream) {
                    (Some(m), Some(s)) => {
                        if let Err(e) = kitten_speak_inner(m, s, &text, &voice) {
                            eprintln!("[tts] synthesis error: {e}");
                        }
                    }
                    (_, None) => eprintln!("[tts] speak skipped: no audio output device"),
                    _ => {}
                }
                done.send(()).ok();
            }

            TtsCmd::Unload { done } => {
                model = None;
                KITTEN_LOADED.store(false, Ordering::Relaxed);
                eprintln!("[tts] KittenTTS model unloaded");
                done.send(()).ok();
            }
        }
    }
    eprintln!("[tts] kitten worker thread exiting");
}

fn kitten_speak_inner(
    model: &KittenTTS, stream: &MixerDeviceSink, text: &str, voice: &str,
) -> Result<(), String> {
    let t0 = std::time::Instant::now();
    let mut samples = model
        .generate(text, voice, SPEED, true)
        .map_err(|e| format!("synthesis failed for {text:?}: {e}"))?;

    if samples.is_empty() {
        eprintln!("[tts] no samples for {text:?} voice={voice:?}");
        return Ok(());
    }

    tts_log(&format!(
        "synthesised {len} samples ({dur:.2} s) in {ms} ms — text={text:?} voice={voice:?}",
        len = samples.len(), dur = samples.len() as f32 / KITTEN_SAMPLE_RATE as f32,
        ms = t0.elapsed().as_millis(),
    ));

    samples.extend(std::iter::repeat_n(0.0_f32, (KITTEN_SAMPLE_RATE as f32 * TAIL_SILENCE_SECS) as usize));
    let player = Player::connect_new(stream.mixer());
    player.append(SamplesBuffer::new(
        std::num::NonZero::new(1u16).unwrap(),
        std::num::NonZero::new(KITTEN_SAMPLE_RATE).unwrap(),
        samples,
    ));
    player.sleep_until_end();
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// ── NeuTTS backend ────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

const NEUTTS_SAMPLE_RATE: u32 = neutts::codec::SAMPLE_RATE;

// ─── NeuTTS config statics ────────────────────────────────────────────────────

static NEUTTS_ENABLED:  AtomicBool = AtomicBool::new(false);
static NEUTTS_LOADING:  AtomicBool = AtomicBool::new(false);
static NEUTTS_READY:    AtomicBool = AtomicBool::new(false);

struct NeuttsRuntimeConfig {
    backbone_repo: String,
    gguf_file:     Option<String>,
    voice_preset:  String,   // "jo" | "dave" | "greta" | "juliette" | "mateo" | ""
    ref_wav_path:  String,   // used when voice_preset is empty
    ref_text:      String,
}

static NEUTTS_CFG: OnceLock<RwLock<NeuttsRuntimeConfig>> = OnceLock::new();

fn neutts_cfg_lock() -> &'static RwLock<NeuttsRuntimeConfig> {
    NEUTTS_CFG.get_or_init(|| {
        RwLock::new(NeuttsRuntimeConfig {
            backbone_repo: "neuphonic/neutts-nano-q4-gguf".into(),
            gguf_file:     None,
            voice_preset:  "jo".into(),
            ref_wav_path:  String::new(),
            ref_text:      String::new(),
        })
    })
}

/// Sync the NeuTTS config statics from `settings.json` data.
/// Call on startup and whenever the user saves NeuTTS settings.
pub fn neutts_apply_config(cfg: &crate::settings::NeuttsConfig) {
    let was_ready = NEUTTS_READY.load(Ordering::Relaxed);

    if let Ok(mut g) = neutts_cfg_lock().write() {
        g.backbone_repo = cfg.backbone_repo.clone();
        g.gguf_file     = if cfg.gguf_file.is_empty() { None } else { Some(cfg.gguf_file.clone()) };
        g.voice_preset  = cfg.voice_preset.clone();
        g.ref_wav_path  = cfg.ref_wav_path.clone();
        g.ref_text      = cfg.ref_text.clone();
    }

    NEUTTS_ENABLED.store(cfg.enabled, Ordering::Relaxed);

    if cfg.enabled && was_ready {
        NEUTTS_READY.store(false, Ordering::Relaxed);
        tts_log("NeuTTS config updated — will reinitialise on next tts_init");
    }
}

// ─── NeuTTS preset voice names (must match samples/ filenames) ───────────────

const NEUTTS_PRESET_NAMES: &[&str] = &["jo", "dave", "greta", "juliette", "mateo"];

fn is_neutts_preset(name: &str) -> bool {
    NEUTTS_PRESET_NAMES.contains(&name)
}

// ─── NeuTTS worker commands ───────────────────────────────────────────────────

enum NeuttsCmd {
    Init {
        backbone_repo: String,
        gguf_file:     Option<String>,
        voice_preset:  String,
        ref_wav_path:  String,
        ref_text:      String,
        cb:   Box<dyn FnMut(neutts::download::LoadProgress) + Send + 'static>,
        done: oneshot::Sender<Result<(), String>>,
    },
    /// `voice_override` is an optional NeuTTS preset name that, when supplied,
    /// overrides the loaded reference codes for this single utterance only.
    Speak { text: String, voice_override: Option<String>, done: oneshot::Sender<()> },
    Unload { done: oneshot::Sender<()> },
}

static NEUTTS_TX: OnceLock<std::sync::mpsc::SyncSender<NeuttsCmd>> = OnceLock::new();

fn get_neutts_tx() -> &'static std::sync::mpsc::SyncSender<NeuttsCmd> {
    NEUTTS_TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<NeuttsCmd>(16);
        std::thread::Builder::new()
            .name("skill-neutts".into())
            .spawn(|| neutts_worker(rx))
            .expect("failed to spawn NeuTTS worker thread");
        tx
    })
}

// ─── NeuTTS worker ───────────────────────────────────────────────────────────

fn neutts_worker(rx: std::sync::mpsc::Receiver<NeuttsCmd>) {
    init_espeak_data_path();

    let mut stream: Option<MixerDeviceSink> = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| eprintln!("[neutts] warning: could not open audio: {e}")).ok();

    let mut model:           Option<neutts::NeuTTS> = None;
    let mut loaded_backbone: String                  = String::new();
    let mut ref_codes:       Vec<i32>                = Vec::new();
    let mut ref_text_cached: String                  = String::new();

    for cmd in rx {
        match cmd {
            // ── Init ─────────────────────────────────────────────────────────
            NeuttsCmd::Init { backbone_repo, gguf_file, voice_preset, ref_wav_path, ref_text, mut cb, done } => {
                NEUTTS_LOADING.store(true, Ordering::Relaxed);

                // ── 1. Load backbone if needed ────────────────────────────────
                if model.is_none() || loaded_backbone != backbone_repo {
                    NEUTTS_READY.store(false, Ordering::Relaxed);
                    match neutts::download::load_from_hub_cb(&backbone_repo, gguf_file.as_deref(), |p| cb(p)) {
                        Ok(m) => {
                            eprintln!("[neutts] backbone ready (repo={backbone_repo})");
                            loaded_backbone = backbone_repo;
                            model = Some(m);
                        }
                        Err(e) => {
                            NEUTTS_LOADING.store(false, Ordering::Relaxed);
                            done.send(Err(format!("neutts: backbone load failed: {e}"))).ok();
                            continue;
                        }
                    }
                }

                // ── 2. Load reference codes ────────────────────────────────────
                //
                // Priority:
                //   a) Preset name  → load .npy from bundled samples (no encoder).
                //   b) Custom WAV   → check SHA-256 cache, then try encoder.
                //   c) Neither      → empty ref (backbone built-in voice).
                let loaded = load_ref_codes(
                    model.as_ref().unwrap(),
                    &voice_preset,
                    &ref_wav_path,
                    &ref_text,
                );
                ref_codes       = loaded.0;
                ref_text_cached = loaded.1;

                NEUTTS_READY.store(true, Ordering::Relaxed);
                NEUTTS_LOADING.store(false, Ordering::Relaxed);
                done.send(Ok(())).ok();
            }

            // ── Speak ─────────────────────────────────────────────────────────
            NeuttsCmd::Speak { text, voice_override, done } => {
                if model.is_none() {
                    match neutts::download::load_from_hub_cb("neuphonic/neutts-nano-q4-gguf", None, |_| {}) {
                        Ok(m) => {
                            loaded_backbone = "neuphonic/neutts-nano-q4-gguf".into();
                            // Apply current preset on lazy init
                            let (preset, wav, txt) = {
                                let g = neutts_cfg_lock().read().unwrap();
                                (g.voice_preset.clone(), g.ref_wav_path.clone(), g.ref_text.clone())
                            };
                            let loaded = load_ref_codes(&m, &preset, &wav, &txt);
                            ref_codes       = loaded.0;
                            ref_text_cached = loaded.1;
                            model = Some(m);
                            NEUTTS_READY.store(true, Ordering::Relaxed);
                        }
                        Err(e) => {
                            eprintln!("[neutts] lazy init failed: {e}");
                            done.send(()).ok();
                            continue;
                        }
                    }
                }
                if stream.is_none() {
                    stream = DeviceSinkBuilder::open_default_sink()
                        .map_err(|e| eprintln!("[neutts] could not open audio: {e}")).ok();
                }

                // Resolve per-utterance reference codes.
                // If a valid preset name was supplied, load it inline for this
                // utterance only — without touching the stored ref_codes state.
                let (eff_codes, eff_text): (std::borrow::Cow<Vec<i32>>, std::borrow::Cow<str>) =
                    if let Some(ref ovr) = voice_override {
                        if is_neutts_preset(ovr) {
                            let loaded = load_ref_codes(
                                model.as_ref().unwrap(), ovr, "", ""
                            );
                            tts_log(&format!("per-utterance preset override: {ovr:?}"));
                            (std::borrow::Cow::Owned(loaded.0),
                             std::borrow::Cow::Owned(loaded.1))
                        } else {
                            (std::borrow::Cow::Borrowed(&ref_codes),
                             std::borrow::Cow::Borrowed(&ref_text_cached))
                        }
                    } else {
                        (std::borrow::Cow::Borrowed(&ref_codes),
                         std::borrow::Cow::Borrowed(&ref_text_cached))
                    };

                match (&model, &stream) {
                    (Some(m), Some(s)) => {
                        if let Err(e) = neutts_speak_inner(m, s, &text, &eff_codes, &eff_text) {
                            eprintln!("[neutts] synthesis error: {e}");
                        }
                    }
                    (_, None) => eprintln!("[neutts] speak skipped: no audio output device"),
                    _ => {}
                }
                done.send(()).ok();
            }

            // ── Unload ────────────────────────────────────────────────────────
            NeuttsCmd::Unload { done } => {
                model = None;
                ref_codes.clear();
                ref_text_cached.clear();
                loaded_backbone.clear();
                NEUTTS_READY.store(false, Ordering::Relaxed);
                NEUTTS_LOADING.store(false, Ordering::Relaxed);
                eprintln!("[neutts] model unloaded");
                done.send(()).ok();
            }
        }
    }
    eprintln!("[neutts] worker thread exiting");
}

// ─── Reference code loading helper ───────────────────────────────────────────

/// Resolve reference codes from either a preset name or a custom WAV path.
/// Returns `(ref_codes, ref_text)`.  Both are empty if no voice is configured.
fn load_ref_codes(
    model:        &neutts::NeuTTS,
    voice_preset: &str,
    ref_wav_path: &str,
    ref_text:     &str,
) -> (Vec<i32>, String) {
    if !voice_preset.is_empty() {
        // ── Preset: load pre-encoded .npy directly (no encoder needed) ────────
        let npy_path = format!("{NEUTTS_SAMPLES_DIR}/{voice_preset}.npy");
        let txt_path = format!("{NEUTTS_SAMPLES_DIR}/{voice_preset}.txt");

        match model.load_ref_codes(std::path::Path::new(&npy_path)) {
            Ok(codes) => {
                let text = std::fs::read_to_string(&txt_path)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                tts_log(&format!(
                    "preset voice '{voice_preset}' loaded ({} tokens)", codes.len()
                ));
                return (codes, text);
            }
            Err(e) => {
                eprintln!(
                    "[neutts] preset voice '{voice_preset}' not found at \
                     {npy_path}: {e}"
                );
            }
        }
    } else if !ref_wav_path.is_empty() {
        // ── Custom WAV: check SHA-256 cache, then try encoder ─────────────────
        let wav_path = std::path::Path::new(ref_wav_path);
        let cache = neutts::RefCodeCache::new()
            .map_err(|e| eprintln!("[neutts] cache open failed: {e}")).ok();

        let cache_hit = cache.as_ref()
            .and_then(|c| c.try_load(wav_path).ok().flatten());

        if let Some((codes, outcome)) = cache_hit {
            tts_log(&format!("custom voice cache hit: {outcome}"));
            return (codes, ref_text.to_string());
        }

        match neutts::codec::NeuCodecEncoder::new() {
            Ok(encoder) => {
                match encoder.encode_wav(wav_path) {
                    Ok(codes) => {
                        if let Some(c) = &cache {
                            if let Ok(outcome) = c.store(wav_path, &codes) {
                                tts_log(&format!("custom voice encoded+cached: {outcome}"));
                            }
                        }
                        return (codes, ref_text.to_string());
                    }
                    Err(e) => eprintln!("[neutts] WAV encoding failed: {e}"),
                }
            }
            Err(e) => eprintln!(
                "[neutts] NeuCodecEncoder not compiled in ({e}). \
                 Run `cargo run --example download_models --manifest-path \
                 ../../../neutts-rs/Cargo.toml && cargo build` to enable \
                 custom voice cloning."
            ),
        }
    }

    // Fallback: empty ref = backbone built-in voice
    tts_log("using backbone built-in voice (no reference)");
    (Vec::new(), String::new())
}

// ─── NeuTTS synthesis + playback ─────────────────────────────────────────────

fn neutts_speak_inner(
    model: &neutts::NeuTTS, stream: &MixerDeviceSink,
    text: &str, ref_codes: &[i32], ref_text: &str,
) -> Result<(), String> {
    let t0 = std::time::Instant::now();

    let mut audio = model.infer(text, ref_codes, ref_text)
        .map_err(|e| format!("neutts synthesis failed for {text:?}: {e}"))?;

    if audio.is_empty() {
        eprintln!("[neutts] synthesis returned no samples for {text:?}");
        return Ok(());
    }

    tts_log(&format!(
        "synthesised {len} samples ({dur:.2} s) in {ms} ms — text={text:?}",
        len = audio.len(), dur = audio.len() as f32 / NEUTTS_SAMPLE_RATE as f32,
        ms = t0.elapsed().as_millis(),
    ));

    audio.extend(std::iter::repeat_n(0.0_f32, (NEUTTS_SAMPLE_RATE as f32 * TAIL_SILENCE_SECS) as usize));
    let player = Player::connect_new(stream.mixer());
    player.append(SamplesBuffer::new(
        std::num::NonZero::new(1u16).unwrap(),
        std::num::NonZero::new(NEUTTS_SAMPLE_RATE).unwrap(),
        audio,
    ));
    player.sleep_until_end();
    Ok(())
}

// ─── NeuTTS progress mapper ───────────────────────────────────────────────────

fn neutts_progress_to_event(p: neutts::download::LoadProgress) -> TtsProgressEvent {
    use neutts::download::LoadProgress as NP;
    match p {
        NP::Fetching { step, total, file, repo, size_mb } => {
            let label = match size_mb {
                Some(mb) => format!("{file} from {repo} (~{mb} MB)"),
                None     => format!("{file} from {repo}"),
            };
            TtsProgressEvent::step(step, total, label)
        }
        NP::Downloading { step, total, downloaded, total_bytes } => {
            let mb_dl  = downloaded  / 1_048_576;
            let mb_tot = total_bytes / 1_048_576;
            TtsProgressEvent::step(step, total, format!("Downloading… {mb_dl}/{mb_tot} MB"))
        }
        NP::Loading { step, total, component } => {
            TtsProgressEvent::step(step, total, component)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Tauri commands ────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

/// Pre-download and warm-up the active TTS engine, broadcasting
/// `"tts-progress"` events to every open window.
///
/// Routes to NeuTTS when [`NEUTTS_ENABLED`] is `true`, otherwise KittenTTS.
/// Safe to call multiple times — immediately emits `{ phase:"ready" }` when
/// the engine is already loaded.
#[tauri::command]
pub async fn tts_init(app: tauri::AppHandle) {
    if NEUTTS_ENABLED.load(Ordering::Relaxed) {
        if NEUTTS_READY.load(Ordering::Relaxed) {
            app.emit("tts-progress", TtsProgressEvent::ready()).ok();
            return;
        }
        if NEUTTS_LOADING.load(Ordering::Relaxed) { return; }

        let (backbone_repo, gguf_file, voice_preset, ref_wav_path, ref_text) = {
            let g = neutts_cfg_lock().read().expect("NeuTTS cfg lock poisoned");
            (g.backbone_repo.clone(), g.gguf_file.clone(),
             g.voice_preset.clone(), g.ref_wav_path.clone(), g.ref_text.clone())
        };

        let tx = get_neutts_tx();
        let (done_tx, done_rx) = oneshot::channel::<Result<(), String>>();
        let app2 = app.clone();
        let cb   = move |p: neutts::download::LoadProgress| {
            app2.emit("tts-progress", neutts_progress_to_event(p)).ok();
        };

        if tx.send(NeuttsCmd::Init {
            backbone_repo, gguf_file, voice_preset, ref_wav_path, ref_text,
            cb: Box::new(cb), done: done_tx,
        }).is_err() {
            eprintln!("[neutts] tts_init: channel send failed");
            return;
        }
        match done_rx.await {
            Ok(Ok(())) => { app.emit("tts-progress", TtsProgressEvent::ready()).ok(); }
            Ok(Err(e)) => { eprintln!("[neutts] init error: {e}"); }
            Err(_)     => { eprintln!("[neutts] tts_init: worker thread died"); }
        }
    } else {
        if KITTEN_LOADED.load(Ordering::Relaxed) {
            app.emit("tts-progress", TtsProgressEvent::ready()).ok();
            return;
        }
        let tx = get_tx();
        let (done_tx, done_rx) = oneshot::channel::<Result<(), String>>();
        let app2 = app.clone();
        let cb   = move |progress: LoadProgress| {
            let event = match progress {
                LoadProgress::Fetching { step, total, file } =>
                    TtsProgressEvent::step(step, total, file),
                LoadProgress::Loading =>
                    TtsProgressEvent::step(4, 4, "Loading ONNX session"),
            };
            app2.emit("tts-progress", event).ok();
        };
        if tx.send(TtsCmd::Init { cb: Box::new(cb), done: done_tx }).is_err() {
            eprintln!("[tts] tts_init: channel send failed");
            return;
        }
        match done_rx.await {
            Ok(Ok(())) => { app.emit("tts-progress", TtsProgressEvent::ready()).ok(); }
            Ok(Err(e)) => { eprintln!("[tts] init error: {e}"); }
            Err(_)     => { eprintln!("[tts] tts_init: worker thread died"); }
        }
    }
}

/// Drop the active engine's in-memory model to free RAM / GPU memory.
///
/// The engine goes idle; a subsequent `tts_init` or `tts_speak` will
/// re-download (from cache) and reload.
/// Unloads the active backend only; the other backend is unaffected.
#[tauri::command]
pub async fn tts_unload(app: tauri::AppHandle) {
    if NEUTTS_ENABLED.load(Ordering::Relaxed) {
        let tx = get_neutts_tx();
        let (done_tx, done_rx) = oneshot::channel::<()>();
        if tx.send(NeuttsCmd::Unload { done: done_tx }).is_ok() {
            done_rx.await.ok();
        }
    } else {
        let tx = get_tx();
        let (done_tx, done_rx) = oneshot::channel::<()>();
        if tx.send(TtsCmd::Unload { done: done_tx }).is_ok() {
            done_rx.await.ok();
        }
    }
    app.emit("tts-progress", TtsProgressEvent::unloaded()).ok();
}

/// Synthesise `text` and play it on the default audio output.
///
/// `voice` is engine-specific:
///
/// | Engine    | `voice` meaning                                                    |
/// |-----------|--------------------------------------------------------------------|
/// | KittenTTS | Voice name (e.g. `"Jasper"`).  Falls back to the active voice.    |
/// | NeuTTS    | Preset name (`"jo"`, `"dave"`, `"greta"`, `"juliette"`, `"mateo"`). Overrides the configured reference for this utterance only. Ignored if not a known preset. |
#[tauri::command]
pub async fn tts_speak(text: String, voice: Option<String>) {
    if NEUTTS_ENABLED.load(Ordering::Relaxed) {
        let tx = get_neutts_tx();
        let (done_tx, done_rx) = oneshot::channel::<()>();
        // Pass voice as a per-utterance preset override for NeuTTS.
        let voice_override = voice.filter(|v| !v.is_empty());
        if tx.send(NeuttsCmd::Speak { text, voice_override, done: done_tx }).is_err() {
            eprintln!("[neutts] tts_speak: channel send failed");
            return;
        }
        done_rx.await.ok();
    } else {
        let voice = voice.filter(|v| !v.is_empty()).unwrap_or_else(get_voice);
        let tx = get_tx();
        let (done_tx, done_rx) = oneshot::channel::<()>();
        if tx.send(TtsCmd::Speak { text, voice, done: done_tx }).is_err() {
            eprintln!("[tts] tts_speak: channel send failed");
            return;
        }
        done_rx.await.ok();
    }
}

/// Return the NeuTTS preset voice names bundled in the `neutts-rs/samples/`
/// directory.  These are always available — no model download required.
/// Each entry is a preset id suitable for passing as `voice` to `tts_speak`.
#[tauri::command]
pub fn tts_list_neutts_voices() -> Vec<serde_json::Value> {
    NEUTTS_PRESET_NAMES.iter().map(|&id| {
        let (lang, flag, gender) = match id {
            "jo"       => ("en-us", "🇺🇸", "♀"),
            "dave"     => ("en-us", "🇺🇸", "♂"),
            "greta"    => ("de",    "🇩🇪", "♀"),
            "juliette" => ("fr-fr", "🇫🇷", "♀"),
            "mateo"    => ("es",    "🇪🇸", "♂"),
            _          => ("",      "",    ""),
        };
        serde_json::json!({ "id": id, "lang": lang, "flag": flag, "gender": gender })
    }).collect()
}

/// Return the voice names bundled in the KittenTTS model.
/// Falls back to `["Jasper"]` if the model has not been loaded yet.
#[tauri::command]
pub async fn tts_list_voices() -> Vec<String> {
    if let Some(voices) = AVAILABLE_VOICES.get() {
        return voices.clone();
    }
    let voices = tokio::task::spawn_blocking(|| {
        download::list_voices_from_hub(HF_REPO).unwrap_or_else(|e| {
            eprintln!("[tts] list_voices_from_hub failed: {e}");
            vec![VOICE_DEFAULT.to_string()]
        })
    })
    .await
    .unwrap_or_else(|_| vec![VOICE_DEFAULT.to_string()]);
    let _ = AVAILABLE_VOICES.set(voices.clone());
    voices
}

/// Return the currently active KittenTTS voice name.
#[tauri::command]
pub fn tts_get_voice() -> String {
    get_voice()
}

/// Persist `voice` as the active KittenTTS voice for subsequent `tts_speak`
/// calls that do not supply an explicit voice.
#[tauri::command]
pub async fn tts_set_voice(voice: String) {
    tts_log(&format!("active KittenTTS voice → {voice:?}"));
    set_voice_inner(voice);
}
