// SPDX-License-Identifier: GPL-3.0-only
// Copyright (C) 2026 NeuroSkill.com
//! TTS — two backends behind a unified Tauri-command façade.
//!
//! ## Feature flags
//!
//! | Flag         | Crate       | Thread name    | Description                           |
//! |--------------|-------------|----------------|---------------------------------------|
//! | `tts-kitten` | `kittentts` | `skill-tts`    | ONNX, English only, ~30 MB            |
//! | `tts-neutts` | `neutts`    | `skill-neutts` | GGUF backbone, voice-cloning, multi-lang |
//!
//! Both are on by default.  Either can be disabled:
//! ```sh
//! cargo build --no-default-features --features tts-neutts
//! ```
//!
//! ## Command surface (always compiled, backend-gated internally)
//!
//! | Command                  | KittenTTS | NeuTTS |
//! |--------------------------|-----------|--------|
//! | `tts_init`               | ✓         | ✓      |
//! | `tts_speak`              | ✓         | ✓      |
//! | `tts_unload`             | ✓         | ✓      |
//! | `tts_list_voices`        | ✓         | —      |
//! | `tts_list_neutts_voices` | —         | ✓      |
//! | `tts_get_voice`          | ✓         | —      |
//! | `tts_set_voice`          | ✓         | —      |

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

use tauri::Emitter;
use tokio::sync::oneshot;

// rodio is an optional dep; only compiled when at least one TTS feature is on.
#[cfg(any(feature = "tts-kitten", feature = "tts-neutts"))]
use rodio::{buffer::SamplesBuffer, DeviceSinkBuilder, MixerDeviceSink, Player};

#[cfg(feature = "tts-kitten")]
use kittentts::{download::{self, LoadProgress}, KittenTTS};

#[cfg(feature = "tts-neutts")]
use std::sync::RwLock;

#[cfg(feature = "tts-neutts")]
use sha2::{Digest, Sha256};

#[cfg(feature = "tts-neutts")]
use hf_hub::{Cache, Repo, api::sync::ApiBuilder as HfApiBuilder};

// ─── Skill directory ──────────────────────────────────────────────────────────
//
// Set once during app startup via `init_tts_dirs`.  All TTS file I/O (model
// downloads, ref-code cache, WAV cache) is rooted here.

static SKILL_DIR: OnceLock<PathBuf> = OnceLock::new();

/// Register the skill data directory.  **Must be called before the first
/// `tts_init` / `tts_speak`.**  Calling more than once is a no-op.
pub fn init_tts_dirs(skill_dir: &std::path::Path) {
    if SKILL_DIR.get().is_some() { return; }

    // Pre-create all TTS subdirectories so worker threads never race on mkdir.
    // Downloaded model blobs (GGUF, pytorch_model.bin) stay in the standard
    // HuggingFace cache (~/.cache/huggingface/hub).  Only converted or
    // generated files live inside skill_dir.
    let dirs: &[&str] = &[
        #[cfg(feature = "tts-neutts")] "models/neutts",   // neucodec_decoder.safetensors
        #[cfg(feature = "tts-neutts")] "cache/neutts-wav",
        #[cfg(feature = "tts-neutts")] "cache/neutts-ref-codes",
    ];
    for sub in dirs {
        let _ = std::fs::create_dir_all(skill_dir.join(sub));
    }

    let _ = SKILL_DIR.set(skill_dir.to_path_buf());
    eprintln!("[tts] skill_dir = {}", skill_dir.display());
}

fn skill_dir() -> PathBuf {
    SKILL_DIR.get().cloned().unwrap_or_else(|| {
        eprintln!("[tts] WARNING: init_tts_dirs not called — falling back to '.'");
        PathBuf::from(".")
    })
}

// ─── Per-backend path helpers ─────────────────────────────────────────────────

/// Where NeuTTS downloads hub models + the converted decoder safetensors.
#[cfg(feature = "tts-neutts")]
fn neutts_model_dir() -> PathBuf { skill_dir().join("models/neutts") }

/// Where NeuTTS stores ref-code `.npy` files for custom voice cloning.
#[cfg(feature = "tts-neutts")]
fn neutts_ref_code_cache_dir() -> PathBuf { skill_dir().join("cache/neutts-ref-codes") }

/// Where NeuTTS stores generated `.wav` files for repeat-playback caching.
#[cfg(feature = "tts-neutts")]
fn neutts_wav_cache_dir() -> PathBuf { skill_dir().join("cache/neutts-wav") }

// ─── Logging ──────────────────────────────────────────────────────────────────

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

#[cfg(any(feature = "tts-kitten", feature = "tts-neutts"))]
const TAIL_SILENCE_SECS: f32 = 1.0;

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

// ─── Engine routing helper ────────────────────────────────────────────────────
//
// Returns `true` when the NeuTTS backend should handle the current request.
//
//  • Both features on → runtime flag `NEUTTS_ENABLED` decides.
//  • Only `tts-neutts` on → always NeuTTS.
//  • Only `tts-kitten` on → always KittenTTS.

#[cfg(all(feature = "tts-kitten", feature = "tts-neutts"))]
static NEUTTS_ENABLED: AtomicBool = AtomicBool::new(false);

#[inline]
fn use_neutts() -> bool {
    // Both compiled: runtime setting.
    #[cfg(all(feature = "tts-neutts", feature = "tts-kitten"))]
    { return NEUTTS_ENABLED.load(Ordering::Relaxed); }

    // Only NeuTTS compiled: always true.
    #[cfg(all(feature = "tts-neutts", not(feature = "tts-kitten")))]
    { return true; }

    // Only KittenTTS compiled (or neither): always false.
    #[allow(unreachable_code)]
    false
}

// ─── espeak-ng data path ──────────────────────────────────────────────────────

/// Resolve the espeak-ng data directory and register it with each compiled
/// TTS crate.  Resolution order: `ESPEAK_DATA_PATH` env var → `.app` bundle
/// → `ESPEAK_DATA_PATH_DEV` baked in by `build.rs`.
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
                eprintln!("[tts] espeak-ng data path (dev): {p}");
                return Some(path.to_path_buf());
            }
        }
        None
    })();

    if let Some(path) = resolved {
        #[cfg(feature = "tts-kitten")]
        kittentts::phonemize::set_data_path(&path);

        #[cfg(feature = "tts-neutts")]
        neutts::phonemize::set_data_path(&path);
    } else {
        eprintln!(
            "[tts] WARNING: espeak-ng data path not resolved — phonemisation will likely fail."
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn init_espeak_data_path() {}

// ══════════════════════════════════════════════════════════════════════════════
// ── KittenTTS backend  (feature = "tts-kitten") ───────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "tts-kitten")]
const HF_REPO:            &str = "KittenML/kitten-tts-mini-0.8";
#[cfg(feature = "tts-kitten")]
const VOICE_DEFAULT:      &str = "Jasper";
#[cfg(feature = "tts-kitten")]
const SPEED:              f32  = 1.0;
#[cfg(feature = "tts-kitten")]
const KITTEN_SAMPLE_RATE: u32  = kittentts::SAMPLE_RATE;

#[cfg(feature = "tts-kitten")]
static AVAILABLE_VOICES: OnceLock<Vec<String>> = OnceLock::new();
#[cfg(feature = "tts-kitten")]
static KITTEN_LOADED: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "tts-kitten")]
static ACTIVE_VOICE: OnceLock<std::sync::RwLock<String>> = OnceLock::new();

#[cfg(feature = "tts-kitten")]
fn voice_lock() -> &'static std::sync::RwLock<String> {
    ACTIVE_VOICE.get_or_init(|| std::sync::RwLock::new(VOICE_DEFAULT.to_string()))
}
#[cfg(feature = "tts-kitten")]
fn get_voice() -> String {
    voice_lock().read().map(|g| g.clone()).unwrap_or_else(|_| VOICE_DEFAULT.to_string())
}
#[cfg(feature = "tts-kitten")]
fn set_voice_inner(voice: String) {
    if let Ok(mut g) = voice_lock().write() { *g = voice; }
}

// ─── KittenTTS worker commands ────────────────────────────────────────────────

#[cfg(feature = "tts-kitten")]
enum TtsCmd {
    Init  { cb: Box<dyn FnMut(LoadProgress) + Send + 'static>, done: oneshot::Sender<Result<(), String>> },
    Speak { text: String, voice: String, done: oneshot::Sender<()> },
    Unload { done: oneshot::Sender<()> },
}

#[cfg(feature = "tts-kitten")]
static TTS_TX: OnceLock<std::sync::mpsc::SyncSender<TtsCmd>> = OnceLock::new();

#[cfg(feature = "tts-kitten")]
fn get_kitten_tx() -> &'static std::sync::mpsc::SyncSender<TtsCmd> {
    TTS_TX.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<TtsCmd>(16);
        std::thread::Builder::new()
            .name("skill-tts".into())
            .spawn(|| kitten_worker(rx))
            .expect("failed to spawn KittenTTS worker thread");
        tx
    })
}

// ─── KittenTTS worker ────────────────────────────────────────────────────────

#[cfg(feature = "tts-kitten")]
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
                        eprintln!("[tts] KittenTTS ready (repo={HF_REPO}, voices={voices:?})");
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
                    (_, None) => eprintln!("[tts] speak skipped: no audio device"),
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
}

#[cfg(feature = "tts-kitten")]
fn kitten_speak_inner(
    model: &KittenTTS, stream: &MixerDeviceSink, text: &str, voice: &str,
) -> Result<(), String> {
    let t0      = std::time::Instant::now();
    let samples = model
        .generate(text, voice, SPEED, true)
        .map_err(|e| format!("synthesis failed for {text:?}: {e}"))?;
    if samples.is_empty() {
        eprintln!("[tts] no samples for {text:?} voice={voice:?}");
        return Ok(());
    }
    tts_log(&format!(
        "synthesised {} samples ({:.2} s) in {} ms — text={text:?} voice={voice:?}",
        samples.len(), samples.len() as f32 / KITTEN_SAMPLE_RATE as f32,
        t0.elapsed().as_millis(),
    ));
    play_f32_audio(stream, samples, KITTEN_SAMPLE_RATE);
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════════
// ── NeuTTS backend  (feature = "tts-neutts") ─────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "tts-neutts")]
const NEUTTS_SAMPLE_RATE: u32 = neutts::codec::SAMPLE_RATE;

/// Absolute path to the bundled NeuTTS sample voices, resolved at compile time.
#[cfg(feature = "tts-neutts")]
const NEUTTS_SAMPLES_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../../neutts-rs/samples");

/// All valid preset voice ids (must match files in `neutts-rs/samples/`).
#[cfg(feature = "tts-neutts")]
const NEUTTS_PRESET_NAMES: &[&str] = &["jo", "dave", "greta", "juliette", "mateo"];

#[cfg(feature = "tts-neutts")]
fn is_neutts_preset(name: &str) -> bool { NEUTTS_PRESET_NAMES.contains(&name) }

// ─── NeuTTS config statics ────────────────────────────────────────────────────

#[cfg(feature = "tts-neutts")]
static NEUTTS_LOADING: AtomicBool = AtomicBool::new(false);
#[cfg(feature = "tts-neutts")]
static NEUTTS_READY:   AtomicBool = AtomicBool::new(false);

#[cfg(feature = "tts-neutts")]
struct NeuttsRuntimeConfig {
    backbone_repo: String,
    gguf_file:     Option<String>,
    voice_preset:  String,
    ref_wav_path:  String,
    ref_text:      String,
}

#[cfg(feature = "tts-neutts")]
static NEUTTS_CFG: OnceLock<RwLock<NeuttsRuntimeConfig>> = OnceLock::new();

#[cfg(feature = "tts-neutts")]
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

// ─── neutts_apply_config — always public, body gated ─────────────────────────

/// Sync NeuTTS config statics from `settings.json` data.
/// Call on startup and whenever the user saves NeuTTS settings.
/// No-op when the `tts-neutts` feature is disabled.
pub fn neutts_apply_config(cfg: &crate::settings::NeuttsConfig) {
    #[cfg(feature = "tts-neutts")]
    {
        let was_ready = NEUTTS_READY.load(Ordering::Relaxed);

        if let Ok(mut g) = neutts_cfg_lock().write() {
            g.backbone_repo = cfg.backbone_repo.clone();
            g.gguf_file     = if cfg.gguf_file.is_empty() { None } else { Some(cfg.gguf_file.clone()) };
            g.voice_preset  = cfg.voice_preset.clone();
            g.ref_wav_path  = cfg.ref_wav_path.clone();
            g.ref_text      = cfg.ref_text.clone();
        }

        // Only update the runtime-enable flag when KittenTTS is also compiled.
        // When only NeuTTS is compiled the engine is always active.
        #[cfg(feature = "tts-kitten")]
        NEUTTS_ENABLED.store(cfg.enabled, Ordering::Relaxed);

        if cfg.enabled && was_ready {
            NEUTTS_READY.store(false, Ordering::Relaxed);
            tts_log("NeuTTS config updated — will reinitialise on next tts_init");
        }
    }
}

// ─── NeuTTS worker commands ───────────────────────────────────────────────────

#[cfg(feature = "tts-neutts")]
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
    /// `voice_override` is a NeuTTS preset name that, when supplied, overrides
    /// the loaded reference codes for this single utterance only.
    Speak { text: String, voice_override: Option<String>, done: oneshot::Sender<()> },
    Unload { done: oneshot::Sender<()> },
}

#[cfg(feature = "tts-neutts")]
static NEUTTS_TX: OnceLock<std::sync::mpsc::SyncSender<NeuttsCmd>> = OnceLock::new();

#[cfg(feature = "tts-neutts")]
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

// ─── skill_neutts_load ───────────────────────────────────────────────────────
//
// Wraps `neutts::download::load_from_hub_cb` with one difference:
// the converted `neucodec_decoder.safetensors` is stored in
// `skill_dir/models/neutts/` instead of the relative `models/` path used
// by the neutts crate.
//
// Downloaded blobs (GGUF backbone, pytorch_model.bin) continue to use the
// standard HuggingFace cache (~/.cache/huggingface/hub) — no change there.
//
//   skill_dir/models/neutts/
//     neucodec_decoder.safetensors   ← converted once, reused forever
//
// Uses the same 3-step progress protocol as the neutts crate so existing
// progress-event mapping code is unchanged.

#[cfg(feature = "tts-neutts")]
fn skill_neutts_load<F>(
    backbone_repo: &str,
    gguf_file:     Option<&str>,
    mut on_progress: F,
) -> Result<neutts::NeuTTS, String>
where
    F: FnMut(neutts::download::LoadProgress),
{
    use neutts::download::{
        LoadProgress, CODEC_DECODER_REPO, CODEC_DECODER_FILE,
        CODEC_SOURCE_FILE, CODEC_DECODER_SIZE_MB, find_model,
        convert_neucodec_checkpoint,
    };

    // Standard HF cache for all downloads.
    let hf_cache = Cache::from_env();
    let api      = HfApiBuilder::new()
        .build()
        .map_err(|e| format!("Failed to init HF client: {e}"))?;

    // ── Step 1/3: backbone GGUF → standard HF cache ───────────────────────────
    let backbone_size_mb = find_model(backbone_repo).map(|m| m.size_mb);
    on_progress(LoadProgress::Fetching {
        step: 1, total: 3,
        file: gguf_file.unwrap_or("*.gguf").to_string(),
        repo: backbone_repo.into(),
        size_mb: backbone_size_mb,
    });

    let resolved_gguf: String = match gguf_file {
        Some(f) => f.to_string(),
        None => {
            let info = api.model(backbone_repo.to_string()).info()
                .map_err(|e| format!("Failed to fetch repo info for '{backbone_repo}': {e}"))?;
            info.siblings.into_iter()
                .map(|s| s.rfilename)
                .find(|f| f.ends_with(".gguf"))
                .ok_or_else(|| format!("No .gguf file in '{backbone_repo}'"))?
        }
    };

    let backbone_path = hf_dl_cb(
        &api, &hf_cache, backbone_repo, &resolved_gguf,
        |dl, tot| on_progress(LoadProgress::Downloading {
            step: 1, total: 3, downloaded: dl, total_bytes: tot,
        }),
    )?;

    // ── Step 2/3: NeuCodec decoder → skill_dir (converted once) ──────────────
    //
    // pytorch_model.bin is downloaded to the standard HF cache.
    // The *converted* safetensors file is written to skill_dir and reused.
    let decoder_dest = neutts_model_dir().join(CODEC_DECODER_FILE);

    let decoder_path = if decoder_dest.exists() {
        on_progress(LoadProgress::Fetching {
            step: 2, total: 3,
            file: CODEC_DECODER_FILE.into(),
            repo: "(skill_dir)".into(),
            size_mb: None,
        });
        decoder_dest
    } else {
        on_progress(LoadProgress::Fetching {
            step: 2, total: 3,
            file: CODEC_SOURCE_FILE.into(),
            repo: CODEC_DECODER_REPO.into(),
            size_mb: Some(CODEC_DECODER_SIZE_MB),
        });
        // pytorch_model.bin → standard HF cache
        let bin_path = hf_dl_cb(
            &api, &hf_cache, CODEC_DECODER_REPO, CODEC_SOURCE_FILE,
            |dl, tot| on_progress(LoadProgress::Downloading {
                step: 2, total: 3, downloaded: dl, total_bytes: tot,
            }),
        )?;
        on_progress(LoadProgress::Loading {
            step: 2, total: 3,
            component: format!("converting {CODEC_SOURCE_FILE} → {CODEC_DECODER_FILE}"),
        });
        // converted safetensors → skill_dir/models/neutts/
        convert_neucodec_checkpoint(&bin_path, &decoder_dest, 16, CODEC_DECODER_REPO)
            .map_err(|e| format!("Checkpoint conversion failed: {e}"))?;
        decoder_dest
    };

    // ── Step 3/3: load from explicit paths ────────────────────────────────────
    on_progress(LoadProgress::Loading {
        step: 3, total: 3,
        component: "backbone + NeuCodec decoder".into(),
    });
    let language = find_model(backbone_repo)
        .map(|m| m.language)
        .unwrap_or("en-us")
        .to_string();
    neutts::NeuTTS::load_with_decoder(&backbone_path, &decoder_path, &language)
        .map_err(|e| format!("Failed to load NeuTTS: {e}"))
}

/// hf-hub download with byte-level progress, checking `cache` before hitting
/// the network.  Uses whatever `Cache` the caller supplies (typically
/// `Cache::from_env()` which maps to `~/.cache/huggingface/hub`).
#[cfg(feature = "tts-neutts")]
fn hf_dl_cb<F: FnMut(u64, u64)>(
    api:      &hf_hub::api::sync::Api,
    cache:    &Cache,
    repo_id:  &str,
    filename: &str,
    mut on_bytes: F,
) -> Result<PathBuf, String> {
    use hf_hub::api::Progress;

    let cache_repo = cache.repo(Repo::model(repo_id.to_string()));
    if let Some(path) = cache_repo.get(filename) {
        on_bytes(1, 1);
        return Ok(path);
    }
    struct Prog<F: FnMut(u64, u64)> { cb: F, done: u64, total: u64 }
    impl<F: FnMut(u64, u64)> Progress for Prog<F> {
        fn init(&mut self, size: usize, _: &str) { self.total = size as u64; (self.cb)(0, self.total); }
        fn update(&mut self, n: usize) { self.done += n as u64; (self.cb)(self.done, self.total); }
        fn finish(&mut self) { (self.cb)(self.total, self.total); }
    }
    api.model(repo_id.to_string())
        .download_with_progress(filename, Prog { cb: on_bytes, done: 0, total: 0 })
        .map_err(|e| format!("Failed to download '{filename}' from '{repo_id}': {e}"))
}

// ─── NeuTTS worker ───────────────────────────────────────────────────────────

#[cfg(feature = "tts-neutts")]
fn neutts_worker(rx: std::sync::mpsc::Receiver<NeuttsCmd>) {
    init_espeak_data_path();

    let mut stream: Option<MixerDeviceSink> = DeviceSinkBuilder::open_default_sink()
        .map_err(|e| eprintln!("[neutts] warning: could not open audio: {e}")).ok();

    let mut model:            Option<neutts::NeuTTS> = None;
    let mut loaded_backbone:  String                 = String::new();
    let mut ref_codes:        Vec<i32>               = Vec::new();
    let mut ref_text_cached:  String                 = String::new();
    // Stable voice identifier used as part of the WAV cache key.
    let mut loaded_voice_key: String                 = "default".to_string();

    for cmd in rx {
        match cmd {
            NeuttsCmd::Init { backbone_repo, gguf_file, voice_preset, ref_wav_path, ref_text, mut cb, done } => {
                NEUTTS_LOADING.store(true, Ordering::Relaxed);

                if model.is_none() || loaded_backbone != backbone_repo {
                    NEUTTS_READY.store(false, Ordering::Relaxed);
                    match skill_neutts_load(
                        &backbone_repo, gguf_file.as_deref(), |p| cb(p)
                    ) {
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

                let (codes, text, vkey) = load_ref_codes(
                    model.as_ref().unwrap(),
                    &voice_preset, &ref_wav_path, &ref_text,
                );
                ref_codes        = codes;
                ref_text_cached  = text;
                loaded_voice_key = vkey;

                NEUTTS_READY.store(true, Ordering::Relaxed);
                NEUTTS_LOADING.store(false, Ordering::Relaxed);
                done.send(Ok(())).ok();
            }

            NeuttsCmd::Speak { text, voice_override, done } => {
                // Lazy init if model was never loaded or was unloaded.
                if model.is_none() {
                    let (repo, gguf, preset, wav, txt) = {
                        let g = neutts_cfg_lock().read().unwrap();
                        (g.backbone_repo.clone(), g.gguf_file.clone(),
                         g.voice_preset.clone(), g.ref_wav_path.clone(), g.ref_text.clone())
                    };
                    match skill_neutts_load(&repo, gguf.as_deref(), |_| {}) {
                        Ok(m) => {
                            loaded_backbone = repo;
                            let (codes, rtext, vkey) = load_ref_codes(&m, &preset, &wav, &txt);
                            ref_codes        = codes;
                            ref_text_cached  = rtext;
                            loaded_voice_key = vkey;
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

                // Per-utterance voice override: if the caller named a valid
                // preset, load those codes inline (without touching stored state).
                let (eff_codes, eff_text, eff_vkey): (
                    std::borrow::Cow<Vec<i32>>,
                    std::borrow::Cow<str>,
                    std::borrow::Cow<str>,
                ) = match voice_override.as_deref().filter(|v| !v.is_empty()) {
                    Some(ovr) if is_neutts_preset(ovr) => {
                        tts_log(&format!("per-utterance preset override: {ovr:?}"));
                        let (c, t, k) = load_ref_codes(
                            model.as_ref().unwrap(), ovr, "", ""
                        );
                        (std::borrow::Cow::Owned(c),
                         std::borrow::Cow::Owned(t),
                         std::borrow::Cow::Owned(k))
                    }
                    _ => (
                        std::borrow::Cow::Borrowed(&ref_codes),
                        std::borrow::Cow::Borrowed(ref_text_cached.as_str()),
                        std::borrow::Cow::Borrowed(loaded_voice_key.as_str()),
                    ),
                };

                if let (Some(m), Some(s)) = (&model, &stream) {
                    neutts_speak_cached(
                        m, s, &text,
                        &eff_codes, &eff_text,
                        &loaded_backbone, &eff_vkey,
                    );
                } else {
                    eprintln!("[neutts] speak skipped: no audio device");
                }
                done.send(()).ok();
            }

            NeuttsCmd::Unload { done } => {
                model = None;
                ref_codes.clear();
                ref_text_cached.clear();
                loaded_backbone.clear();
                loaded_voice_key = "default".to_string();
                NEUTTS_READY.store(false, Ordering::Relaxed);
                NEUTTS_LOADING.store(false, Ordering::Relaxed);
                eprintln!("[neutts] model unloaded");
                done.send(()).ok();
            }
        }
    }
}

// ─── WAV cache ────────────────────────────────────────────────────────────────
//
// Generated audio is stored as 16-bit PCM WAV files under:
//   skill_dir/cache/neutts-wav/{sha256(backbone + voice_key + text)}.wav
//
// The SHA-256 key ensures correctness: any change in model, voice, or text
// produces a different filename.  Files persist across restarts.

/// Content-addressed WAV path for one NeuTTS utterance.
/// `voice_key` is the preset name, `"custom-{hash}"`, or `"default"`.
#[cfg(feature = "tts-neutts")]
fn wav_cache_path(backbone_repo: &str, voice_key: &str, text: &str) -> PathBuf {
    let mut h = Sha256::new();
    h.update(backbone_repo.as_bytes());
    h.update(b"\0");
    h.update(voice_key.as_bytes());
    h.update(b"\0");
    h.update(text.as_bytes());
    let hex = format!("{:x}", h.finalize());
    neutts_wav_cache_dir().join(format!("{hex}.wav"))
}

// ─── Reference code loading helper ───────────────────────────────────────────
//
// Returns `(ref_codes, ref_text, voice_key)`.
// `voice_key` is a stable string identifier used as part of the WAV cache key:
//   - preset:  the preset id, e.g. `"jo"`
//   - custom:  `"custom-{sha256_of_wav_file}"`
//   - default: `"default"` (backbone built-in voice)

#[cfg(feature = "tts-neutts")]
fn load_ref_codes(
    model:        &neutts::NeuTTS,
    voice_preset: &str,
    ref_wav_path: &str,
    ref_text:     &str,
) -> (Vec<i32>, String, String) {
    if !voice_preset.is_empty() {
        let npy_path = format!("{NEUTTS_SAMPLES_DIR}/{voice_preset}.npy");
        let txt_path = format!("{NEUTTS_SAMPLES_DIR}/{voice_preset}.txt");
        match model.load_ref_codes(std::path::Path::new(&npy_path)) {
            Ok(codes) => {
                let text = std::fs::read_to_string(&txt_path)
                    .map(|s| s.trim().to_string()).unwrap_or_default();
                tts_log(&format!(
                    "preset voice '{voice_preset}' loaded ({} tokens)", codes.len()
                ));
                return (codes, text, voice_preset.to_string());
            }
            Err(e) => eprintln!(
                "[neutts] preset '{voice_preset}' not found at {npy_path}: {e}"
            ),
        }
    } else if !ref_wav_path.is_empty() {
        let wav_path = std::path::Path::new(ref_wav_path);

        // Compute a stable key for the custom WAV.
        let voice_key = neutts::cache::sha256_file(wav_path)
            .map(|h| format!("custom-{h}"))
            .unwrap_or_else(|_| format!("custom-{ref_wav_path}"));

        let ref_code_cache = neutts::RefCodeCache::with_dir(neutts_ref_code_cache_dir())
            .map_err(|e| eprintln!("[neutts] ref-code cache open failed: {e}")).ok();
        let cache_hit = ref_code_cache.as_ref()
            .and_then(|c| c.try_load(wav_path).ok().flatten());

        if let Some((codes, outcome)) = cache_hit {
            tts_log(&format!("custom voice ref-code cache hit: {outcome}"));
            return (codes, ref_text.to_string(), voice_key);
        }
        match neutts::codec::NeuCodecEncoder::new() {
            Ok(encoder) => match encoder.encode_wav(wav_path) {
                Ok(codes) => {
                    if let Some(c) = &ref_code_cache {
                        if let Ok(outcome) = c.store(wav_path, &codes) {
                            tts_log(&format!("custom voice encoded+cached: {outcome}"));
                        }
                    }
                    return (codes, ref_text.to_string(), voice_key);
                }
                Err(e) => eprintln!("[neutts] WAV encoding failed: {e}"),
            },
            Err(e) => eprintln!("[neutts] NeuCodecEncoder not available ({e})"),
        }
    }
    tts_log("using backbone built-in voice (no reference)");
    (Vec::new(), String::new(), "default".to_string())
}

// ─── NeuTTS synthesis ─────────────────────────────────────────────────────────

#[cfg(feature = "tts-neutts")]
fn neutts_synthesize(
    model: &neutts::NeuTTS, text: &str, ref_codes: &[i32], ref_text: &str,
) -> Result<Vec<f32>, String> {
    let t0    = std::time::Instant::now();
    let audio = model.infer(text, ref_codes, ref_text)
        .map_err(|e| format!("neutts synthesis failed for {text:?}: {e}"))?;
    if audio.is_empty() {
        return Err(format!("synthesis returned no samples for {text:?}"));
    }
    tts_log(&format!(
        "synthesised {} samples ({:.2} s) in {} ms — text={text:?}",
        audio.len(), audio.len() as f32 / NEUTTS_SAMPLE_RATE as f32,
        t0.elapsed().as_millis(),
    ));
    Ok(audio)
}

// ─── Playback helpers ─────────────────────────────────────────────────────────

/// Play raw f32 PCM samples (with a short tail of silence appended).
#[cfg(any(feature = "tts-kitten", feature = "tts-neutts"))]
fn play_f32_audio(stream: &MixerDeviceSink, mut samples: Vec<f32>, sample_rate: u32) {
    samples.extend(std::iter::repeat_n(
        0.0_f32, (sample_rate as f32 * TAIL_SILENCE_SECS) as usize,
    ));
    let player = Player::connect_new(stream.mixer());
    player.append(SamplesBuffer::new(
        std::num::NonZero::new(1u16).unwrap(),
        std::num::NonZero::new(sample_rate).unwrap(),
        samples,
    ));
    player.sleep_until_end();
}

/// Play a cached WAV file through the mixer.
///
/// Reads with `hound::WavReader` (16-bit PCM Int, written by
/// `NeuTTS::write_wav`) and converts i16 → f32 for `SamplesBuffer`, keeping
/// the same code path as fresh synthesis so no additional rodio/symphonia
/// format probing is needed.
#[cfg(feature = "tts-neutts")]
fn play_wav_file(stream: &MixerDeviceSink, path: &std::path::Path) {
    let reader = match hound::WavReader::open(path) {
        Ok(r)  => r,
        Err(e) => { eprintln!("[neutts] WAV cache open failed ({}): {e}", path.display()); return; }
    };
    let spec        = reader.spec();
    let sample_rate = spec.sample_rate;
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / i16::MAX as f32)
        .collect();

    if samples.is_empty() {
        eprintln!("[neutts] WAV cache empty: {}", path.display());
        return;
    }
    tts_log(&format!(
        "WAV cache playback: {} samples @ {} Hz ({})",
        samples.len(), sample_rate, path.display()
    ));
    play_f32_audio(stream, samples, sample_rate);
}

// ─── NeuTTS speak: cache check → synthesise → cache write → play ─────────────

#[cfg(feature = "tts-neutts")]
fn neutts_speak_cached(
    model:      &neutts::NeuTTS,
    stream:     &MixerDeviceSink,
    text:       &str,
    ref_codes:  &[i32],
    ref_text:   &str,
    backbone:   &str,
    voice_key:  &str,
) {
    let cache_path = wav_cache_path(backbone, voice_key, text);

    if cache_path.exists() {
        tts_log(&format!("WAV cache hit: {}", cache_path.display()));
        play_wav_file(stream, &cache_path);
        return;
    }

    match neutts_synthesize(model, text, ref_codes, ref_text) {
        Ok(audio) => {
            // Write to cache before playing so the file is complete on disk.
            if let Err(e) = model.write_wav(&audio, &cache_path) {
                eprintln!("[neutts] WAV cache write failed: {e}");
            } else {
                tts_log(&format!("WAV cached: {}", cache_path.display()));
            }
            play_f32_audio(stream, audio, NEUTTS_SAMPLE_RATE);
        }
        Err(e) => eprintln!("[neutts] synthesis error: {e}"),
    }
}

// ─── NeuTTS progress mapper ───────────────────────────────────────────────────

#[cfg(feature = "tts-neutts")]
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
            TtsProgressEvent::step(step, total,
                format!("Downloading… {}/{} MB",
                    downloaded / 1_048_576, total_bytes / 1_048_576))
        }
        NP::Loading { step, total, component } => {
            TtsProgressEvent::step(step, total, component)
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// ── Tauri commands ────────────────────────────────────────────────────────────
// ══════════════════════════════════════════════════════════════════════════════
//
// All commands are always compiled and registered.  The active backend is
// selected at runtime by `use_neutts()`.  When a feature is disabled the
// corresponding branches simply compile away.

/// Pre-download and warm-up the active TTS engine, emitting `"tts-progress"`
/// events.  Immediately emits `{ phase:"ready" }` if already loaded.
#[tauri::command]
pub async fn tts_init(app: tauri::AppHandle) {
    if use_neutts() {
        #[cfg(feature = "tts-neutts")]
        {
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
            let app2 = app.clone();
            let cb   = move |p| { app2.emit("tts-progress", neutts_progress_to_event(p)).ok(); };
            let (done_tx, done_rx) = oneshot::channel::<Result<(), String>>();
            if get_neutts_tx().send(NeuttsCmd::Init {
                backbone_repo, gguf_file, voice_preset, ref_wav_path, ref_text,
                cb: Box::new(cb), done: done_tx,
            }).is_err() { return; }
            match done_rx.await {
                Ok(Ok(())) => { app.emit("tts-progress", TtsProgressEvent::ready()).ok(); }
                Ok(Err(e)) => eprintln!("[neutts] init error: {e}"),
                Err(_)     => eprintln!("[neutts] worker thread died"),
            }
        }
    } else {
        #[cfg(feature = "tts-kitten")]
        {
            if KITTEN_LOADED.load(Ordering::Relaxed) {
                app.emit("tts-progress", TtsProgressEvent::ready()).ok();
                return;
            }
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
            let (done_tx, done_rx) = oneshot::channel::<Result<(), String>>();
            if get_kitten_tx().send(TtsCmd::Init { cb: Box::new(cb), done: done_tx }).is_err() {
                return;
            }
            match done_rx.await {
                Ok(Ok(())) => { app.emit("tts-progress", TtsProgressEvent::ready()).ok(); }
                Ok(Err(e)) => eprintln!("[tts] init error: {e}"),
                Err(_)     => eprintln!("[tts] worker thread died"),
            }
        }
    }
}

/// Drop the active engine's in-memory model to free RAM / VRAM.
/// A subsequent `tts_init` or `tts_speak` will reload from the local cache.
#[tauri::command]
pub async fn tts_unload(app: tauri::AppHandle) {
    if use_neutts() {
        #[cfg(feature = "tts-neutts")]
        {
            let (done_tx, done_rx) = oneshot::channel::<()>();
            if get_neutts_tx().send(NeuttsCmd::Unload { done: done_tx }).is_ok() {
                done_rx.await.ok();
            }
        }
    } else {
        #[cfg(feature = "tts-kitten")]
        {
            let (done_tx, done_rx) = oneshot::channel::<()>();
            if get_kitten_tx().send(TtsCmd::Unload { done: done_tx }).is_ok() {
                done_rx.await.ok();
            }
        }
    }
    app.emit("tts-progress", TtsProgressEvent::unloaded()).ok();
}

/// Synthesise `text` and play it on the default audio output.
///
/// `voice` is engine-specific:
///
/// | Engine    | `voice` meaning                                                         |
/// |-----------|-------------------------------------------------------------------------|
/// | KittenTTS | Voice name (e.g. `"Jasper"`).  Falls back to the active stored voice.  |
/// | NeuTTS    | Preset name (`"jo"`, `"dave"`, `"greta"`, `"juliette"`, `"mateo"`).    |
///             | Overrides the configured reference for this utterance only.             |
#[tauri::command]
pub async fn tts_speak(text: String, voice: Option<String>) {
    if use_neutts() {
        #[cfg(feature = "tts-neutts")]
        {
            let voice_override = voice.filter(|v| !v.is_empty());
            let (done_tx, done_rx) = oneshot::channel::<()>();
            if get_neutts_tx().send(NeuttsCmd::Speak { text, voice_override, done: done_tx }).is_ok() {
                done_rx.await.ok();
            }
        }
    } else {
        #[cfg(feature = "tts-kitten")]
        {
            let voice = voice.filter(|v| !v.is_empty()).unwrap_or_else(get_voice);
            let (done_tx, done_rx) = oneshot::channel::<()>();
            if get_kitten_tx().send(TtsCmd::Speak { text, voice, done: done_tx }).is_ok() {
                done_rx.await.ok();
            }
        }
    }
}

/// Return the voice names bundled in the KittenTTS model.
/// Returns `["Jasper"]` when `tts-kitten` is disabled.
#[tauri::command]
pub async fn tts_list_voices() -> Vec<String> {
    #[cfg(feature = "tts-kitten")]
    {
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
        return voices;
    }

    #[cfg(not(feature = "tts-kitten"))]
    vec!["Jasper".to_string()]
}

/// Return the NeuTTS preset voices bundled in `neutts-rs/samples/`.
/// Each entry is `{ id, lang, flag, gender }`.
/// Returns an empty array when `tts-neutts` is disabled.
#[tauri::command]
pub fn tts_list_neutts_voices() -> Vec<serde_json::Value> {
    #[cfg(feature = "tts-neutts")]
    {
        return NEUTTS_PRESET_NAMES.iter().map(|&id| {
            let (lang, flag, gender) = match id {
                "jo"       => ("en-us", "🇺🇸", "♀"),
                "dave"     => ("en-us", "🇺🇸", "♂"),
                "greta"    => ("de",    "🇩🇪", "♀"),
                "juliette" => ("fr-fr", "🇫🇷", "♀"),
                "mateo"    => ("es",    "🇪🇸", "♂"),
                _          => ("",      "",    ""),
            };
            serde_json::json!({ "id": id, "lang": lang, "flag": flag, "gender": gender })
        }).collect();
    }

    #[cfg(not(feature = "tts-neutts"))]
    vec![]
}

/// Return the currently active KittenTTS voice name.
/// Returns `"Jasper"` when `tts-kitten` is disabled.
#[tauri::command]
pub fn tts_get_voice() -> String {
    #[cfg(feature = "tts-kitten")]
    { return get_voice(); }

    #[allow(unreachable_code)]
    "Jasper".to_string()
}

/// Persist `voice` as the active KittenTTS voice for subsequent `tts_speak`
/// calls that do not supply an explicit voice.  No-op when `tts-kitten` is disabled.
#[tauri::command]
pub async fn tts_set_voice(voice: String) {
    #[cfg(feature = "tts-kitten")]
    {
        tts_log(&format!("active KittenTTS voice → {voice:?}"));
        set_voice_inner(voice);
    }
    #[cfg(not(feature = "tts-kitten"))]
    let _ = voice;
}
