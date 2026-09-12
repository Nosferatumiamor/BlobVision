// BlobVision desktop shell (foundation).
//
// Replaces pywebview + Gradio: this Rust process spawns the existing Python
// FastAPI engine (app/python/blobvision_api.py) as a child process on
// startup, opens a native window pointed at the Vite frontend, and kills the
// Python process when the window closes. The frontend talks to the Python
// API directly over HTTP (see tauri/src/main.ts) — this shell does no
// request proxying, it only owns the child process's lifecycle.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::fs;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::Manager;

// Must match API_BASE in tauri/src/main.ts. Deliberately not 7860 (the
// Gradio app's default) so both can run side by side during the migration.
const API_PORT: &str = "8420";

struct PythonEngine(Mutex<Option<Child>>);

// Portable path resolution: walks up from the RUNNING EXE'S OWN location
// looking for the sibling venv/ + app/python/ that mark the repo root —
// mirrors how blobvision_paths.py resolves BLOBVISION_ROOT from __file__
// instead of cwd, so this works the same way regardless of where the
// portable app folder was copied to. Replaces an earlier version that used
// CARGO_MANIFEST_DIR, a compile-time constant baked in as an absolute path
// on whichever machine built the binary — correct for `cargo tauri dev` on
// that same machine, but wrong (and silently so — the Python child would
// just fail to spawn) for a release binary run from anywhere else,
// including this same machine's own `target/release/` output copied
// elsewhere, or a different machine entirely.
//
// Checking for both venv/Scripts/python.exe AND app/python/blobvision_api.py
// (not just "is there a venv/ folder") avoids a false-positive match against
// an unrelated venv/ someone happens to have sitting a few levels up.
fn repo_root() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe() should always succeed");
    let mut dir = exe.parent().expect("exe path should have a parent directory").to_path_buf();
    loop {
        let python = dir.join("venv").join("Scripts").join("python.exe");
        let api_script = dir.join("app").join("python").join("blobvision_api.py");
        if python.is_file() && api_script.is_file() {
            return dir;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => panic!(
                "Could not find the BlobVision repo root (venv/Scripts/python.exe + \
                 app/python/blobvision_api.py) by walking up from {}. Is this exe \
                 still inside the portable BlobVision folder?",
                exe.display()
            ),
        }
    }
}

fn spawn_python_engine() -> std::io::Result<Child> {
    let root = repo_root();
    let python = root.join("venv").join("Scripts").join("python.exe");
    let api_script = root.join("app").join("python").join("blobvision_api.py");

    // Stdio::inherit() rather than piped(): blobvision_engine.py prints
    // progress lines with print(..., flush=True) in a lot of places, and on
    // Windows those flush() calls can raise OSError(22) against an anonymous
    // pipe in a way they never do against a real console/inherited handle —
    // that exception was propagating out of engine.generate() and turning
    // into a 500 on /generate. Inheriting this process's own stdio sidesteps
    // the whole class of issue (and is simpler — dev mode already has a
    // console attached, so the Python output shows up right here for free).
    // Mirrors app/run.ps1's env vars for the Gradio launcher: the app expects
    // model weights to already be present locally and must not silently hit
    // the network mid-session (see CLAUDE.md).
    //
    // BLOBVISION_SKETCH_FULL_GPU is deliberately NOT set here — let
    // resolve_sketch_full_gpu() in blobvision_engine.py auto-detect from
    // actual free VRAM instead of hardcoding a mode for every machine. This
    // used to force "0" (CPU-offload) unconditionally, based on an earlier
    // finding that full-GPU mode made VQGAN's own optimization loop 5-9x
    // slower "right at the edge of OOM" on whatever card that was measured
    // on. Re-measured on a 12.9GB card (real A/B test: 4 redux generations
    // in each mode): VQGAN's steady-state it/s was statistically identical
    // between the two modes (~3.3-3.4 it/s either way), and total time per
    // generation was within ~1s of each other — full-GPU's park/unpark
    // overhead roughly cancels out CPU-offload's own per-step shuffling
    // cost. The one clear difference was SDXL's own load+warmup time: ~90s
    // full-GPU vs ~170-180s CPU-offload on the same card. The old "5-9x
    // slower" finding doesn't reproduce here — it was presumably measured
    // on a card with much less VRAM headroom, which auto-detect already
    // handles correctly (falls back to CPU-offload below its own
    // total>=11GB / free>=7GB threshold). Hardcoding one mode for every
    // machine was strictly worse than letting that threshold decide.
    let child = Command::new(python)
        .arg(api_script)
        .arg("--port")
        .arg(API_PORT)
        .arg("--no-warmup")
        .current_dir(&root)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    Ok(child)
}

// Mirrors blobvision_ui.py's open_gallery(): opens the output folder in
// Explorer. All families now share one flat outputs/ folder (see this
// session's outputs-restructuring work — filenames carry a V/D/S/M type
// tag instead of a subfolder), so there's no per-family subdir to resolve
// anymore.
#[tauri::command]
fn open_outputs_folder() -> Result<(), String> {
    let out_dir = repo_root().join("outputs");
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;
    Command::new("explorer")
        .arg(&out_dir)
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![open_outputs_folder])
        .setup(|app| {
            let child = spawn_python_engine().map_err(|err| {
                format!(
                    "Failed to start blobvision_api.py — is the venv set up? ({err})"
                )
            })?;
            app.manage(PythonEngine(Mutex::new(Some(child))));
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the BlobVision Tauri application")
        .run(|app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                let state = app_handle.state::<PythonEngine>();
                let child_opt = state.0.lock().unwrap().take();
                if let Some(mut child) = child_opt {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                // child.kill() is a hard TerminateProcess on Windows, so
                // Python never gets a chance to run its own atexit cleanup
                // (see blobvision_api.py's) — sweep the uploads scratch dir
                // here instead, best-effort. outputs/ itself and its sketch/
                // subfolder are real kept content and are never touched.
                let _ = fs::remove_dir_all(repo_root().join("outputs").join("uploads"));
            }
        });
}
