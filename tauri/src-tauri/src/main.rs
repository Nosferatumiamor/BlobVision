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
use std::fs::File;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;

// winbase.h — process creation flag telling Windows not to allocate a
// console for this process tree even when a console-subsystem executable
// would normally get one. See spawn_python_engine()'s release-mode branch.
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

use tauri::Manager;

// Must match API_BASE in tauri/src/main.ts. Deliberately not 7860 (the
// Gradio app's default) so both can run side by side during the migration.
const API_PORT: &str = "8420";

struct PythonEngine(Mutex<Option<Child>>);

// Portable path resolution: walks up from the RUNNING EXE'S OWN location
// looking for the sibling app/python/blobvision_api.py that marks the repo
// root — mirrors how blobvision_paths.py resolves BLOBVISION_ROOT from
// __file__ instead of cwd, so this works the same way regardless of where
// the portable app folder was copied to. Replaces an earlier version that
// used CARGO_MANIFEST_DIR, a compile-time constant baked in as an absolute
// path on whichever machine built the binary — correct for `cargo tauri
// dev` on that same machine, but wrong (and silently so — the Python child
// would just fail to spawn) for a release binary run from anywhere else,
// including this same machine's own `target/release/` output copied
// elsewhere, or a different machine entirely.
//
// Deliberately does NOT also require venv/ to exist (an earlier version
// checked for venv/Scripts/python.exe too) — on a genuinely fresh install
// there IS no venv yet, that's exactly what bootstrap_venv_if_missing()
// below is for, and repo_root() has to resolve correctly *before* that can
// even run.
fn repo_root() -> PathBuf {
    let exe = std::env::current_exe().expect("current_exe() should always succeed");
    let mut dir = exe.parent().expect("exe path should have a parent directory").to_path_buf();
    loop {
        let api_script = dir.join("app").join("python").join("blobvision_api.py");
        if api_script.is_file() {
            return dir;
        }
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => panic!(
                "Could not find the BlobVision repo root (app/python/blobvision_api.py) \
                 by walking up from {}. Is this exe still inside the portable BlobVision \
                 folder?",
                exe.display()
            ),
        }
    }
}

// The venv's interpreter can live in either of two layouts depending on how
// it was created — see bootstrap_venv_if_missing()'s own doc comment for
// the full story:
//   - venv/Scripts/<name>.exe: a virtualenv- or `python -m venv`-created
//     venv (this dev machine's own).
//   - venv/<name>.exe: the flat layout bootstrap_venv.ps1 produces from
//     Python's official embeddable package, for a machine that had no
//     Python environment at all before BlobVision set one up.
fn find_python_exe(root: &Path, name: &str) -> Option<PathBuf> {
    let nested = root.join("venv").join("Scripts").join(name);
    if nested.is_file() {
        return Some(nested);
    }
    let flat = root.join("venv").join(name);
    if flat.is_file() {
        return Some(flat);
    }
    None
}

// Runs app/scripts/bootstrap_venv.ps1 when neither venv layout above is
// present — a completely fresh machine that's never run BlobVision. A
// no-op (returns immediately) once a venv exists, whichever layout it is,
// so this is safe to call unconditionally on every launch rather than
// needing its own separate "have we already done this" flag.
fn bootstrap_venv_if_missing(root: &Path) -> std::io::Result<()> {
    if find_python_exe(root, "python.exe").is_some() {
        return Ok(());
    }
    let script = root.join("app").join("scripts").join("bootstrap_venv.ps1");
    let log_dir = root.join("logs");
    let _ = fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("bootstrap.log");
    let mut cmd = Command::new("powershell");
    cmd.arg("-NoProfile")
        .arg("-ExecutionPolicy")
        .arg("Bypass")
        .arg("-File")
        .arg(&script)
        .arg("-TargetDir")
        .arg(root.join("venv"))
        .current_dir(root)
        .creation_flags(CREATE_NO_WINDOW);
    if let Ok(log_file) = File::create(&log_path) {
        if let Ok(err_file) = log_file.try_clone() {
            cmd.stdout(Stdio::from(log_file)).stderr(Stdio::from(err_file));
        }
    }
    let status = cmd.status()?;
    if !status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("bootstrap_venv.ps1 failed (see {})", log_path.display()),
        ));
    }
    Ok(())
}

fn spawn_python_engine() -> std::io::Result<Child> {
    let root = repo_root();
    bootstrap_venv_if_missing(&root)?;
    // Release builds launch pythonw.exe, not python.exe — see the
    // release-mode branch below for why (a virtualenv-created venv's own
    // python.exe is a launcher stub that relaunches the real interpreter as
    // a grandchild process, and that relaunch pops its own console window
    // regardless of this process's own stdio/creation-flag setup). Dev
    // builds keep python.exe, matching the plain console workflow `cargo
    // tauri dev` already runs in.
    let python_exe_name = if cfg!(debug_assertions) { "python.exe" } else { "pythonw.exe" };
    let python = find_python_exe(&root, python_exe_name).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{} not found under venv/ (checked both venv/Scripts/ and venv/) even after bootstrap", python_exe_name),
        )
    })?;
    let api_script = root.join("app").join("python").join("blobvision_api.py");

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
    let mut cmd = Command::new(python);
    cmd.arg(api_script)
        .arg("--port")
        .arg(API_PORT)
        .arg("--no-warmup")
        .current_dir(&root)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1");

    if cfg!(debug_assertions) {
        // Dev builds (`cargo tauri dev`) already run from a real console —
        // inherit straight into it for zero-latency live output during
        // development.
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    } else {
        // Release builds are windows_subsystem="windows" (no console of
        // their own). Two things had to line up here, found by testing the
        // real built binary end to end (a real user's report + a live
        // process-tree dump), not by reasoning about it in the abstract:
        //
        // 1. venv/Scripts/python.exe in this project is NOT a plain
        //    interpreter — it's a launcher stub (this venv was built with
        //    virtualenv, not stdlib venv; its python.exe is 2.6x the base
        //    install's own) that relaunches the REAL interpreter as a
        //    GRANDCHILD process. That relaunch is the stub's own compiled
        //    logic, entirely opaque to us — redirecting THIS process's own
        //    stdio (Stdio::inherit() has nothing valid to inherit, hence
        //    the console-subsystem child getting a brand-new console
        //    window otherwise) and even CREATE_NO_WINDOW on the stub
        //    itself changed nothing: the process-tree dump showed a
        //    conhost.exe still spawned as a SIBLING of the grandchild
        //    interpreter, both children of the stub, meaning the stub
        //    decides on its own whether its child gets a console,
        //    independent of what we do to the stub's own creation.
        //    Switching to pythonw.exe (the GUI-subsystem sibling shipped
        //    right next to python.exe in Scripts/) fixed it outright: its
        //    own relaunch targets the base install's pythonw.exe too, and
        //    a GUI-subsystem process never gets an auto-allocated console
        //    at any point in that chain — confirmed via a live window
        //    enumeration showing zero stray windows after the fix,
        //    where there was reliably one before it.
        // 2. pythonw.exe's own sys.stdout/stderr are normally None (no
        //    console to write to) — but explicit stdio redirection still
        //    works through both the stub and the venv's own relaunch:
        //    confirmed the log file below actually receives every line
        //    (SDXL placement check, warmup progress, etc.) end to end.
        //
        // Stdio::piped() was tried before Stdio::inherit() and rejected
        // for a different, still-true reason: blobvision_engine.py's
        // frequent print(..., flush=True) calls raised OSError(22) against
        // an anonymous pipe on Windows. A real log FILE avoids that too —
        // no "nobody's reading the other end" failure mode the way a pipe
        // has.
        //
        // CREATE_NO_WINDOW is kept as a harmless belt-and-suspenders for
        // the direct child even though it alone didn't fix the grandchild
        // console (see point 1) — no reason to remove a flag that's doing
        // no harm and covers this process's own console-allocation
        // decision correctly.
        cmd.creation_flags(CREATE_NO_WINDOW);
        let log_dir = root.join("logs");
        let _ = fs::create_dir_all(&log_dir);
        let log_path = log_dir.join("blobvision-api.log");
        if let Ok(log_file) = File::create(&log_path) {
            if let Ok(err_file) = log_file.try_clone() {
                cmd.stdout(Stdio::from(log_file)).stderr(Stdio::from(err_file));
            }
        }
        // If the log file couldn't be created/cloned, cmd's own default
        // (inherit) applies — same stray-console fallback as before this
        // fix, better than failing to launch at all over a logging nicety.
    }

    cmd.spawn()
}

// Mirrors blobvision_ui.py's open_gallery(): opens the output folder in
// Explorer. All families now share one flat outputs/ folder (see this
// session's outputs-restructuring work — filenames carry a V/D/S/M type
// tag instead of a subfolder), so there's no per-family subdir to resolve
// anymore.
// Lets the frontend show live progress during a first-run bootstrap
// (bootstrap_venv_if_missing, above) — the ONE stretch of a launch where
// the Python API isn't running yet at all, so there's no HTTP endpoint to
// poll for status. logs/bootstrap.log already has everything (pip's own
// output includes package sizes and download progress), so this just
// tails it: pass back whatever's new since `offset` plus the file's
// current length, so the caller can poll incrementally instead of
// re-fetching a log that can grow to several hundred KB over a full
// install. `Ok(("", 0))` (not an error) when the file doesn't exist yet —
// on an ordinary launch with no bootstrap needed, it may never appear.
#[tauri::command]
fn read_bootstrap_log(offset: u64) -> Result<(String, u64), String> {
    use std::io::{Read, Seek, SeekFrom};
    let path = repo_root().join("logs").join("bootstrap.log");
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(_) => return Ok((String::new(), 0)),
    };
    let len = file.metadata().map_err(|e| e.to_string())?.len();
    // The log can be recreated (a later relaunch re-running bootstrap)
    // between polls — an offset past the new file's end means "start over"
    // rather than seeking past EOF.
    let start = if offset > len { 0 } else { offset };
    file.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    // from_utf8_lossy rather than read_to_string: this can race the
    // writer mid-append, landing exactly inside a multi-byte UTF-8
    // sequence — a hard error there would just make the poll flap.
    Ok((String::from_utf8_lossy(&buf).into_owned(), len))
}

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
        .invoke_handler(tauri::generate_handler![open_outputs_folder, read_bootstrap_log])
        .setup(|app| {
            // Managed empty, filled in once spawn_python_engine() actually
            // finishes — see the background thread below for why this
            // can't just be a blocking call right here.
            app.manage(PythonEngine(Mutex::new(None)));
            let handle_for_thread = app.handle().clone();
            thread::spawn(move || {
                // On a machine that's never run BlobVision before,
                // spawn_python_engine() now includes bootstrap_venv_if_missing()
                // — a real, from-scratch Python environment setup (downloads
                // PyTorch and ~150 other packages, several GB, easily 10+
                // minutes). Running that inline in .setup() would block the
                // window itself from ever appearing, which would look
                // exactly like a hung/crashed app with zero feedback. This
                // thread lets the window open immediately; the frontend's
                // own "Engine unreachable, retrying..." messaging (already
                // built for the ordinary case of the API taking a moment to
                // start) covers this longer wait too, just for longer.
                match spawn_python_engine() {
                    Ok(child) => {
                        let state = handle_for_thread.state::<PythonEngine>();
                        *state.0.lock().unwrap() = Some(child);
                    }
                    Err(err) => {
                        // Best-effort diagnostic trail — there's no
                        // running API to report this over HTTP, and no
                        // console in a release build to print it to.
                        let root = repo_root();
                        let _ = fs::create_dir_all(root.join("logs"));
                        let _ = fs::write(
                            root.join("logs").join("startup-error.log"),
                            format!("Failed to start blobvision_api.py: {err}"),
                        );
                    }
                }
            });
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
