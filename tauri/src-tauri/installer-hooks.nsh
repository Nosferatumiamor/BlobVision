; Custom pages aren't available through tauri.conf.json's NSIS config in
; this Tauri version (no license/extra-page field on NsisConfig — checked
; against the actual struct, not just the docs) — a full custom `template`
; would mean forking and maintaining Tauri's entire default installer.nsi
; just for this one message. NSIS_HOOK_PREINSTALL (an officially documented
; installerHooks macro) fires right before file copying starts, which is
; the closest supported hook point to "before installing" — see
; https://v2.tauri.app/distribute/windows-installer/#installer-hooks

!macro NSIS_HOOK_PREINSTALL
  MessageBox MB_OK|MB_ICONINFORMATION "BlobVision - before you install$\n$\n\
HARDWARE: tested on an NVIDIA RTX 3060 (12 GB VRAM) with 32 GB of RAM. A GPU with at least 12 GB VRAM is recommended for smooth results across every mode (VQGAN+CLIP, DeepDream, Style Transfer). An NVIDIA GPU with up-to-date drivers is effectively required - there's no practical CPU-only path. Speed and output size will vary a lot depending on your own hardware.$\n$\n\
DISK SPACE: first launch downloads a Python environment plus the SDXL Turbo, VQGAN+CLIP and Style Transfer model weights - several GB on top of this install. Make sure the install drive has room.$\n$\n\
OUTPUTS: every generated image/video stays in the app's outputs folder until you delete it yourself - nothing is cleaned up automatically. Clear it out occasionally if you generate a lot, especially video.$\n$\n\
Click OK to continue installing."
!macroend
