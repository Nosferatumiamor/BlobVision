// VQGAN + DeepDream + Style Transfer family screens — reproduces the
// layout/behavior of the Gradio sidebars (blobvision_ui.py's
// vqgan-sidebar-panel / deepdream-sidebar-panel / style-sidebar-panel)
// against blobvision_api.py's HTTP API.
//
// Must match the port the Rust shell passes to blobvision_api.py in
// src-tauri/src/main.rs. Deliberately not 7860 (the Gradio app's default) so
// both can run side by side during the Gradio -> Tauri migration.
import { GRIMOIRE_PROMPT_CATEGORIES, GRIMOIRE_NEGATIVE_CATEGORIES, type GrimoireCategory } from "./grimoire-data";

const API_BASE = "http://127.0.0.1:8420";
const HEALTH_POLL_MS = 1000;
// Used by ensureSamEmbedded's mid-session retry budget (the API is already
// known to be up by then) — NOT by waitForEngine's initial wait, which needs
// its own much longer budget; see ENGINE_STARTUP_TIMEOUT_MS below.
const HEALTH_TIMEOUT_MS = 60000;
// A fresh install (see main.rs's bootstrap_venv_if_missing / app/scripts/
// bootstrap_venv.ps1) downloads a whole Python environment + PyTorch + ~90
// packages before the API is reachable at all — easily several minutes,
// vs. a normal launch's well-under-a-minute startup. Has to be generous
// enough that a legitimate bootstrap still quietly finishing doesn't get
// mistaken for a dead process and reported as permanently "unreachable".
const ENGINE_STARTUP_TIMEOUT_MS = 40 * 60 * 1000;
// Beyond ordinary launch time — first bootstrap is the likely explanation,
// worth saying so instead of leaving "Checking engine..." looking stuck.
const ENGINE_STARTUP_SLOW_HINT_MS = 60000;

type Family = "vqgan" | "deepdream" | "style";

// Mirrors blobvision_ui.py's MODE_BLURBS / ITER_SLIDER_MAX / DEFAULT_ITERATIONS /
// DEFAULT_DENOISE and blobvision_engine.py's ASPECT_FORMATS — kept in sync by
// hand since the frontend has no build-time link to the Python constants.
const MODE_BLURBS: Record<string, string> = {
  redux: "SDXL sketch → VQGAN blob. Fast default.",
  legacy: "VQGAN-only, 150-500 steps. Optional source image.",
};

// Longer-form family/mode explanations shown in the output canvas once a
// non-SDXL model has actually loaded (see updateOutputHint) — the sidebar
// blurbs above stay short since they're always on-screen next to the
// controls, these are the "what is this and why" version for the otherwise
// empty output area.
const LOADING_HINT_TEXT =
  "Selecting a family while SDXL loads (~5s after warmup) decides which model loads next to it. Switching families after that loads nothing by itself — hit Degenerate once in a different family to swap its model in.";
const VQGAN_LONG_BLURBS: Record<string, string> = {
  redux:
    "Redux recreates the classic VQGAN look without the painful workflow. It uses SDXL for prompt accuracy, then degrades the result with the original VQGAN+CLIP notebook.",
  legacy:
    "Legacy runs the original historical model in all its chaotic glory. Slow, unruly, and unpredictable. While it traditionally uses 500 steps, 150 is usually enough.",
};
const DEEPDREAM_LONG_BLURB =
  "DeepDream lets you generate or corrupt images in both text-to-image and img2img modes. Its hallucinations emerge in vivid rainbow colors, producing fragments of dogs, birds, snakes, and all kinds of LSD-like surreal visions.\n\nMixed6a is recommended with a high intensity setting.";
const STYLE_LONG_BLURB =
  'Style Transfer was one of the first AI techniques to go viral. It applies the style of a reference image to another image of your choice. Select both images using the fields around this window.\n\nWe\'ve also added optional SDXL style presets, inspired by modern style transfer techniques such as ChatGPT\'s popular "Ghibli style." They can be selected below the img2img section.';

const ITER_MAX: Record<string, number> = { redux: 100, legacy: 500 };
const DEFAULT_ITERATIONS: Record<string, number> = { redux: 25, legacy: 150 };
// Video always drives VQGAN's "corrupt" (img2img) transform per keyframe,
// regardless of which of the two modes above is selected — mode only
// matters for image generation now that video lives on the shared drop
// zone instead of being its own third mode (see updateVideoOnlyFieldsVisibility).
const VIDEO_DEFAULT_ITERATIONS = 15;
const DEFAULT_DENOISE = 0.5;
const ASPECT_PIXELS: Record<string, [number, number]> = {
  "1:1": [384, 384],
  "16:9": [480, 272],
  "9:16": [272, 480],
};
let lastSeed: number | null = null;
let currentFamily: Family = "vqgan";
// What's currently loaded into the shared #init-image-drop zone — null means
// empty. Drives whether the video-only extra fields (keyframe thinning,
// encode range) show, whether Generate routes to a video job, and which of
// the img/video previews inside the drop zone is visible.
let initFileKind: "image" | "video" | null = null;
// Filename returned by /video/upload — re-sent by /video/generate instead of
// re-uploading the whole clip on every generate click (e.g. after tweaking
// keyframe thinning). Cleared whenever a different file is picked/dropped.
let videoUploadedPath: string | null = null;
let videoProbeInfo: { duration: number; fps: number; width: number; height: number; long_warning: boolean; duration_label: string } | null = null;
// Backs the trim-range slider's percentage math — 0 while no video is
// loaded (the slider is greyed out via #video-only-fields in that state).
let videoDuration = 0;
let videoCodecsReady = false;
// Which output element (image vs video) should stay visible across a
// family/mode switch — set whenever a generate call actually completes, so
// switching tabs shows your last real result of the right kind instead of
// always defaulting to the image slot (see syncOutputVisibilityForMode).
let lastResultWasVideo = false;
let stylePresetsLoaded = false;
// The non-SDXL model currently resident in VRAM — at most one family's model
// is ever loaded alongside SDXL. Switching tabs never loads/unloads anything
// by itself; only the moment SDXL warmup finishes (whichever family the
// user happens to be on right then gets preloaded next — no extra fixed
// delay added on top, see startStagedWarmup) and hitting Generate in a
// family that isn't resident trigger a swap (see ensureFamilyResident).
// null until the first swap actually happens.
let residentFamily: Family | null = null;
let residentFamilySwap: Promise<void> | null = null;
let studioMode = false;
type StudioAspect = "1:1" | "16:9" | "9:16" | "custom";
const STUDIO_ASPECT_RATIOS: Record<Exclude<StudioAspect, "custom">, [number, number]> = {
  "1:1": [1, 1],
  "16:9": [16, 9],
  "9:16": [9, 16],
};
const STUDIO_EXPORT_MAX_DIM = 1280;
// Deliberately the dumbest possible sticker set — real color emoji glyphs,
// drawn straight from the system emoji font. NOT recolorable via canvas
// fillStyle (color emoji are COLR/bitmap glyphs baked into the font, unlike
// plain text) — see the Stickers section of chat for the tradeoff and the
// hand-drawn-shape alternative that WOULD support custom color + text.
const STUDIO_STICKERS = [
  "\u{1F602}", "\u{1F923}", "\u{1F4A9}", "\u{1F595}", "\u{1F480}", "\u{1F921}",
  "\u{1F975}", "\u{1F92F}", "\u{1F62D}", "\u{1F4A5}", "\u{1F525}", "\u{2B50}",
  "\u{1F47D}", "\u{1F922}", "\u{1F92E}", "\u{1F608}", "\u{1F479}", "\u{1F4AF}",
  "\u{1F346}", "\u{1F351}", "\u{1F631}", "\u{1F644}",
  "\u{1F60E}", "\u{1F4AA}", "\u{1F618}", "\u{1F60D}",
];

// A sticker is either a system emoji glyph (fixed-color, drawn via fillText)
// or a hand-drawn vector shape (fully recolorable in principle, currently
// fixed to a classic white/black comic-bubble look) drawn via canvas paths.
// Union instead of just widening the emoji string type so placeStudioSticker/
// the cursor preview can tell which drawing path to take.
type StudioSticker = { kind: "emoji"; glyph: string } | { kind: "shape"; id: string };

function studioStickerKey(s: StudioSticker): string {
  return s.kind === "emoji" ? "emoji:" + s.glyph : "shape:" + s.id;
}

// Speech-bubble body — a plain ellipse, no tail. Two variants (wide oval vs
// rounder) share this one drawer, parameterized by how tall the ellipse is
// relative to its width. Deliberately tail-less: the tail is a separate
// sticker (studioDrawBubbleTail) so it can be positioned/rotated
// independently of the bubble body, per the request.
function studioDrawBubbleBody(ctx: CanvasRenderingContext2D, size: number, heightRatio: number) {
  const rx = size * 0.46;
  const ry = rx * heightRatio;
  ctx.beginPath();
  ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.lineWidth = Math.max(2, size * 0.035);
  ctx.strokeStyle = "#000000";
  ctx.stroke();
}

// Bubble tail — an isosceles triangle, filled white, but stroked ONLY on
// its two slanted sides. The base (the wide edge, meant to be tucked under/
// overlapping a bubble body) is deliberately left unstroked so the two
// pieces can be composed to look like one continuous outline instead of
// showing a seam where they meet.
function studioDrawBubbleTail(ctx: CanvasRenderingContext2D, size: number) {
  const apexY = -size * 0.42;
  const baseY = size * 0.34;
  const halfBase = size * 0.26;
  ctx.beginPath();
  ctx.moveTo(0, apexY);
  ctx.lineTo(-halfBase, baseY);
  ctx.lineTo(halfBase, baseY);
  ctx.closePath();
  ctx.fillStyle = "#ffffff";
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(-halfBase, baseY);
  ctx.lineTo(0, apexY);
  ctx.lineTo(halfBase, baseY);
  ctx.lineWidth = Math.max(2, size * 0.035);
  ctx.strokeStyle = "#000000";
  ctx.lineJoin = "round";
  ctx.stroke();
}

// SVG mirrors of the three draw functions above, used for the grid-button
// icon and the floating cursor preview (both DOM, not canvas) — same
// proportions (viewBox -50..50, i.e. size=100) as the canvas drawers so the
// preview matches what actually gets stamped.
const STUDIO_SHAPE_STICKERS: { id: string; label: string; svg: string; draw: (ctx: CanvasRenderingContext2D, size: number) => void }[] = [
  {
    id: "bubble-oval", label: "Speech bubble (oval)",
    svg: '<svg viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><ellipse cx="0" cy="0" rx="46" ry="' + (46 * 0.56).toFixed(1) + '" fill="#fff" stroke="#000" stroke-width="6"/></svg>',
    draw: (ctx, size) => studioDrawBubbleBody(ctx, size, 0.56),
  },
  {
    id: "bubble-round", label: "Speech bubble (round)",
    svg: '<svg viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg"><ellipse cx="0" cy="0" rx="46" ry="' + (46 * 0.86).toFixed(1) + '" fill="#fff" stroke="#000" stroke-width="6"/></svg>',
    draw: (ctx, size) => studioDrawBubbleBody(ctx, size, 0.86),
  },
  {
    id: "bubble-tail", label: "Speech bubble tail",
    svg: '<svg viewBox="-50 -50 100 100" xmlns="http://www.w3.org/2000/svg">'
      + '<polygon points="0,-42 -26,34 26,34" fill="#fff"/>'
      + '<polyline points="-26,34 0,-42 26,34" fill="none" stroke="#000" stroke-width="6" stroke-linejoin="round"/></svg>',
    draw: (ctx, size) => studioDrawBubbleTail(ctx, size),
  },
];
let studioAspect: StudioAspect = "custom";
let studioZoom = 1;
// Offset, in NATURAL IMAGE PIXELS, of the point centered in the crop frame
// from the image's own center — 0,0 = centered. Deliberately free/unclamped:
// dragging or rotating past the image's own edge is allowed and just
// reveals #studio-image-backdrop (a blurred cover-fit copy) instead of
// forcing the pan to stay "safely" centered.
let studioPanX = 0;
let studioPanY = 0;
let studioImageRotate = 0; // degrees, rotates the source image within the crop frame
let studioNaturalW = 0;
let studioNaturalH = 0;
let studioPanning = false;
let studioPanStartClientX = 0;
let studioPanStartClientY = 0;
let studioPanStartPanX = 0;
let studioPanStartPanY = 0;
let studioRotating = false;
let studioRotateStartAngle = 0;
let studioRotateStartValue = 0;
let studioUpscaleApplyCount = 0;
const STUDIO_UPSCALE_APPLY_LIMIT = 2;
// Persistent "patchwork" canvas behind the active image — SNAP stamps the
// current on-screen composition onto it without touching the active image,
// which stays in front and fully manipulable. Survives switching to a
// different image (thumbnail click) — only CREATE MEME (baked in as the
// backdrop) or the X-clear button reset it.
let studioPatchworkCanvas: HTMLCanvasElement | null = null;
// Persistent canvas for placed stickers — same lifecycle/reset points as
// studioPatchworkCanvas, but rendered ON TOP of the active foreground
// (#studio-sticker-layer sits after #studio-image-bg in the DOM) instead of
// behind it, since a sticker stamped on the background would just be
// covered up by the photo at normal zoom.
let studioStickerCanvas: HTMLCanvasElement | null = null;
// Snapshots of BOTH scratch canvases taken just before each mutating op
// (SNAP, sticker placement) so Undo can restore them together — see
// pushStudioUndo()/undoStudioLastAction() further down. A `null` field means
// that canvas didn't exist yet at that point.
interface StudioUndoEntry {
  patchwork: ImageBitmap | null;
  sticker: ImageBitmap | null;
}
let studioUndoStack: StudioUndoEntry[] = [];
const STUDIO_UNDO_LIMIT = 10;
// The currently "armed" sticker — non-null while the user is in
// stamp-placement mode; a click on the viewport stamps it instead of
// starting a pan. Stays armed across multiple placements/image switches
// until explicitly toggled off, so "pourrir" a whole image with repeated
// stamps doesn't need re-picking the sticker each time.
let studioActiveSticker: StudioSticker | null = null;
let studioStickerSizePercent = 18; // % of min(canvas width, height)
let studioStickerRotateDeg = 0;
let studioStickerMirrored = false;
let studioStickerPreviewEl: HTMLDivElement | null = null;
// Fractions of the viewport's own width/height, not pixels, so a dragged
// position survives window resizes and translates cleanly onto the export
// canvas regardless of its resolution (see applyTextTransform/drawMemeText).
const studioTopTextOffset = { x: 0, y: 0 };
const studioBottomTextOffset = { x: 0, y: 0 };

interface StudioTextStyle {
  rotate: number; // degrees
  scale: number; // multiplies the auto-fit font size
  contour: number; // stroke width as a fraction of font size (0 = none)
  fillColor: string;
  strokeColor: string;
}
const studioTopStyle: StudioTextStyle = { rotate: 0, scale: 1, contour: 0.08, fillColor: "#ffffff", strokeColor: "#000000" };
const studioBottomStyle: StudioTextStyle = { rotate: 0, scale: 1, contour: 0.08, fillColor: "#ffffff", strokeColor: "#000000" };

// Assigned once during the wiring section at the bottom of this file — safe
// to reference from resetStudioFullState() since that's only ever called
// well after wiring runs (in response to user interaction).
type ScrubberController = { reset: (v: number) => void };
let studioTopRotateScrub: ScrubberController;
let studioTopScaleScrub: ScrubberController;
let studioTopContourScrub: ScrubberController;
let studioBottomRotateScrub: ScrubberController;
let studioBottomScaleScrub: ScrubberController;
let studioBottomContourScrub: ScrubberController;

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function $<T extends HTMLElement>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (!el) throw new Error("missing element: " + selector);
  return el;
}

const engineStatusEl = $<HTMLDivElement>("#engine-status");
const familyBtns = document.querySelectorAll<HTMLButtonElement>(".family-btn");
const generateBtn = $<HTMLButtonElement>("#generate-btn");
const promptFieldsEl = $<HTMLDivElement>("#prompt-fields");
const promptEl = $<HTMLTextAreaElement>("#prompt");
const negativePromptEl = $<HTMLInputElement>("#negative-prompt");
const vqganModeFieldEl = $<HTMLDivElement>("#vqgan-mode-field");
const modeRadios = document.querySelectorAll<HTMLInputElement>('input[name="mode"]');
const modeBlurbEl = $<HTMLParagraphElement>("#mode-blurb");
const ddBlurbEl = $<HTMLParagraphElement>("#dd-blurb");
const formatBtnEl = $<HTMLButtonElement>("#format-btn");
const formatBtnValueEl = $<HTMLSpanElement>("#format-btn-value");
const grimoireBtnEl = $<HTMLButtonElement>("#grimoire-btn");
const promptGrimoireBtnEl = $<HTMLButtonElement>("#prompt-grimoire-btn");
const negativeGrimoireBtnEl = $<HTMLButtonElement>("#negative-grimoire-btn");
const grimoireOverlayEl = $<HTMLDivElement>("#grimoire-overlay");
const grimoireCloseBtnEl = $<HTMLButtonElement>("#grimoire-close-btn");
const grimoireTabs = document.querySelectorAll<HTMLButtonElement>(".grimoire-tab");
const grimoireListEl = $<HTMLDivElement>("#grimoire-list");
const vqganOnlyFieldsEl = $<HTMLDivElement>("#vqgan-only-fields");
const iterationsEl = $<HTMLInputElement>("#iterations");
const iterationsValueEl = $<HTMLSpanElement>("#iterations-value");
const denoiseFieldEl = $<HTMLLabelElement>("#denoise-field");
const denoiseEl = $<HTMLInputElement>("#denoise");
const denoiseValueEl = $<HTMLSpanElement>("#denoise-value");
const upscale2xEl = $<HTMLInputElement>("#upscale-2x");
const upscale4xEl = $<HTMLInputElement>("#upscale-4x");
const upscaleRowEl = $<HTMLDivElement>("#upscale-row");
const videoOnlyFieldsEl = $<HTMLDivElement>("#video-only-fields");
const videoProbeHintEl = $<HTMLParagraphElement>("#video-probe-hint");
const videoFrameStepRadios = document.querySelectorAll<HTMLInputElement>('input[name="video-frame-step"]');
const videoFrameStepFieldEl = $<HTMLDivElement>("#video-frame-step-field");
const videoEncodeFromEl = $<HTMLInputElement>("#video-encode-from");
const videoEncodeToEl = $<HTMLInputElement>("#video-encode-to");
const videoRangeFieldEl = $<HTMLDivElement>("#video-range-field");
const videoRangeValueEl = $<HTMLSpanElement>("#video-range-value");
const videoRangeSliderEl = $<HTMLDivElement>("#video-range-slider");
const videoRangeFillEl = $<HTMLDivElement>("#video-range-fill");
const videoRangeHandleFromEl = $<HTMLButtonElement>("#video-range-handle-from");
const videoRangeHandleToEl = $<HTMLButtonElement>("#video-range-handle-to");
const videoRangeTooltipEl = $<HTMLDivElement>("#video-range-tooltip");
const videoCodecsGateEl = $<HTMLDivElement>("#video-codecs-gate");
const videoCodecsInstallBtnEl = $<HTMLButtonElement>("#video-codecs-install-btn");
const outputVideoEl = $<HTMLVideoElement>("#output-video");
const deepdreamOnlyFieldsEl = $<HTMLDivElement>("#deepdream-only-fields");
const ddLayerRadios = document.querySelectorAll<HTMLInputElement>('input[name="dd-layer"]');
const ddIntensityEl = $<HTMLInputElement>("#dd-intensity");
const ddIntensityValueEl = $<HTMLSpanElement>("#dd-intensity-value");
const styleOnlyFieldsEl = $<HTMLDivElement>("#style-only-fields");
const stylePresetFieldEl = $<HTMLLabelElement>("#style-preset-field");
const stylePresetEl = $<HTMLSelectElement>("#style-preset");
const stylePresetToggleEl = $<HTMLInputElement>("#style-preset-toggle");
const styleCustomPromptFieldEl = $<HTMLLabelElement>("#style-custom-prompt-field");
const styleCustomPromptEl = $<HTMLInputElement>("#style-custom-prompt");
const stylePresetStrengthFieldEl = $<HTMLLabelElement>("#style-preset-strength-field");
const stylePresetStrengthEl = $<HTMLInputElement>("#style-preset-strength");
const stylePresetStrengthValueEl = $<HTMLSpanElement>("#style-preset-strength-value");
const styleClassicFieldsEl = $<HTMLDivElement>("#style-classic-fields");
const styleImageDropEl = $<HTMLDivElement>("#style-image-drop");
const styleImageHintEl = $<HTMLSpanElement>("#style-image-hint");
const styleImageEl = $<HTMLInputElement>("#style-image");
const styleImagePreviewEl = $<HTMLImageElement>("#style-image-preview");
const styleImageClearBtnEl = $<HTMLButtonElement>("#style-image-clear");
const styleImageDownloadBtnEl = $<HTMLButtonElement>("#style-image-download");
const styleImageCopyBtnEl = $<HTMLButtonElement>("#style-image-copy");
const styleStrengthEl = $<HTMLInputElement>("#style-strength");
const styleStrengthValueEl = $<HTMLSpanElement>("#style-strength-value");
const contentWeightEl = $<HTMLInputElement>("#content-weight");
const contentWeightValueEl = $<HTMLSpanElement>("#content-weight-value");
const styleStepsEl = $<HTMLInputElement>("#style-steps");
const styleStepsValueEl = $<HTMLSpanElement>("#style-steps-value");
const initImageReuseEl = $<HTMLInputElement>("#init-image-reuse");
const initImageDropEl = $<HTMLDivElement>("#init-image-drop");
const initImageHintEl = $<HTMLSpanElement>("#init-image-hint");
const initImageEl = $<HTMLInputElement>("#init-image");
const initImagePreviewEl = $<HTMLImageElement>("#init-image-preview");
const initVideoPreviewEl = $<HTMLVideoElement>("#init-video-preview");
const initImageClearBtnEl = $<HTMLButtonElement>("#init-image-clear");
const initImagePasteBtnEl = $<HTMLButtonElement>("#init-image-paste");
const initImageDownloadBtnEl = $<HTMLButtonElement>("#init-image-download");
const initImageCopyBtnEl = $<HTMLButtonElement>("#init-image-copy");
const samMaskCanvasEl = $<HTMLCanvasElement>("#sam-mask-canvas");
const samToggleRowEl = $<HTMLDivElement>("#sam-toggle-row");
const samWandBtnEl = $<HTMLButtonElement>("#sam-wand-btn");
const samClearBtnEl = $<HTMLButtonElement>("#sam-clear-btn");
const samOverlayEl = $<HTMLDivElement>("#sam-overlay");
const samOverlayCloseBtnEl = $<HTMLButtonElement>("#sam-overlay-close-btn");
const samOverlayCanvasWrapEl = $<HTMLDivElement>("#sam-overlay-canvas-wrap");
const samOverlayImageEl = $<HTMLImageElement>("#sam-overlay-image");
const samOverlayMaskCanvasEl = $<HTMLCanvasElement>("#sam-overlay-mask-canvas");
const samStatusEl = $<HTMLParagraphElement>("#sam-status");
const samShowMaskRowEl = $<HTMLLabelElement>("#sam-show-mask-row");
const samShowMaskToggleEl = $<HTMLInputElement>("#sam-show-mask-toggle");
const samOverlayClearBtnEl = $<HTMLButtonElement>("#sam-overlay-clear-btn");
const samOverlayResetBtnEl = $<HTMLButtonElement>("#sam-overlay-reset-btn");
const samInvertBtnEl = $<HTMLButtonElement>("#sam-invert-btn");
const samSizeEl = $<HTMLInputElement>("#sam-size");
const samSizeValueEl = $<HTMLSpanElement>("#sam-size-value");
const samSoftenEl = $<HTMLInputElement>("#sam-soften");
const samSoftenValueEl = $<HTMLSpanElement>("#sam-soften-value");
const samExpandEl = $<HTMLInputElement>("#sam-expand");
const samExpandValueEl = $<HTMLSpanElement>("#sam-expand-value");
const samCutBtnEl = $<HTMLButtonElement>("#sam-cut-btn");
const samKeepBtnEl = $<HTMLButtonElement>("#sam-keep-btn");
const samCutColorEl = $<HTMLInputElement>("#sam-cut-color");
const samCutColorRowEl = $<HTMLDivElement>("#sam-cut-color-row");
const samColorSwatchBtns = document.querySelectorAll<HTMLButtonElement>(".sam-color-swatch");
const samCutModeRadios = document.querySelectorAll<HTMLInputElement>('input[name="sam-cut-mode"]');
const samExportCutBtnEl = $<HTMLButtonElement>("#sam-export-cut-btn");
const seedValueEl = $<HTMLInputElement>("#seed-value");
const randomSeedEl = $<HTMLInputElement>("#random-seed");
const reuseSeedEl = $<HTMLInputElement>("#reuse-seed");
const genStatusEl = $<HTMLDivElement>("#gen-status");
const genStatusFillEl = $<HTMLDivElement>("#gen-status-fill");
const genStatusTextEl = $<HTMLSpanElement>("#gen-status-text");
const outputHintEl = $<HTMLParagraphElement>("#output-hint");
const outputLoadingBannerEl = $<HTMLParagraphElement>("#output-loading-banner");
const bootstrapPanelEl = $<HTMLDivElement>("#bootstrap-panel");
const bootstrapConsoleEl = $<HTMLPreElement>("#bootstrap-console");
const modelsInstallPanelEl = $<HTMLDivElement>("#models-install-panel");
const modelsCbSdxlEl = $<HTMLInputElement>("#models-cb-sdxl");
const modelsCbVqganEl = $<HTMLInputElement>("#models-cb-vqgan");
const modelsCbStyleEl = $<HTMLInputElement>("#models-cb-style");
const modelsProgressSdxlEl = $<HTMLSpanElement>("#models-progress-sdxl");
const modelsProgressVqganEl = $<HTMLSpanElement>("#models-progress-vqgan");
const modelsProgressStyleEl = $<HTMLSpanElement>("#models-progress-style");
const modelsInstallBtnEl = $<HTMLButtonElement>("#models-install-btn");
const modelsInstallStatusEl = $<HTMLParagraphElement>("#models-install-status");
const outputImageEl = $<HTMLImageElement>("#output-image");
const outputClearBtnEl = $<HTMLButtonElement>("#output-clear");
const useAsInitBtnEl = $<HTMLButtonElement>("#use-as-init-btn");
const downloadBtnEl = $<HTMLButtonElement>("#download-btn");
const copyBtnEl = $<HTMLButtonElement>("#copy-btn");
const lightboxEl = $<HTMLDivElement>("#lightbox");
const lightboxImageEl = $<HTMLImageElement>("#lightbox-image");
const metadataOutEl = $<HTMLTextAreaElement>("#metadata-out");
const galleryBtnEl = $<HTMLButtonElement>("#gallery-btn");
const cancelBtnEl = $<HTMLButtonElement>("#cancel-btn");
const layoutEl = $<HTMLElement>("#layout");
const studioBtnEl = $<HTMLButtonElement>("#studio-btn");
const studioEl = $<HTMLElement>("#studio");
const studioCanvasEl = $<HTMLDivElement>("#studio-canvas");
const studioViewportEl = $<HTMLDivElement>("#studio-viewport");
const studioImageBackdropEl = $<HTMLDivElement>("#studio-image-backdrop");
const studioPatchworkLayerEl = $<HTMLDivElement>("#studio-patchwork-layer");
const studioStickerLayerEl = $<HTMLDivElement>("#studio-sticker-layer");
const studioImageBgEl = $<HTMLDivElement>("#studio-image-bg");
const studioImageEl = $<HTMLImageElement>("#studio-image");
const studioAspectRadios = document.querySelectorAll<HTMLInputElement>('input[name="studio-aspect"]');
const studioZoomEl = $<HTMLInputElement>("#studio-zoom");
const studioZoomValueEl = $<HTMLSpanElement>("#studio-zoom-value");
const studioImageRotateEl = $<HTMLInputElement>("#studio-rotate");
const studioImageRotateValueEl = $<HTMLSpanElement>("#studio-rotate-value");
const studioUpscale2xEl = $<HTMLInputElement>("#studio-upscale-2x");
const studioUpscale4xEl = $<HTMLInputElement>("#studio-upscale-4x");
const studioUpscaleApplyBtnEl = $<HTMLButtonElement>("#studio-upscale-apply");
const studioUndoBtnEl = $<HTMLButtonElement>("#studio-undo-btn");
const studioSnapBtnEl = $<HTMLButtonElement>("#studio-snap-btn");
const studioStickerGridEl = $<HTMLDivElement>("#studio-sticker-grid");
const studioStickerShapeGridEl = $<HTMLDivElement>("#studio-sticker-shape-grid");
const studioStickerSizeEl = $<HTMLInputElement>("#studio-sticker-size");
const studioStickerSizeValueEl = $<HTMLSpanElement>("#studio-sticker-size-value");
const studioStickerRotateEl = $<HTMLInputElement>("#studio-sticker-rotate");
const studioStickerRotateValueEl = $<HTMLSpanElement>("#studio-sticker-rotate-value");
const studioStickerMirrorBtnEl = $<HTMLButtonElement>("#studio-sticker-mirror-btn");
const studioMirrorBtnEl = $<HTMLButtonElement>("#studio-mirror-btn");
const studioTextTopEl = $<HTMLDivElement>("#studio-text-top");
const studioTextBottomEl = $<HTMLDivElement>("#studio-text-bottom");
const studioTopRotateEl = $<HTMLDivElement>("#studio-top-rotate");
const studioTopScaleEl = $<HTMLDivElement>("#studio-top-scale");
const studioTopContourEl = $<HTMLDivElement>("#studio-top-contour");
const studioTopColorSwatchEl = $<HTMLDivElement>("#studio-top-color-swatch");
const studioTopFillColorEl = $<HTMLInputElement>("#studio-top-fill-color");
const studioTopStrokeColorEl = $<HTMLInputElement>("#studio-top-stroke-color");
const studioBottomRotateEl = $<HTMLDivElement>("#studio-bottom-rotate");
const studioBottomScaleEl = $<HTMLDivElement>("#studio-bottom-scale");
const studioBottomContourEl = $<HTMLDivElement>("#studio-bottom-contour");
const studioBottomColorSwatchEl = $<HTMLDivElement>("#studio-bottom-color-swatch");
const studioBottomFillColorEl = $<HTMLInputElement>("#studio-bottom-fill-color");
const studioBottomStrokeColorEl = $<HTMLInputElement>("#studio-bottom-stroke-color");
const studioFlashEl = $<HTMLDivElement>("#studio-flash");
const studioClearBtnEl = $<HTMLButtonElement>("#studio-clear");
const studioPasteBtnEl = $<HTMLButtonElement>("#studio-paste");
const studioCopyBtnEl = $<HTMLButtonElement>("#studio-copy");
const studioDownloadBtnEl = $<HTMLButtonElement>("#studio-download");
const studioStatusEl = $<HTMLDivElement>("#studio-status");
const studioStripEl = $<HTMLDivElement>("#studio-strip");
const studioTopTextEl = $<HTMLInputElement>("#studio-top-text");
const studioBottomTextEl = $<HTMLInputElement>("#studio-bottom-text");
const studioCreateBtnEl = $<HTMLButtonElement>("#studio-create-btn");

function setEngineStatus(text: string, cls: "status-pending" | "status-ok" | "status-error") {
  engineStatusEl.textContent = text;
  engineStatusEl.className = "status " + cls;
}

// Matches a "N/M" step count anywhere in the message — the only progress
// granularity actually available today: video jobs report it via lines like
// "Video: blobify 4/12..." / "Video: RIFE pass 1/2...", each stage its own
// fraction (the bar resets rather than tracking one weighted whole-job
// percentage — simpler, and still an honest per-stage signal). Plain
// single-image generation has no polling/step reporting at all yet, so
// those messages never match and fall back to the indeterminate sweep.
const PROGRESS_STEP_RE = /(\d+)\s*\/\s*(\d+)/;

function setGenStatus(text: string, cls: "status-pending" | "status-ok" | "status-error") {
  genStatusEl.hidden = false;
  genStatusTextEl.textContent = text;
  genStatusEl.className = "status " + cls;
  if (cls === "status-ok") {
    genStatusEl.classList.remove("status-indeterminate");
    genStatusFillEl.style.width = "100%";
    return;
  }
  if (cls === "status-error") {
    genStatusEl.classList.remove("status-indeterminate");
    genStatusFillEl.style.width = "0%";
    return;
  }
  const match = text.match(PROGRESS_STEP_RE);
  const total = match ? Number(match[2]) : 0;
  if (match && total > 0) {
    genStatusEl.classList.remove("status-indeterminate");
    genStatusFillEl.style.width = Math.min(100, (Number(match[1]) / total) * 100) + "%";
  } else {
    genStatusEl.classList.add("status-indeterminate");
  }
}

function familyModeBlurb(): string {
  if (currentFamily === "vqgan") return VQGAN_LONG_BLURBS[currentMode()] ?? VQGAN_LONG_BLURBS.redux;
  if (currentFamily === "deepdream") return DEEPDREAM_LONG_BLURB;
  return STYLE_LONG_BLURB;
}

// The output canvas hint starts as the model-loading explanation, then
// switches to a family/mode description once a non-SDXL model has actually
// loaded (residentFamily set) — and stays hidden entirely whenever a real
// output image is on screen. Called after every event that could change
// which of those three states applies. The loading banner tracks the exact
// same "still on the explanation, not the description" condition — small
// top-right status badge alone wasn't loud enough for the one genuinely
// multi-minute wait in a session (real user feedback).
function updateOutputHint() {
  if (!outputImageEl.hidden || !outputVideoEl.hidden) return;
  // Both alternate panels already explain why nothing's loading — stacking
  // the loading banner/hint underneath either would just be noise. The
  // bootstrap console (no engine reachable at all yet) takes priority over
  // the models panel (engine reachable, but some weights missing) since
  // they can never both be genuinely relevant at once — the first only
  // shows pre-/health, the second only after it.
  if (!bootstrapPanelEl.hidden || !modelsInstallPanelEl.hidden) {
    outputHintEl.hidden = true;
    outputLoadingBannerEl.hidden = true;
    return;
  }
  const stillLoading = residentFamily === null;
  outputHintEl.textContent = stillLoading ? LOADING_HINT_TEXT : familyModeBlurb();
  outputHintEl.hidden = false;
  outputLoadingBannerEl.hidden = !stillLoading;
}

// A video job's result is a <video>, every other generate call's is an
// <img> — only one should ever be visible at a time, but switching families
// shouldn't wipe whichever one you're NOT looking at (so flipping back shows
// your last result of that type again, rather than an empty canvas). Driven
// by what the last completed generate() call actually produced
// (lastResultWasVideo), not by the current mode/family, since video is no
// longer its own mode. Called from onModeChange()/switchFamily().
function syncOutputVisibilityForMode() {
  if (lastResultWasVideo && outputVideoEl.src) {
    outputImageEl.hidden = true;
    outputVideoEl.hidden = false;
  } else {
    outputVideoEl.hidden = true;
    outputImageEl.hidden = !outputImageEl.src;
  }
}

function currentMode(): string {
  return Array.from(modeRadios).find((r) => r.checked)?.value ?? "redux";
}

// Was a radio-row (input[name="aspect"]); replaced by the FORMAT dropdown
// button. currentAspect()/setAspect() are the sole chokepoint every other
// call site goes through, so this is a self-contained swap — nothing
// downstream (drop-handlers auto-selecting "custom", /generate body params)
// needed to change.
let currentAspectValue = "1:1";
// Set by updateVideoOnlyFieldsVisibility(): video has no source image to
// size a custom output from, so "custom" is greyed out in the FORMAT menu
// while a video is loaded.
let formatCustomDisabled = false;

function currentAspect(): string {
  return currentAspectValue;
}

function setAspect(value: string) {
  currentAspectValue = value;
  formatBtnValueEl.textContent = value;
}

function currentDdLayer(): string {
  return Array.from(ddLayerRadios).find((r) => r.checked)?.value ?? "mixed6a";
}

function currentUpscaleScale(): number {
  if (upscale4xEl.checked) return 4;
  if (upscale2xEl.checked) return 2;
  return 0;
}

// FastAPI's own 422 validation errors shape `detail` as an array of
// {loc, msg, type} objects rather than a string (HTTPException(...) calls
// from our own endpoints always send a plain string) — stringify either
// shape into something readable instead of letting `new Error(detail)`
// coerce an array/object into "[object Object]".
function errorDetailToString(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const msgs = detail.map((item) =>
      item && typeof item === "object" && "msg" in item ? String((item as any).msg) : JSON.stringify(item),
    );
    if (msgs.length) return msgs.join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

async function fetchJson(path: string, init?: RequestInit): Promise<any> {
  const res = await fetch(API_BASE + path, init);
  if (!res.ok) {
    let detail: unknown = res.statusText;
    try {
      const errBody = await res.json();
      detail = errBody.detail ?? detail;
    } catch {
      // response wasn't JSON — keep statusText
    }
    throw new Error(errorDetailToString(detail, res.statusText));
  }
  return res.json();
}

type ModelKey = "sdxl" | "vqgan" | "style";
const MODEL_CHECKBOXES: Record<ModelKey, HTMLInputElement> = {
  sdxl: modelsCbSdxlEl,
  vqgan: modelsCbVqganEl,
  style: modelsCbStyleEl,
};
const MODEL_PROGRESS_ELS: Record<ModelKey, HTMLSpanElement> = {
  sdxl: modelsProgressSdxlEl,
  vqgan: modelsProgressVqganEl,
  style: modelsProgressStyleEl,
};

interface ModelStatus {
  ready: boolean;
  installing: boolean;
  progress: string | null;
}

// None of SDXL Turbo / VQGAN+CLIP / Style Transfer's VGG19 ship with the
// app or download automatically (main.rs forces HF_HUB_OFFLINE for normal
// operation — see blobvision_engine.enable_hub_downloads) — a fresh install
// has all three missing. Cached here so startStagedWarmup()/switchFamily()
// can skip a doomed warmup call instead of eating an avoidable error — see
// familyModelsReady(). Checked once right after the engine becomes
// reachable, and again after any install run.
let modelsStatusCache: Record<ModelKey, ModelStatus> | null = null;

// A target's "ready" flag from the backend can go true before its download
// actually finishes: e.g. SDXL Turbo's readiness check looks for
// model_index.json (diffusers' own manifest), which a snapshot_download can
// write to disk well before the multi-GB weight shards land, and
// OpenCLIP's check is "destination dir is non-empty" for the same reason.
// installing (driven by this app's own /models/install thread tracking, not
// a filesystem heuristic) doesn't have that race, so treat a target as
// genuinely done only once BOTH agree — otherwise a still-downloading SDXL
// could get struck through as done in the panel, or get warmed up against
// an incomplete set of files.
function modelDone(status: ModelStatus): boolean {
  return status.ready && !status.installing;
}

// Which gated models a family's OWN warmup needs — used to decide whether
// to even attempt it. DeepDream isn't listed: its GoogLeNet weights come
// from torchvision's own hub, unaffected by the HF_HUB_OFFLINE/
// TRANSFORMERS_OFFLINE the Rust shell forces (those only gate
// huggingface_hub-based fetches), so it always just works.
function familyModelsReady(family: Family): boolean {
  if (!modelsStatusCache) return true; // unknown — don't block on a fluke
  if (family === "vqgan") return modelDone(modelsStatusCache.vqgan);
  if (family === "style") return modelDone(modelsStatusCache.style);
  return true;
}

// Targets requested by the in-flight "Download selected" click, kept
// disabled/unchecked until every one of them is either ready or has failed
// (status[key].installing goes false without ready going true). Previously
// the button re-enabled itself on every 5s poll tick regardless of whether
// the download was still running, which invited impatient repeat clicks —
// each one spawned another background thread racing the first to write the
// same partial file, which is exactly how a slow VQGAN download could end
// up truly stuck rather than just slow (see the matching backend fix in
// blobvision_api.py's _model_installing guard).
let pendingInstallTargets: Set<ModelKey> = new Set();

// Where to grab each model by hand and where it belongs, for when every
// automated source is unreachable (proxy/firewall, all mirrors down, etc.)
// — shown alongside a failed download so the user isn't stuck with no
// path forward besides retrying the same broken sources.
const MODEL_MANUAL_FALLBACK: Record<ModelKey, string> = {
  sdxl:
    'Manual fallback: download "stabilityai/sdxl-turbo" from huggingface.co and copy its files into models/sdxl-turbo/ next to the app.',
  vqgan:
    "Manual fallback: get config.yaml + model.ckpt from huggingface.co/boris/vqgan_f16_16384 (or the heibox.uni-heidelberg.de mirror) and place them in models/vqgan/ as vqgan_imagenet_f16_16384.yaml and .ckpt. The CLIP weights come from huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K into models/open_clip/.",
  style:
    "Manual fallback: VGG19 normally downloads from download.pytorch.org via torchvision — if that's blocked, fetch vgg19-dcbb9e9d.pth from there directly and place it in models/style-transfer/vgg19_imagenet.pth.",
};

// One-line status for a single model's own row — kept short since it sits
// right next to that model's checkbox label rather than stacked with every
// other target's text in one hard-to-read block at the bottom of the panel.
function describeModelStatus(status: ModelStatus): string {
  if (modelDone(status)) return "done";
  if (status.progress && status.progress.startsWith("Failed:")) return status.progress;
  if (status.installing) return status.progress || "starting…";
  return "";
}

async function refreshModelsInstallPanel(): Promise<boolean> {
  let status: Record<ModelKey, ModelStatus>;
  try {
    status = await fetchJson("/models/status");
  } catch {
    return true; // can't tell — don't block startStagedWarmup on a fluke
  }
  modelsStatusCache = status;
  const allReady = (Object.keys(MODEL_CHECKBOXES) as ModelKey[]).every((key) => modelDone(status[key]));
  modelsInstallPanelEl.hidden = allReady;
  (Object.keys(MODEL_CHECKBOXES) as ModelKey[]).forEach((key) => {
    const cb = MODEL_CHECKBOXES[key];
    const ready = status[key].ready;
    const done = modelDone(status[key]);
    // Ticked (green once done) rather than emptied out on completion — an
    // unchecked box read as "nothing happened here" rather than "finished".
    cb.checked = done ? true : !ready;
    cb.disabled = ready || status[key].installing;
    cb.parentElement!.classList.toggle("models-install-row-done", done);
    MODEL_PROGRESS_ELS[key].textContent = describeModelStatus(status[key]);
  });

  if (pendingInstallTargets.size > 0) {
    const failedTargets = Array.from(pendingInstallTargets).filter(
      (key) => !status[key].installing && !modelDone(status[key]),
    );
    if (failedTargets.length > 0) {
      modelsInstallStatusEl.hidden = false;
      modelsInstallStatusEl.textContent = failedTargets.map((key) => MODEL_MANUAL_FALLBACK[key]).join("\n");
    } else {
      modelsInstallStatusEl.hidden = true;
    }
    const stillGoing = Array.from(pendingInstallTargets).some((key) => status[key].installing);
    if (!stillGoing) {
      modelsInstallBtnEl.disabled = false;
      const anyFailed = failedTargets.length > 0;
      pendingInstallTargets = new Set();
      if (!anyFailed) {
        // Missing models that were skipped when startStagedWarmup() first ran
        // would have left engine-status stuck on an error — now that at least
        // these are here, retry rather than making the user relaunch the app.
        startStagedWarmup();
      }
    }
  }

  updateOutputHint();
  return allReady;
}

modelsInstallBtnEl.addEventListener("click", async () => {
  const targets = (Object.keys(MODEL_CHECKBOXES) as ModelKey[]).filter(
    (key) => MODEL_CHECKBOXES[key].checked && !MODEL_CHECKBOXES[key].disabled,
  );
  if (targets.length === 0) return;
  modelsInstallBtnEl.disabled = true;
  (Object.values(MODEL_CHECKBOXES) as HTMLInputElement[]).forEach((cb) => (cb.disabled = true));
  modelsInstallStatusEl.hidden = true;
  targets.forEach((key) => (MODEL_PROGRESS_ELS[key].textContent = "starting…"));
  try {
    await fetchJson("/models/install", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets }),
    });
  } catch (err) {
    modelsInstallStatusEl.textContent = "Install failed to start: " + (err as Error).message;
    modelsInstallBtnEl.disabled = false;
    (Object.values(MODEL_CHECKBOXES) as HTMLInputElement[]).forEach((cb) => (cb.disabled = false));
    return;
  }
  pendingInstallTargets = new Set(targets);
  const poll = async () => {
    await refreshModelsInstallPanel();
    if (pendingInstallTargets.size > 0) setTimeout(poll, 2000);
  };
  setTimeout(poll, 2000);
});

// Tails logs/bootstrap.log via the Rust side (read_bootstrap_log in
// main.rs) — the one phase with no HTTP endpoint to poll, since the
// Python API doesn't exist yet at all while this runs. Silently a no-op
// (returns false, touches nothing) on an ordinary launch where the file
// never appears, so this is safe to call on every waitForEngine tick
// regardless of whether this launch is actually bootstrapping.
let bootstrapLogOffset = 0;
async function pollBootstrapLog(): Promise<boolean> {
  let text: string;
  let newOffset: number;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    [text, newOffset] = await invoke<[string, number]>("read_bootstrap_log", {
      offset: bootstrapLogOffset,
    });
  } catch {
    return false;
  }
  if (newOffset < bootstrapLogOffset) bootstrapConsoleEl.textContent = ""; // log was recreated
  bootstrapLogOffset = newOffset;
  if (!text) return !bootstrapPanelEl.hidden;
  bootstrapPanelEl.hidden = false;
  bootstrapConsoleEl.textContent += text;
  bootstrapConsoleEl.scrollTop = bootstrapConsoleEl.scrollHeight;
  updateOutputHint();
  return true;
}

async function waitForEngine() {
  const start = Date.now();
  const deadline = start + ENGINE_STARTUP_TIMEOUT_MS;
  let shownSlowHint = false;
  while (Date.now() < deadline) {
    try {
      await fetchJson("/health");
      setEngineStatus("Engine reachable — warming up...", "status-ok");
      bootstrapPanelEl.hidden = true;
      // NOT generateBtn.disabled = false here — the backend responding to
      // /health just means the process is up, not that any model has
      // actually loaded yet (that's startStagedWarmup(), which can take
      // minutes). Enabling this early let Generate be clicked mid-warmup,
      // well before there was a resident model to generate anything with.
      await refreshModelsInstallPanel();
      startStagedWarmup();
      return;
    } catch {
      // API server not up yet — ordinarily just normal spawn time (well
      // under a minute), but on a genuinely fresh machine this covers a
      // real multi-minute Python/PyTorch bootstrap running in the
      // background (bootstrap_venv_if_missing in main.rs). Only escalate
      // the message once we're past how long an ordinary launch ever
      // takes, so this doesn't cry wolf on every normal startup.
      const isBootstrapping = await pollBootstrapLog();
      if (!shownSlowHint && (isBootstrapping || Date.now() - start > ENGINE_STARTUP_SLOW_HINT_MS)) {
        shownSlowHint = true;
        setEngineStatus(
          "Still starting… first launch downloads a Python environment, speed depends on your connection and disk.",
          "status-pending",
        );
      }
    }
    await new Promise((r) => setTimeout(r, HEALTH_POLL_MS));
  }
  setEngineStatus("Engine unreachable — is blobvision_api.py running on :8420?", "status-error");
}

// Loads a single family's model weights.
//
// VQGAN specifically uses /warmup/vqgan, NOT /warmup?mode=redux: the latter
// ends by activating the VQGAN VRAM phase, which parks SDXL back to CPU —
// but redux mode's actual first generate() step is always the SDXL sketch,
// so that would silently undo the SDXL warmup. /warmup/vqgan preloads
// VQGAN's weights without touching which phase is active, leaving SDXL
// resident.
// /warmup/vqgan and /style/warmup both respond 200 with {ok:false, reason}
// when their weights aren't installed rather than throwing (a clean,
// expected outcome, not a server error) — has to be checked explicitly, or
// ensureFamilyResident()'s callers would see a "successful" resident swap
// that didn't actually load anything.
async function loadFamily(family: Family) {
  if (family === "vqgan") {
    const result = await fetchJson("/warmup/vqgan", { method: "POST" });
    if (!result.ok) throw new Error(result.reason ?? "VQGAN+CLIP weights not installed.");
  } else if (family === "deepdream") {
    await fetchJson("/deepdream/warmup", { method: "POST" });
  } else {
    const result = await fetchJson("/style/warmup", { method: "POST" });
    if (!result.ok) throw new Error(result.reason ?? "Style Transfer weights not installed.");
  }
}

async function unloadFamily(family: Family) {
  if (family === "vqgan") await fetchJson("/unload/vqgan", { method: "POST" });
  else if (family === "deepdream") await fetchJson("/deepdream/unload", { method: "POST" });
  else await fetchJson("/style/unload", { method: "POST" });
}

// Only SDXL + at most one other family's model are ever resident at once —
// never all three. Swaps happen in exactly two situations: the initial 5s
// grace window right after SDXL warmup (see startStagedWarmup), and the
// first Generate click in a family that isn't the current resident (see
// onGenerate). Switching tabs by itself never loads or unloads anything.
async function ensureFamilyResident(family: Family) {
  if (residentFamily === family) return;
  // Concurrent callers (e.g. a fast double-click on Generate) await the same
  // in-flight swap instead of racing two unload/load sequences.
  if (residentFamilySwap) {
    await residentFamilySwap;
    if (residentFamily === family) return;
  }
  const swap = (async () => {
    if (residentFamily !== null) {
      try {
        await unloadFamily(residentFamily);
      } catch (err) {
        console.error("Unload failed for " + residentFamily + ":", err);
      }
    }
    await loadFamily(family);
    residentFamily = family;
  })();
  residentFamilySwap = swap;
  try {
    await swap;
  } finally {
    if (residentFamilySwap === swap) residentFamilySwap = null;
  }
}

// Mirrors blobvision_engine.py's warmup_staged(): SDXL loads first since
// every family's sketch pass can use it (blocking — the engine status badge
// reflects this), then whichever family the user is actually looking at
// the MOMENT that finishes gets its own model loaded next. No extra fixed
// delay is added on top of that — SDXL's own load already takes anywhere
// from tens of seconds to a few minutes, which is already far more decision
// window than a bolted-on wait would add, so currentFamily is read
// immediately once /warmup/sdxl resolves. No other family is preloaded in
// the background; the next swap only happens when the user actually
// generates in a different family (see onGenerate).
//
// /warmup/sdxl and /warmup/vqgan both return as soon as the real model
// weights are loaded and placed — the CUDA-kernel-priming dummy pass each
// one also does (run_sketch_warmup / run_vqgan_warmup) now runs in a
// background thread on the backend instead of being awaited here (see
// _spawn_background_warmup in blobvision_api.py). That's why Generate can
// unlock noticeably before those passes are done: the common case (a human
// takes at least a few seconds to type a prompt) lets them finish quietly
// in the background before Generate is actually clicked. If someone clicks
// Generate before they finish, the real generate() call just queues behind
// them on the backend's engine lock — same total cost, just not always
// paid up front.
// The backend runs at a boosted OS process priority (ABOVE_NORMAL) from the
// moment it starts, specifically to compete better for CPU scheduling
// against memory-heavy background apps during this one-time loading burst
// (see _set_process_priority in blobvision_api.py). Dropping it back to
// normal is fire-and-forget — a failed request here shouldn't block the UI
// from reporting the warmup outcome, and if it fails the process just stays
// slightly more aggressive than ideal rather than breaking anything.
function dropStartupPriority() {
  fetchJson("/priority/normal", { method: "POST" }).catch((err) => {
    console.error("Failed to drop startup CPU priority:", err);
  });
}

async function startStagedWarmup() {
  // Skip the call entirely rather than let it fail — with the weights
  // missing, /warmup/sdxl returning {ok:false} promptly is strictly better
  // than eating a round-trip (or, before the backend guard, a raw
  // exception) just to learn what we already knew from /models/status. The
  // models-install-panel (shown by refreshModelsInstallPanel, called right
  // before this) is the actionable fix, so this only needs to log, not
  // surface as an "error".
  if (modelsStatusCache && !modelsStatusCache.sdxl.ready) {
    setEngineStatus("SDXL Turbo not installed — see Download models below.", "status-error");
  } else {
    try {
      await fetchJson("/warmup/sdxl", { method: "POST" });
      setEngineStatus("SDXL ready — loading " + currentFamily + "...", "status-ok");
    } catch (err) {
      setEngineStatus("SDXL warmup failed: " + (err as Error).message, "status-error");
      dropStartupPriority();
      // Enabled even on failure: a broken warmup should surface as a clear
      // "Generation failed: ..." from the actual generate call (existing
      // error handling in onGenerate), not as a silently-stuck disabled
      // button with no way to even attempt it or see why.
      generateBtn.disabled = false;
      return;
    }
  }

  const family = currentFamily;
  if (!familyModelsReady(family)) {
    // Same reasoning as the SDXL skip above — ensureFamilyResident() would
    // just fail cleanly (backend guard) or throw (network/other), neither
    // of which teaches the user anything the install panel doesn't already
    // say. Generate stays disabled: there's genuinely no model to use.
    setEngineStatus(family + " weights not installed — see Download models below.", "status-error");
    dropStartupPriority();
    updateOutputHint();
    return;
  }
  try {
    await ensureFamilyResident(family);
    setEngineStatus("Engine ready (SDXL + " + family + ")", "status-ok");
  } catch (err) {
    setEngineStatus("SDXL ready — " + family + " warmup failed: " + (err as Error).message, "status-error");
  }
  // Only now — after the full staged warmup (SDXL + whichever family the
  // user landed on) actually finished, success or not — is there a real
  // model loaded to generate with.
  generateBtn.disabled = false;
  dropStartupPriority();
  updateOutputHint();
}

// Applies the current mode's iteration range/default + denoise visibility —
// shared by onModeChange (radio change) and updateVideoOnlyFieldsVisibility
// (restoring image-mode defaults when a video is cleared).
function applyIterationDefaultsForMode() {
  const mode = currentMode();
  iterationsEl.max = String(ITER_MAX[mode]);
  iterationsEl.value = String(DEFAULT_ITERATIONS[mode]);
  iterationsValueEl.textContent = iterationsEl.value;
  // Deslop (denoise_fidelity) matters for redux — only legacy (pure
  // text2img) has no use for it.
  denoiseFieldEl.hidden = mode === "legacy";
}

// Mode changes what iteration range/default and denoise visibility make
// sense — mirrors blobvision_ui.py's on_mode_change. Video no longer has its
// own mode (see updateVideoOnlyFieldsVisibility): redux/legacy now only
// affect image generation, though the mode radio itself is greyed out
// while a video is loaded since video always uses the "corrupt" transform
// regardless of which one is selected.
function onModeChange() {
  const mode = currentMode();
  modeBlurbEl.textContent = MODE_BLURBS[mode];
  applyIterationDefaultsForMode();
  syncOutputVisibilityForMode();
  updateOutputHint();
}

// Shows/hides the keyframe-thinning + encode-range fields beneath the
// shared drop zone, greys out the VQGAN mode radio (video always runs the
// "corrupt" transform regardless of redux/legacy), hides the upscale row
// (not wired for the video pipeline), disables the "custom" aspect preset
// (video has no source image to size a custom output from), and forces a
// faster iteration default suited to per-keyframe processing — all driven
// by initFileKind rather than a mode, since video is just "whatever file
// type is currently loaded" now. Called whenever the loaded file changes
// (image/video/cleared) or the family switches.
function updateVideoOnlyFieldsVisibility() {
  const isVideo = initFileKind === "video";
  videoOnlyFieldsEl.hidden = !isVideo;
  upscaleRowEl.hidden = isVideo;
  formatCustomDisabled = isVideo;
  if (isVideo && currentAspect() === "custom") setAspect("1:1");
  setFieldsInactive($<HTMLDivElement>("#mode-radio"), isVideo && currentFamily === "vqgan");
  if (currentFamily === "vqgan") {
    if (isVideo) {
      iterationsEl.max = "100";
      iterationsEl.value = String(VIDEO_DEFAULT_ITERATIONS);
      iterationsValueEl.textContent = iterationsEl.value;
      denoiseFieldEl.hidden = false;
    } else {
      applyIterationDefaultsForMode();
    }
  }
  if (isVideo) refreshVideoCodecsGate();
  updatePromptUsability();
}

const PROMPT_PLACEHOLDER_DEFAULT = "Describe SLOP";
const NEGATIVE_PROMPT_PLACEHOLDER_DEFAULT = "Separate slop with commas";
const PROMPT_PLACEHOLDER_USELESS_WITH_VIDEO = "Useless with video";

// DeepDream's video pipeline runs plain img2img per keyframe (no CLIP/text
// guidance at all — see DeepDreamEngine.generate_video()), unlike VQGAN's
// video mode where the prompt DOES steer the per-frame "corrupt" pass. Grey
// out + relabel the prompt fields so that distinction is visible instead of
// silently ignored. Scoped to DeepDream only; VQGAN's prompt still matters
// in video, and Style Transfer hides the prompt fields entirely already.
function updatePromptUsability() {
  const uselessForVideo = currentFamily === "deepdream" && initFileKind === "video";
  setFieldsInactive(promptFieldsEl, uselessForVideo);
  promptEl.placeholder = uselessForVideo ? PROMPT_PLACEHOLDER_USELESS_WITH_VIDEO : PROMPT_PLACEHOLDER_DEFAULT;
  negativePromptEl.placeholder = uselessForVideo
    ? PROMPT_PLACEHOLDER_USELESS_WITH_VIDEO
    : NEGATIVE_PROMPT_PLACEHOLDER_DEFAULT;
}

// Presets fetched once and cached — mirrors blobvision_engine.py's
// STYLE_PRESETS dict, exposed via GET /style/presets.
async function loadStylePresets() {
  if (stylePresetsLoaded) return;
  stylePresetsLoaded = true;
  try {
    const presets: { key: string; label: string; strength: number }[] = await fetchJson("/style/presets");
    for (const p of presets) {
      const opt = document.createElement("option");
      opt.value = p.key;
      opt.textContent = p.label;
      opt.dataset.strength = String(p.strength);
      stylePresetEl.appendChild(opt);
    }
  } catch (err) {
    console.error("Failed to load style presets:", err);
  }
}

// Greys out (rather than hides) a field block: keeps the layout stable and
// makes clear the fields are reachable again by flipping the preset toggle,
// instead of them vanishing outright.
function setFieldsInactive(container: HTMLElement, inactive: boolean) {
  container.classList.toggle("fields-inactive", inactive);
  container
    .querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("input, select, textarea")
    .forEach((el) => {
      el.disabled = inactive;
    });
}

function stylePresetActive(): boolean {
  return stylePresetToggleEl.checked && !!stylePresetEl.value;
}

// Mirrors blobvision_ui.py's on_st_preset_change: an active SDXL preset is
// the SDXL Turbo restyle path (needs a prompt only if "custom"); otherwise
// it's classic VGG19 style transfer (needs a style reference image). The
// preset-toggle checkbox lets a preset be flipped on/off without losing the
// dropdown selection, so classic fields are greyed out rather than hidden.
function onStylePresetChange() {
  const isActive = stylePresetActive();
  setFieldsInactive(styleClassicFieldsEl, isActive);
  setFieldsInactive(stylePresetStrengthFieldEl, !isActive);
  styleCustomPromptFieldEl.hidden = !isActive || stylePresetEl.value !== "custom";
  styleImageHintEl.textContent = isActive ? "Turn SDXL preset off to restore" : "Drop pics here";
  if (isActive) {
    const opt = stylePresetEl.selectedOptions[0];
    const strength = Number(opt?.dataset.strength ?? 0.5);
    stylePresetStrengthEl.value = String(strength);
    stylePresetStrengthValueEl.textContent = strength.toFixed(2);
  }
}

function switchFamily(family: Family) {
  exitStudioMode();
  currentFamily = family;
  initImageEl.accept = "image/*,video/*";
  familyBtns.forEach((btn) => btn.classList.toggle("active", btn.dataset.family === family));
  vqganModeFieldEl.hidden = family !== "vqgan";
  ddBlurbEl.hidden = family !== "deepdream";
  promptFieldsEl.hidden = family === "style";
  vqganOnlyFieldsEl.hidden = family !== "vqgan";
  deepdreamOnlyFieldsEl.hidden = family !== "deepdream";
  styleOnlyFieldsEl.hidden = family !== "style";
  stylePresetFieldEl.hidden = family !== "style";
  stylePresetStrengthFieldEl.hidden = family !== "style";
  updateVideoOnlyFieldsVisibility();
  if (family === "vqgan") onModeChange(); // also calls syncOutputVisibilityForMode()
  else syncOutputVisibilityForMode();
  if (family === "style") {
    loadStylePresets();
    onStylePresetChange();
  }
  // No model load/unload here — switching tabs is a pure UI action. A swap
  // only happens via the initial warmup grace window or an actual Generate
  // click (see ensureFamilyResident).
  updateOutputHint();
}

function resolveSeed(): number {
  if (randomSeedEl.checked) {
    return Math.floor(Math.random() * 2 ** 31);
  }
  if (reuseSeedEl.checked && lastSeed !== null) {
    return lastSeed;
  }
  return Number(seedValueEl.value) || 0;
}

function hasOutputImage(): boolean {
  return !outputImageEl.hidden && !!outputImageEl.src;
}

function hasOutputVideo(): boolean {
  return !outputVideoEl.hidden && !!outputVideoEl.src;
}

function hasInitImage(): boolean {
  return initFileKind === "image" && !initImagePreviewEl.hidden && !!initImagePreviewEl.src;
}

function hasInitVideo(): boolean {
  return initFileKind === "video";
}

function hasInitFile(): boolean {
  return hasInitImage() || hasInitVideo();
}

function showOutput(outputUrl: string) {
  lastResultWasVideo = false;
  outputHintEl.hidden = true;
  outputLoadingBannerEl.hidden = true;
  outputVideoEl.hidden = true;
  outputVideoEl.src = "";
  outputImageEl.src = API_BASE + outputUrl + "?t=" + Date.now();
  outputImageEl.hidden = false;
}

function showOutputVideo(outputUrl: string) {
  lastResultWasVideo = true;
  outputHintEl.hidden = true;
  outputLoadingBannerEl.hidden = true;
  outputImageEl.hidden = true;
  outputImageEl.src = "";
  outputVideoEl.src = API_BASE + outputUrl + "?t=" + Date.now();
  outputVideoEl.hidden = false;
}

function clearOutput() {
  outputImageEl.hidden = true;
  outputImageEl.src = "";
  outputVideoEl.hidden = true;
  outputVideoEl.src = "";
  updateOutputHint();
}

// sketch_url is the SDXL sketch generated from the prompt — only present
// when the user didn't supply their own img2img source (an uploaded image
// skips the sketch step entirely). Shown in the same square slot as a
// manually dropped image. By default it's an informational preview only —
// it doesn't touch initImageEl.files, so the next generate still requests a
// fresh sketch — unless "reuse img" is checked, in which case it's fetched
// and locked into initImageEl.files just like a real drop, so the next
// generate reuses this exact sketch instead of making a new one. Not
// forcing aspect to custom here (unlike showInitImagePreview): a sketch is
// already rendered at the currently selected aspect's exact dimensions.
async function showSketchPreview(sketchUrl: string) {
  const src = API_BASE + sketchUrl + "?t=" + Date.now();
  initImagePreviewEl.src = src;
  initImagePreviewEl.hidden = false;
  initImageHintEl.hidden = true;
  initImageDropEl.classList.add("has-image");
  if (initImageReuseEl.checked) {
    try {
      const blob = await fetchAsBlob(src);
      const file = new File([blob], "sketch.png", { type: blob.type || "image/png" });
      const transfer = new DataTransfer();
      transfer.items.add(file);
      initImageEl.files = transfer.files;
    } catch (err) {
      console.error("Failed to lock in sketch for reuse:", err);
    }
  }
}

async function generateVqgan() {
  const prompt = promptEl.value.trim();
  const mode = currentMode();
  const aspect = currentAspect();
  const body = new FormData();
  body.set("mode", mode);
  body.set("prompt", prompt);
  if (negativePromptEl.value.trim()) body.set("negative_prompt", negativePromptEl.value.trim());
  body.set("iterations", iterationsEl.value);
  if (mode === "redux") body.set("denoise_fidelity", denoiseEl.value);
  body.set("seed", String(resolveSeed()));
  if (aspect !== "custom") {
    const [w, h] = ASPECT_PIXELS[aspect];
    body.set("width", String(w));
    body.set("height", String(h));
  } else {
    body.set("aspect", "custom");
  }
  const upscaleScale = currentUpscaleScale();
  if (upscaleScale) body.set("upscale", String(upscaleScale));
  if (initFileKind === "image" && initImageEl.files?.[0]) body.set("init_image", initImageEl.files[0]);
  if (initFileKind === "image" && samPoints.length > 0) {
    const maskBlob = await bakeSamMaskForGenerate();
    if (maskBlob) {
      body.set("mask", maskBlob, "mask.png");
      body.set("invert", String(samInvertOn));
    }
  }
  if (samCutAccumCanvas) {
    // Keep's inversion is already baked into the accumulator per-commit
    // (see commitCutMask), so its own pixels always directly mean "cut
    // here" — no separate invert flag needed at generate time.
    const cutBlob = await new Promise<Blob | null>((resolve) => samCutAccumCanvas!.toBlob((b) => resolve(b), "image/png"));
    if (cutBlob) {
      body.set("cut_mask", cutBlob, "cut_mask.png");
      const cutMode = document.querySelector<HTMLInputElement>('input[name="sam-cut-mode"]:checked')?.value || "alpha";
      body.set("cut_mode", cutMode);
      if (cutMode === "color") body.set("cut_color", samCutColorEl.value);
    }
  }

  const result = await fetchJson("/generate", { method: "POST", body });
  showOutput(result.output_url);
  if (result.sketch_url && !initImageEl.files?.[0]) await showSketchPreview(result.sketch_url);
  return result;
}

async function generateDeepdream() {
  const prompt = promptEl.value.trim();
  const aspect = currentAspect();
  const body = new FormData();
  body.set("prompt", prompt);
  if (negativePromptEl.value.trim()) body.set("negative_prompt", negativePromptEl.value.trim());
  body.set("intensity", ddIntensityEl.value);
  body.set("layer", currentDdLayer());
  body.set("seed", String(resolveSeed()));
  if (aspect !== "custom") {
    const [w, h] = ASPECT_PIXELS[aspect];
    body.set("width", String(w));
    body.set("height", String(h));
  } else {
    body.set("aspect", "custom");
  }
  if (initFileKind === "image" && initImageEl.files?.[0]) body.set("sketch", initImageEl.files[0]);
  const upscaleScale = currentUpscaleScale();
  if (upscaleScale) body.set("upscale", String(upscaleScale));

  const result = await fetchJson("/deepdream/generate", { method: "POST", body });
  showOutput(result.output_url);
  if (result.sketch_url && !initImageEl.files?.[0]) await showSketchPreview(result.sketch_url);
  return result;
}

async function generateStyle() {
  const preset = stylePresetActive() ? stylePresetEl.value : "";
  const aspect = currentAspect();
  if (initFileKind !== "image" || !initImageEl.files?.[0]) throw new Error("Upload a content image (img2img) first.");

  const body = new FormData();
  body.set("content_image", initImageEl.files[0]);
  if (preset) {
    body.set("preset", preset);
    if (preset === "custom") body.set("custom_prompt", styleCustomPromptEl.value.trim());
    body.set("strength", stylePresetStrengthEl.value);
  } else {
    if (!styleImageEl.files?.[0]) throw new Error("Upload a style reference image.");
    body.set("style_image", styleImageEl.files[0]);
    body.set("style_strength", styleStrengthEl.value);
    body.set("content_weight", contentWeightEl.value);
    body.set("steps", styleStepsEl.value);
  }
  body.set("seed", String(resolveSeed()));
  if (aspect !== "custom") {
    const [w, h] = ASPECT_PIXELS[aspect];
    body.set("width", String(w));
    body.set("height", String(h));
  } else {
    body.set("aspect", "custom");
  }
  const upscaleScale = currentUpscaleScale();
  if (upscaleScale) body.set("upscale", String(upscaleScale));

  const result = await fetchJson("/style/generate", { method: "POST", body });
  showOutput(result.output_url);
  return result;
}

// Same preset-vs-classic branching as generateStyle, extended to video: an
// active SDXL preset bypasses the classic Gatys engine entirely (fast,
// same per-keyframe shape as VQGAN/DeepDream's own video modes); no preset
// runs the classic optimizer per keyframe at its own reduced step count
// (blobvision_style.py's VIDEO_STEPS — not the Steps slider, which only
// applies to still images) and needs a style reference image, same as the
// still-image classic path does.
async function generateStyleVideo() {
  if (!videoUploadedPath) throw new Error("Upload a source video first (or wait for the upload to finish).");

  if (videoProbeInfo?.long_warning) {
    const ok = window.confirm(
      "This clip is " + videoProbeInfo.duration_label + " long — processing may take a while. Continue?",
    );
    if (!ok) throw new Error("Cancelled.");
  }

  await refreshVideoCodecsGate();
  if (!videoCodecsReady) throw new Error("Video codecs not installed — click Install codecs first.");

  const preset = stylePresetActive() ? stylePresetEl.value : "";
  const aspect = currentAspect();
  const seed = resolveSeed();
  const body = new FormData();
  body.set("path", videoUploadedPath);
  body.set("aspect", aspect === "custom" ? "1:1" : aspect);
  if (preset) {
    body.set("preset", preset);
    if (preset === "custom") body.set("custom_prompt", styleCustomPromptEl.value.trim());
    body.set("strength", stylePresetStrengthEl.value);
  } else {
    if (!styleImageEl.files?.[0]) throw new Error("Upload a style reference image.");
    body.set("style_image", styleImageEl.files[0]);
    body.set("style_strength", styleStrengthEl.value);
    body.set("content_weight", contentWeightEl.value);
  }
  body.set("seed", String(seed));
  body.set("frame_step", String(currentVideoFrameStep()));
  if (videoRangeIsLimited()) {
    body.set("use_encode_range", "true");
    body.set("encode_from", videoEncodeFromEl.value);
    body.set("encode_to", videoEncodeToEl.value);
  }

  const startResult = await fetchJson("/style/video/generate", { method: "POST", body });
  const jobId: string = startResult.job_id;
  const done = await pollVideoJob(jobId);
  showOutputVideo(done.output_url);
  return { seed, metadata: done.metadata };
}

// All three families now support video. Style Transfer's classic Gatys
// loop is far too slow per-frame at its normal 300-step default, so its
// video path either runs at a much-reduced step count (see VIDEO_STEPS in
// blobvision_style.py) when no SDXL preset is active, or bypasses Gatys
// entirely and uses the SDXL preset per keyframe (fast, same as VQGAN/
// DeepDream's own video modes) when one is active — see generateStyleVideo.
function isVideoFamily(): boolean {
  return currentFamily === "vqgan" || currentFamily === "deepdream" || currentFamily === "style";
}

function isVideoGenerate(): boolean {
  return hasInitVideo() && isVideoFamily();
}

async function onGenerate() {
  const prompt = promptEl.value.trim();
  if (isVideoGenerate()) {
    // VQGAN's "corrupt" transform is CLIP-guided by the prompt; DeepDream's
    // img2img pass isn't prompt-driven at all, so only VQGAN requires one.
    if (currentFamily === "vqgan" && !prompt) {
      setGenStatus("Enter a prompt.", "status-error");
      return;
    }
    // Style Transfer video without a preset runs the classic engine, which
    // needs a style reference image same as the still-image path does; with
    // a preset active it needs nothing extra (SDXL bypasses the reference
    // image entirely).
    if (currentFamily === "style" && !stylePresetActive() && !styleImageEl.files?.[0]) {
      setGenStatus("Upload a style reference image.", "status-error");
      return;
    }
    if (!videoUploadedPath) {
      setGenStatus("Still uploading/reading the video — wait a moment and try again.", "status-error");
      return;
    }
  } else if (currentFamily === "style") {
    if (!hasInitImage()) {
      setGenStatus("Upload a content image (img2img) first.", "status-error");
      return;
    }
  } else if (!prompt && !hasInitImage()) {
    setGenStatus("Enter a prompt (or upload an img2img source).", "status-error");
    return;
  }

  generateBtn.disabled = true;
  outputImageEl.hidden = true;
  outputVideoEl.hidden = true;
  // The "loads models, can take a minute or two" warning is only true when
  // this family isn't the currently-resident one — that's the only case
  // where ensureFamilyResident below actually unloads/loads anything;
  // otherwise it's a no-op and generation starts immediately.
  const stillLoading = residentFamily !== currentFamily;
  setGenStatus(
    "Generating (" + currentFamily + ")..." +
      (stillLoading ? " switching models, can take a minute or two." : ""),
    "status-pending",
  );

  try {
    await ensureFamilyResident(currentFamily);
    updateOutputHint();
    let result;
    if (isVideoGenerate() && currentFamily === "vqgan") result = await generateVqganVideo();
    else if (isVideoGenerate() && currentFamily === "deepdream") result = await generateDeepdreamVideo();
    else if (isVideoGenerate() && currentFamily === "style") result = await generateStyleVideo();
    else if (currentFamily === "vqgan") result = await generateVqgan();
    else if (currentFamily === "deepdream") result = await generateDeepdream();
    else result = await generateStyle();
    lastSeed = result.seed;
    seedValueEl.value = String(result.seed);
    metadataOutEl.value = JSON.stringify(result.metadata);
    setGenStatus("Done — seed " + result.seed, "status-ok");
  } catch (err) {
    const message = (err as Error).message;
    // Covers both the video-job path (VIDEO_CANCELLED_MESSAGE, set by
    // pollVideoJob) and single-image generation's AbortedError, whose exact
    // wording ("Generation aborted") comes straight from blobvision_cancel
    // on the backend — matching on the word rather than an exact string so
    // either phrasing reads as a cancel, not a failure.
    if (/cancell?ed|aborted/i.test(message)) setGenStatus("Cancelled.", "status-error");
    else setGenStatus("Generation failed: " + message, "status-error");
  } finally {
    generateBtn.disabled = false;
  }
}

function currentVideoFrameStep(): number {
  return Number(Array.from(videoFrameStepRadios).find((r) => r.checked)?.value ?? "4");
}

// Picks whichever fixed aspect preset the source's own ratio is closest to
// (log-scale comparison so 16:9 vs 9:16 aren't equally "close" to a square
// source) — video has no "custom" output size, so this is the only aspect
// auto-detection it gets (mirrors blobvision_ui.py's guess_aspect_from_size).
function guessAspectFromSize(width: number, height: number): "1:1" | "16:9" | "9:16" {
  const ratio = width / height;
  const candidates: ["1:1" | "16:9" | "9:16", number][] = [
    ["1:1", 1], ["16:9", 16 / 9], ["9:16", 9 / 16],
  ];
  let best: "1:1" | "16:9" | "9:16" = "1:1";
  let bestDiff = Infinity;
  for (const [name, targetRatio] of candidates) {
    const diff = Math.abs(Math.log(ratio) - Math.log(targetRatio));
    if (diff < bestDiff) {
      bestDiff = diff;
      best = name;
    }
  }
  return best;
}

// --- video trim-range slider ---

function formatTimecode(seconds: number): string {
  const s = Math.max(0, seconds);
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  return m + ":" + rem.toFixed(1).padStart(4, "0");
}

// True once either handle has moved off the full [0, duration] extent —
// drives whether encode_from/encode_to are actually sent to /generate (see
// the three video generate functions), replacing the old explicit "Limit
// encode range" checkbox: an untouched slider already means "no limit".
function videoRangeIsLimited(): boolean {
  if (videoDuration <= 0) return false;
  const from = Number(videoEncodeFromEl.value) || 0;
  const to = Number(videoEncodeToEl.value) || 0;
  return from > 0.05 || to < videoDuration - 0.05;
}

function updateVideoRangeVisual() {
  const from = Number(videoEncodeFromEl.value) || 0;
  const to = Number(videoEncodeToEl.value) || 0;
  const dur = videoDuration || 1;
  const fromPct = clamp((from / dur) * 100, 0, 100);
  const toPct = clamp((to / dur) * 100, 0, 100);
  videoRangeHandleFromEl.style.left = fromPct + "%";
  videoRangeHandleToEl.style.left = toPct + "%";
  videoRangeFillEl.style.left = fromPct + "%";
  videoRangeFillEl.style.width = Math.max(0, toPct - fromPct) + "%";
  videoRangeValueEl.textContent = formatTimecode(from) + " – " + formatTimecode(to);
}

function resetVideoRange() {
  videoDuration = 0;
  videoEncodeFromEl.value = "0";
  videoEncodeToEl.value = "0";
  updateVideoRangeVisual();
}

// Drags either handle via pointer capture; shows a timecode tooltip pinned
// above the handle while dragging, hides it on release. `which` picks which
// underlying hidden input (from/to) gets updated — the visible slider is
// purely a view over those two inputs, so the existing /generate call sites
// that read .value off them didn't need to change.
function startVideoRangeDrag(which: "from" | "to", downEvent: PointerEvent) {
  if (videoDuration <= 0) return;
  downEvent.preventDefault();
  const handle = which === "from" ? videoRangeHandleFromEl : videoRangeHandleToEl;
  try {
    // Best-effort: keeps the drag smooth if the pointer briefly leaves the
    // small handle. Not required for correctness — the listeners below are
    // on document, not the handle, so dragging still works if this throws
    // (seen with synthetic/edge-case pointer ids in some webviews).
    handle.setPointerCapture(downEvent.pointerId);
  } catch {
    /* ignore */
  }
  videoRangeTooltipEl.hidden = false;

  const onMove = (moveEvent: PointerEvent) => {
    const rect = videoRangeSliderEl.getBoundingClientRect();
    const pct = clamp((moveEvent.clientX - rect.left) / rect.width, 0, 1);
    let seconds = pct * videoDuration;
    if (which === "from") {
      const toVal = Number(videoEncodeToEl.value) || videoDuration;
      seconds = Math.min(seconds, toVal);
      videoEncodeFromEl.value = seconds.toFixed(1);
    } else {
      const fromVal = Number(videoEncodeFromEl.value) || 0;
      seconds = Math.max(seconds, fromVal);
      videoEncodeToEl.value = seconds.toFixed(1);
    }
    updateVideoRangeVisual();
    videoRangeTooltipEl.style.left = handle.style.left;
    videoRangeTooltipEl.textContent = formatTimecode(seconds);
  };
  const onUp = () => {
    videoRangeTooltipEl.hidden = true;
    document.removeEventListener("pointermove", onMove);
    document.removeEventListener("pointerup", onUp);
  };
  document.addEventListener("pointermove", onMove);
  document.addEventListener("pointerup", onUp);
  onMove(downEvent);
}

function updateVideoProbeHint() {
  if (!videoProbeInfo) {
    videoProbeHintEl.textContent = "";
    resetVideoRange();
    return;
  }
  const { duration_label, fps, width, height, duration, long_warning } = videoProbeInfo;
  videoProbeHintEl.textContent =
    duration_label + " · " + fps.toFixed(1) + " fps · " + width + "x" + height +
    (long_warning ? " — long clip, this will take a while." : "");
  videoDuration = duration;
  videoEncodeFromEl.max = String(duration);
  videoEncodeToEl.max = String(duration);
  videoEncodeFromEl.value = "0";
  videoEncodeToEl.value = duration.toFixed(1);
  updateVideoRangeVisual();
  setAspect(guessAspectFromSize(width, height));
}

// Deletes a previously-uploaded source clip from the backend's uploads/
// scratch dir — fire-and-forget (matches dropStartupPriority()'s pattern):
// a failed cleanup shouldn't block the UI, it just leaves that one file for
// the app-close sweep to catch instead.
function discardVideoUpload(path: string) {
  const body = new FormData();
  body.set("path", path);
  fetchJson("/video/discard", { method: "POST", body }).catch((err) => {
    console.error("Failed to discard uploaded video:", err);
  });
}

// Fires right after a file is picked/dropped (not deferred to Generate) so
// duration/fps/long-clip warning are visible before committing — mirrors
// blobvision_ui.py's on_video_upload. Re-sent by generate as just a
// filename (see /video/upload's docstring), so this only re-uploads the
// whole clip when the file actually changes.
async function uploadAndProbeVideo(file: File) {
  // A previous clip loaded into this same drop zone (picked/dropped again
  // without clicking × first) would otherwise be orphaned in uploads/.
  if (videoUploadedPath) discardVideoUpload(videoUploadedPath);
  videoUploadedPath = null;
  videoProbeInfo = null;
  videoProbeHintEl.textContent = "Uploading & reading video...";
  try {
    const body = new FormData();
    body.set("video", file);
    const uploadResult = await fetchJson("/video/upload", { method: "POST", body });
    const uploadedPath: string = uploadResult.path;
    videoUploadedPath = uploadedPath;
    videoProbeInfo = await fetchJson("/video/probe?path=" + encodeURIComponent(uploadedPath));
    updateVideoProbeHint();
  } catch (err) {
    videoProbeHintEl.textContent = "Could not read video: " + (err as Error).message;
  }
}

async function refreshVideoCodecsGate() {
  try {
    const status = await fetchJson("/video/codecs/status");
    videoCodecsReady = !!status.ready;
  } catch {
    videoCodecsReady = false;
  }
  videoCodecsGateEl.hidden = videoCodecsReady;
  // The keyframe-thinning/encode-range fields stay hidden until codecs are
  // actually installed — mirrors blobvision_ui.py's _video_mode_ui_updates
  // (show_video = mode==="video" && codecs ready). The drop zone itself
  // always works regardless (uploading/probing a clip doesn't need RIFE/
  // ffmpeg — only the actual generate step does).
  [videoFrameStepFieldEl, videoRangeFieldEl].forEach((field) => {
    field.hidden = !videoCodecsReady;
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Matched in onGenerate's catch block to show "Cancelled." instead of
// "Generation failed: ..." — a user-requested cancel isn't a failure.
const VIDEO_CANCELLED_MESSAGE = "Video generation cancelled.";

// No percent/ETA from the backend (see VideoJob in blobvision_api.py) —
// just whatever human-readable lines the engine's on_progress callback has
// appended so far, so the status area shows the latest one ("Video:
// blobify 4/12...") the same way the old Gradio console-tail did.
async function pollVideoJob(jobId: string): Promise<{ output_url: string; metadata: unknown }> {
  for (;;) {
    const status = await fetchJson("/video/status/" + jobId);
    const lastLine: string | undefined = status.lines?.[status.lines.length - 1];
    if (lastLine) setGenStatus(lastLine, "status-pending");
    if (status.state === "done") return { output_url: status.output_url, metadata: status.metadata };
    if (status.state === "error") throw new Error(status.error || "Video generation failed.");
    if (status.state === "cancelled") throw new Error(VIDEO_CANCELLED_MESSAGE);
    await sleep(1000);
  }
}

async function generateVqganVideo() {
  const prompt = promptEl.value.trim();
  if (!videoUploadedPath) throw new Error("Upload a source video first (or wait for the upload to finish).");

  if (videoProbeInfo?.long_warning) {
    const ok = window.confirm(
      "This clip is " + videoProbeInfo.duration_label + " long — processing may take a while. Continue?",
    );
    if (!ok) throw new Error("Cancelled.");
  }

  await refreshVideoCodecsGate();
  if (!videoCodecsReady) throw new Error("Video codecs not installed — click Install codecs first.");

  const aspect = currentAspect();
  const seed = resolveSeed();
  const body = new FormData();
  body.set("path", videoUploadedPath);
  body.set("prompt", prompt);
  if (negativePromptEl.value.trim()) body.set("negative_prompt", negativePromptEl.value.trim());
  body.set("aspect", aspect === "custom" ? "1:1" : aspect);
  body.set("iterations", iterationsEl.value);
  body.set("denoise_fidelity", denoiseEl.value);
  body.set("seed", String(seed));
  body.set("frame_step", String(currentVideoFrameStep()));
  if (videoRangeIsLimited()) {
    body.set("use_encode_range", "true");
    body.set("encode_from", videoEncodeFromEl.value);
    body.set("encode_to", videoEncodeToEl.value);
  }

  const startResult = await fetchJson("/video/generate", { method: "POST", body });
  const jobId: string = startResult.job_id;
  const done = await pollVideoJob(jobId);
  showOutputVideo(done.output_url);
  return { seed, metadata: done.metadata };
}

// Same shape as generateVqganVideo but posts to DeepDream's own video
// endpoint — no prompt (DeepDream's img2img pass isn't CLIP/prompt-guided),
// intensity/layer instead of iterations/denoise_fidelity.
async function generateDeepdreamVideo() {
  if (!videoUploadedPath) throw new Error("Upload a source video first (or wait for the upload to finish).");

  if (videoProbeInfo?.long_warning) {
    const ok = window.confirm(
      "This clip is " + videoProbeInfo.duration_label + " long — processing may take a while. Continue?",
    );
    if (!ok) throw new Error("Cancelled.");
  }

  await refreshVideoCodecsGate();
  if (!videoCodecsReady) throw new Error("Video codecs not installed — click Install codecs first.");

  const aspect = currentAspect();
  const seed = resolveSeed();
  const body = new FormData();
  body.set("path", videoUploadedPath);
  body.set("aspect", aspect === "custom" ? "1:1" : aspect);
  body.set("intensity", ddIntensityEl.value);
  body.set("layer", currentDdLayer());
  body.set("seed", String(seed));
  body.set("frame_step", String(currentVideoFrameStep()));
  if (videoRangeIsLimited()) {
    body.set("use_encode_range", "true");
    body.set("encode_from", videoEncodeFromEl.value);
    body.set("encode_to", videoEncodeToEl.value);
  }

  const startResult = await fetchJson("/deepdream/video/generate", { method: "POST", body });
  const jobId: string = startResult.job_id;
  const done = await pollVideoJob(jobId);
  showOutputVideo(done.output_url);
  return { seed, metadata: done.metadata };
}

// --- SAM2 magic wand selection (Phase 1: image mode only, VQGAN) ---
// Clicking the init image (or "Magic wand" explicitly) opens a larger
// overlay (#sam-overlay) — all actual clicking-to-select and the Segment
// size/Soften/Expand-shrink/Invert/Cut/Keep controls live only there, not
// in the sidebar (the small preview is too fiddly a click target, and the
// sidebar has no room for controls that only matter while actively
// selecting). Opening the overlay also kicks off ensureSamEmbedded() right
// away (not just on the first click) so the ~10s first-use SAM2 model load
// is already underway by the time the user clicks. Closing the overlay
// (×, the backdrop, or Escape — all equivalent) always keeps whatever was
// built; there's no separate "Apply" step, since the moment the user has
// placed a mask/cut, wanting it used on Generate is the whole point.
// Only "Clear mask" (sidebar row, or the overlay's own "Clear selection")
// removes everything — both the live blobify selection AND any
// accumulated Cut/Keep zones, treated as one combined "mask" from the
// user's point of view even though they're composited server-side as two
// separate passes (see blobvision_api.py's /generate: mask/invert first,
// then cut_mask/cut_mode/cut_color last).
//
// Each click queries SAM2 for the segment under that point and always
// adds it to the accumulation; Ctrl+click subtracts one instead — combined
// into one mask server-side in the order clicked (see
// blobvision_sam.py's Sam2Engine.segment), NOT a single multi-point
// prompt, since these are meant to be distinct objects/carve-outs, not
// extra hints about one region. Erosion/feather refinement of that mask
// happens entirely client-side (see refineSamMask) so slider drags never
// round-trip to the backend.
//
// Cut/Keep commit the CURRENT selection into a separate, persistent
// "what's been cut" accumulator (samCutAccumCanvas) that's rendered live
// right in the overlay (and the sidebar thumbnail) as an opaque
// checkerboard (alpha mode) or solid fill (color mode) — not just a pink
// tint, since the whole point is to preview what will actually disappear.
// Committing does NOT close the overlay: the live selection is cleared so
// the user can immediately pick a different zone and Cut/Keep that too,
// building up the cutout across several clicks in one sitting.

interface SamPoint {
  x: number;
  y: number;
  label: number; // 1 = union into the accumulated selection, 0 = subtract from it
  size: number; // -1..1, frozen at click time — picks small/medium/large among SAM2's per-point candidates
}

let samPoints: SamPoint[] = [];
let samEmbedToken: string | null = null;
let samEmbeddedFileKey: string | null = null;
let samRawMaskImg: HTMLImageElement | null = null;
let samNaturalW = 0;
let samNaturalH = 0;
let samInvertOn = false;
// Guards against a second click firing while the first is still in flight —
// the very first click of a session can take ~10s (SAM2's lazy model load),
// and without this guard a second click during that window can race the
// embed/segment calls against each other (the backend's embed token gets
// overwritten mid-flight, so the first click's own segment request then
// fails with a stale-token error that has nothing useful to retry).
let samRequestInFlight = false;
// Shared by both the proactive preload (openSamOverlay) and the click
// handler's own ensureSamEmbedded() call, so whichever fires second just
// awaits the same in-flight request instead of racing a second /sam2/embed.
let samEmbedPromise: Promise<boolean> | null = null;
// Committed Cut/Keep zones, UNIONED together as each Cut/Keep is pressed
// (see commitCutMask) — grows across multiple commits within the same
// overlay session, unlike samPoints above which resets after each commit.
// Kept at natural resolution; Keep's inversion is baked in per-commit
// before unioning, so this canvas's own pixel values always directly mean
// "cut here" — nothing further to invert when reading it back.
let samCutAccumCanvas: HTMLCanvasElement | null = null;
// Tracks the has-a-mask state across renders so the Show-mask toggle can
// re-default to checked exactly on the empty→non-empty transition,
// without fighting a user who deliberately unchecked it mid-session.
let samMaskWasPresent = false;

function samFileKey(file: File): string {
  return file.name + ":" + file.size + ":" + file.lastModified;
}

function setSamStatus(text: string, isError: boolean) {
  samStatusEl.textContent = text;
  samStatusEl.classList.toggle("status-error", isError);
  samStatusEl.classList.toggle("sam-status-bubble", text.length > 0);
}

function hasSamMask(): boolean {
  return samPoints.length > 0 || samCutAccumCanvas !== null;
}

// Keeps every "is there a mask right now" affordance in sync in one
// place: the sidebar's own combined Clear, Cut/Keep's enabled state (they
// act on the LIVE selection specifically), the overlay's own Unselect
// (live selection only) and Reset (committed cuts only) buttons — each
// enabled independently based on what actually exists to undo — the
// Show-mask row's visibility, and the accent ring around the drop zone (a
// pink wash can all but vanish against a warm photo, but a border doesn't
// depend on the image's own colors to read as "a mask is active here").
function updateSamMaskControls() {
  const has = hasSamMask();
  samClearBtnEl.disabled = !has;
  samOverlayClearBtnEl.disabled = samPoints.length === 0;
  samOverlayResetBtnEl.disabled = samCutAccumCanvas === null;
  samCutBtnEl.disabled = samPoints.length === 0;
  samKeepBtnEl.disabled = samPoints.length === 0;
  samExportCutBtnEl.disabled = samCutAccumCanvas === null || !hasOutputImage();
  initImageDropEl.classList.toggle("has-mask", has);
  samShowMaskRowEl.hidden = !has;
  if (has && !samMaskWasPresent) samShowMaskToggleEl.checked = true;
  samMaskWasPresent = has;
}

// Clears just the current click-in-progress selection, leaving any
// already-committed Cut/Keep zones untouched (the overlay's "Unselect").
function unselectSam() {
  samPoints = [];
  samRawMaskImg = null;
  updateSamMaskControls();
  renderSamMaskPreview();
}

// Restores pixels removed by Cut/Keep, leaving the current live selection
// untouched (the overlay's "Reset").
function resetCutMask() {
  samCutAccumCanvas = null;
  updateSamMaskControls();
  renderSamMaskPreview();
}

// Sidebar's single combined action — wipes both the live selection and
// any committed cuts, back to a fully blank slate.
function clearSamMask() {
  samPoints = [];
  samRawMaskImg = null;
  samCutAccumCanvas = null;
  updateSamMaskControls();
  setSamStatus("", false);
  renderSamMaskPreview();
}

// Bakes the CURRENT selection (same refineSamMask pipeline as the live
// blobify mask, at natural resolution — Keep inverted first) and unions it
// into the persistent cut accumulator, then clears just the live selection
// so the next clicks start fresh for whatever gets cut/kept next. Leaves
// the overlay open.
async function commitCutMask(keep: boolean) {
  if (samPoints.length === 0 || !samRawMaskImg || !samNaturalW || !samNaturalH) return;
  const refined = refineSamMask(samRawMaskImg, samNaturalW, samNaturalH, Number(samSoftenEl.value), Number(samExpandEl.value));
  if (keep) {
    const rctx = refined.getContext("2d")!;
    const data = rctx.getImageData(0, 0, samNaturalW, samNaturalH);
    for (let i = 0; i < data.data.length; i += 4) {
      const v = 255 - data.data[i];
      data.data[i] = v;
      data.data[i + 1] = v;
      data.data[i + 2] = v;
    }
    rctx.putImageData(data, 0, 0);
  }
  if (!samCutAccumCanvas) {
    samCutAccumCanvas = document.createElement("canvas");
    samCutAccumCanvas.width = samNaturalW;
    samCutAccumCanvas.height = samNaturalH;
  }
  const actx = samCutAccumCanvas.getContext("2d")!;
  actx.globalCompositeOperation = "lighter"; // additive-clamped == max/OR for near-binary masks
  actx.drawImage(refined, 0, 0);
  actx.globalCompositeOperation = "source-over";
  samPoints = [];
  samRawMaskImg = null;
  updateSamMaskControls();
  renderSamMaskPreview();
}

// /sam2/segment returns a PIL mode-"L" PNG — a bare grayscale VALUE per
// pixel, no alpha channel at all (drawn into a canvas, that value ends up
// replicated into R/G/B with alpha uniformly 255). commitCutMask
// accumulates straight from that (see morphMask/blurEdgeReplicate, both
// plain RGB pixel edits), so samCutAccumCanvas carries the mask the same
// way: as a grayscale VALUE, not as alpha. Canvas's destination-out/
// destination-in only ever read a source's ALPHA channel, so this converts
// that grayscale value into alpha first — after which the compositing below
// reproduces the backend's Image.composite(...) math exactly.
function grayscaleMaskToAlpha(source: CanvasImageSource, w: number, h: number): HTMLCanvasElement {
  const tmp = document.createElement("canvas");
  tmp.width = w;
  tmp.height = h;
  const tctx = tmp.getContext("2d")!;
  tctx.drawImage(source, 0, 0, w, h);
  const imgData = tctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i + 3] = d[i]; // alpha = grayscale value (R === G === B already)
  }
  tctx.putImageData(imgData, 0, 0);
  return tmp;
}

// Applies the accumulated Cut/Keep selection straight to the last generated
// image and downloads the result, entirely client-side — no new VQGAN run.
// The cut is pure pixel compositing (identical math to the backend's own
// _apply_cut_mask, just done here instead of round-tripping a fresh
// /generate call), so there's nothing about it that actually needs the
// model. Only meaningful once something's been generated — same
// "cut always applies to a generation's output, never the raw upload"
// rule the backend's cut_mask param already follows.
async function exportCutFromOutput() {
  if (!samCutAccumCanvas || !hasOutputImage()) return;
  samExportCutBtnEl.disabled = true;
  try {
    const blob = await fetchAsBlob(outputImageEl.src);
    const bitmap = await createImageBitmap(blob);
    const canvas = document.createElement("canvas");
    canvas.width = bitmap.width;
    canvas.height = bitmap.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

    // The mask was captured at the init image's own resolution, which can
    // differ from the output's (upscale, aspect-driven resize) — stretched
    // to fit here exactly like the backend's own cut_img.resize(base.size).
    const alphaMask = grayscaleMaskToAlpha(samCutAccumCanvas, samCutAccumCanvas.width, samCutAccumCanvas.height);
    const cutMode = document.querySelector<HTMLInputElement>('input[name="sam-cut-mode"]:checked')?.value || "alpha";
    if (cutMode === "color") {
      // fill blended in proportional to the mask value — matches
      // Image.composite(fill, base, cut_img).
      const fillCanvas = document.createElement("canvas");
      fillCanvas.width = canvas.width;
      fillCanvas.height = canvas.height;
      const fctx = fillCanvas.getContext("2d")!;
      fctx.fillStyle = samCutColorEl.value;
      fctx.fillRect(0, 0, fillCanvas.width, fillCanvas.height);
      fctx.globalCompositeOperation = "destination-in";
      fctx.drawImage(alphaMask, 0, 0, fillCanvas.width, fillCanvas.height);
      ctx.drawImage(fillCanvas, 0, 0);
    } else {
      // Punches transparency in proportion to the mask value — matches
      // new_alpha = base_alpha * (1 - cut/255): destination-out erases the
      // destination's alpha by exactly the drawn source's own alpha, and
      // the base image starts fully opaque, so this reproduces that
      // subtraction exactly.
      ctx.globalCompositeOperation = "destination-out";
      ctx.drawImage(alphaMask, 0, 0, canvas.width, canvas.height);
    }

    const outBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob((b) => resolve(b), "image/png"));
    if (!outBlob) throw new Error("Could not encode the cut image.");
    const blobUrl = URL.createObjectURL(outBlob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filenameFromUrl(outputImageEl.src).replace(/\.png$/i, "") + "_cut.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
  } catch (err) {
    setGenStatus("Export cut failed: " + (err as Error).message, "status-error");
  } finally {
    samExportCutBtnEl.disabled = samCutAccumCanvas === null || !hasOutputImage();
  }
}

function updateSamToggleVisibility() {
  const show = initFileKind === "image";
  samToggleRowEl.hidden = !show;
  if (!show) resetSamSelection();
}

function resetSamSelection() {
  samPoints = [];
  samEmbedToken = null;
  samEmbeddedFileKey = null;
  samEmbedPromise = null;
  samRawMaskImg = null;
  samCutAccumCanvas = null;
  samRequestInFlight = false;
  samMaskWasPresent = false;
  updateSamMaskControls();
  setSamStatus("", false);
  closeSamOverlay();
  const ctx = samMaskCanvasEl.getContext("2d");
  if (ctx) ctx.clearRect(0, 0, samMaskCanvasEl.width, samMaskCanvasEl.height);
}

// Loads (or reuses) the SAM2 embedding for the current init image. Called
// proactively the moment the overlay opens — not just on the first click —
// so the ~10s first-use model load is already underway by the time the
// user actually clicks a point, instead of only starting then.
async function ensureSamEmbedded(): Promise<boolean> {
  const file = initImageEl.files?.[0];
  if (!file) return false;
  const key = samFileKey(file);
  if (samEmbedToken && samEmbeddedFileKey === key) return true;
  if (samEmbedPromise) return samEmbedPromise;
  setSamStatus("Loading SAM2 (first use can take ~10s)…", false);
  samEmbedPromise = (async () => {
    try {
      // Retries only on a network-level failure (fetch() itself rejecting
      // with TypeError — "Failed to fetch") — e.g. the overlay opened
      // fast enough that the backend process hadn't even finished starting
      // up yet (heavy ML imports can easily take 10s+ before it's even
      // listening — the same order of magnitude waitForEngine() already
      // budgets for via HEALTH_TIMEOUT_MS, so this reuses that budget
      // rather than a shorter guess that wouldn't actually cover it). A
      // real HTTP error (bad request, SAM2 weights missing, etc.) surfaces
      // from fetchJson as a plain Error, not a TypeError, and is never
      // worth retrying blind, so it's re-thrown immediately instead of
      // burning the retry budget on it.
      const deadline = Date.now() + HEALTH_TIMEOUT_MS;
      let lastErr: unknown;
      for (;;) {
        try {
          const body = new FormData();
          body.set("image", file);
          const embedResult = await fetchJson("/sam2/embed", { method: "POST", body });
          samEmbedToken = embedResult.token;
          samEmbeddedFileKey = key;
          setSamStatus("", false);
          return true;
        } catch (err) {
          if (!(err instanceof TypeError)) throw err;
          lastErr = err;
          if (Date.now() >= deadline) throw lastErr;
          setSamStatus("Waiting for BlobVision's backend to finish starting…", false);
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    } catch (err) {
      setSamStatus("Magic wand: " + (err as Error).message, true);
      return false;
    } finally {
      samEmbedPromise = null;
    }
  })();
  return samEmbedPromise;
}

function openSamOverlay() {
  if (initFileKind !== "image" || !initImageEl.files?.[0]) return;
  samOverlayImageEl.src = initImagePreviewEl.src;
  samOverlayEl.hidden = false;
  samWandBtnEl.classList.add("active");
  if (samOverlayImageEl.complete && samOverlayImageEl.naturalWidth) {
    positionSamOverlayCanvas();
  } else {
    samOverlayImageEl.addEventListener("load", positionSamOverlayCanvas, { once: true });
  }
  void ensureSamEmbedded();
}

function closeSamOverlay() {
  samOverlayEl.hidden = true;
  samWandBtnEl.classList.remove("active");
  samOverlayMaskCanvasEl.classList.remove("sam-subtract-cursor");
  // Re-paint the small sidebar indicator so the mask/cut stays visibly
  // applied on the drag-and-drop preview after the overlay closes, not
  // just while it's open.
  if (hasSamMask()) positionSamCanvas();
}

// Sizes/positions the small passive sidebar indicator to exactly match the
// preview image's own "object-fit: contain" letterboxed rect (mirrors the
// meme studio's studioViewportToCanvasPoint trick: make the target
// element's own rect BE the coordinate space). Reads initImagePreviewEl's
// own rendered rect, not the drop zone's, so parent padding/box-sizing
// can't throw it off.
function positionSamCanvas() {
  if (!initImagePreviewEl.naturalWidth || !initImagePreviewEl.naturalHeight) return;
  const box = initImagePreviewEl.getBoundingClientRect();
  const dropBox = initImageDropEl.getBoundingClientRect();
  const natW = initImagePreviewEl.naturalWidth;
  const natH = initImagePreviewEl.naturalHeight;
  const scale = Math.min(box.width / natW, box.height / natH);
  const dispW = natW * scale;
  const dispH = natH * scale;
  samMaskCanvasEl.style.width = dispW + "px";
  samMaskCanvasEl.style.height = dispH + "px";
  samMaskCanvasEl.style.left = box.left - dropBox.left + (box.width - dispW) / 2 + "px";
  samMaskCanvasEl.style.top = box.top - dropBox.top + (box.height - dispH) / 2 + "px";
  samMaskCanvasEl.width = natW;
  samMaskCanvasEl.height = natH;
  renderSamMaskPreview();
}

// Same idea for the overlay's bigger canvas — simpler here since the
// overlay image is sized by its own natural aspect ratio (max-width/
// max-height, no fixed-aspect letterboxing to account for), so its own
// rendered rect can be copied onto the canvas directly.
function positionSamOverlayCanvas() {
  if (!samOverlayImageEl.naturalWidth || !samOverlayImageEl.naturalHeight) return;
  samNaturalW = samOverlayImageEl.naturalWidth;
  samNaturalH = samOverlayImageEl.naturalHeight;
  const wrapRect = samOverlayCanvasWrapEl.getBoundingClientRect();
  const imgRect = samOverlayImageEl.getBoundingClientRect();
  samOverlayMaskCanvasEl.style.width = imgRect.width + "px";
  samOverlayMaskCanvasEl.style.height = imgRect.height + "px";
  samOverlayMaskCanvasEl.style.left = imgRect.left - wrapRect.left + "px";
  samOverlayMaskCanvasEl.style.top = imgRect.top - wrapRect.top + "px";
  samOverlayMaskCanvasEl.width = samNaturalW;
  samOverlayMaskCanvasEl.height = samNaturalH;
  renderSamMaskPreview();
}

async function onSamOverlayCanvasClick(e: MouseEvent) {
  if (samRequestInFlight) return;
  const rect = samOverlayMaskCanvasEl.getBoundingClientRect();
  const x = ((e.clientX - rect.left) / rect.width) * samOverlayMaskCanvasEl.width;
  const y = ((e.clientY - rect.top) / rect.height) * samOverlayMaskCanvasEl.height;
  if (x < 0 || y < 0 || x > samOverlayMaskCanvasEl.width || y > samOverlayMaskCanvasEl.height) return;
  const size = Number(samSizeEl.value);
  // Plain click always adds to the accumulated selection — Clear mask is
  // the explicit way to start over, so a click never has to silently wipe
  // prior points. Ctrl+click subtracts instead (users reach for Ctrl more
  // reflexively than Shift here) — an explicit modifier rather than
  // hit-testing whatever's already selected, since that made add/remove
  // feel like it depended on where exactly you clicked instead of a
  // deliberate choice.
  const candidate = [...samPoints];
  const label = e.ctrlKey ? 0 : 1;
  candidate.push({ x, y, label, size });
  await requestSamSegment(candidate);
}

// Takes the CANDIDATE point list rather than mutating samPoints directly —
// a failed request (or one ignored by the in-flight guard) must never
// corrupt an already-good selection, so samPoints only gets updated once
// the request actually succeeds.
async function requestSamSegment(candidatePoints: SamPoint[]) {
  const file = initImageEl.files?.[0];
  if (!file || candidatePoints.length === 0) return;
  samRequestInFlight = true;
  setSamStatus(samEmbedToken ? "Selecting…" : "Loading SAM2 (first use can take ~10s)…", false);
  try {
    if (!(await ensureSamEmbedded())) return;
    const res = await fetch(API_BASE + "/sam2/segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token: samEmbedToken, points: candidatePoints }),
    });
    if (!res.ok) throw new Error("Segmentation failed (" + res.status + ")");
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error("Failed to load mask"));
      img.src = url;
    });
    samRawMaskImg = img;
    samPoints = candidatePoints;
    updateSamMaskControls();
    setSamStatus("", false);
    positionSamCanvas();
    renderSamMaskPreview();
  } catch (err) {
    setSamStatus("Magic wand: " + (err as Error).message, true);
  } finally {
    samRequestInFlight = false;
  }
}

// Blurs `source` after padding it with edge-replicated pixels, so the blur
// "sees" a continuation of the edge content instead of an implicit
// transparent/black boundary — otherwise erosion or feathering would
// falsely eat into the image's own frame border on an inverted (background)
// selection, not just around the real subject. Shared by the morphological
// expand/shrink pass and the final soften pass below.
function blurEdgeReplicate(source: CanvasImageSource, w: number, h: number, radiusPx: number): HTMLCanvasElement {
  if (radiusPx <= 0) {
    const copy = document.createElement("canvas");
    copy.width = w;
    copy.height = h;
    copy.getContext("2d")!.drawImage(source, 0, 0, w, h);
    return copy;
  }
  const pad = Math.min(80, Math.ceil(radiusPx) + 8);
  const padded = document.createElement("canvas");
  padded.width = w + pad * 2;
  padded.height = h + pad * 2;
  const pctx = padded.getContext("2d")!;
  pctx.drawImage(source, pad, pad, w, h);
  pctx.drawImage(source, 0, 0, w, 1, pad, 0, w, pad); // top
  pctx.drawImage(source, 0, h - 1, w, 1, pad, pad + h, w, pad); // bottom
  pctx.drawImage(source, 0, 0, 1, h, 0, pad, pad, h); // left
  pctx.drawImage(source, w - 1, 0, 1, h, pad + w, pad, pad, h); // right
  pctx.drawImage(source, 0, 0, 1, 1, 0, 0, pad, pad); // top-left corner
  pctx.drawImage(source, w - 1, 0, 1, 1, pad + w, 0, pad, pad); // top-right corner
  pctx.drawImage(source, 0, h - 1, 1, 1, 0, pad + h, pad, pad); // bottom-left corner
  pctx.drawImage(source, w - 1, h - 1, 1, 1, pad + w, pad + h, pad, pad); // bottom-right corner

  const blurred = document.createElement("canvas");
  blurred.width = padded.width;
  blurred.height = padded.height;
  const bctx = blurred.getContext("2d")!;
  bctx.filter = "blur(" + radiusPx + "px)";
  bctx.drawImage(padded, 0, 0);

  const result = document.createElement("canvas");
  result.width = w;
  result.height = h;
  result.getContext("2d")!.drawImage(blurred, pad, pad, w, h, 0, 0, w, h);
  return result;
}

// True morphological dilate (expandPx > 0) / erode (expandPx < 0) of the
// mask's boundary — the After Effects "Spread"/"Choke" trick: blur by the
// requested radius, then threshold back to a hard binary mask. A LOW
// threshold after the blur means even a faint bleed from |expandPx| pixels
// away now counts as selected, growing the boundary outward; a HIGH
// threshold keeps only the surviving "core", shrinking it inward. This
// moves the boundary itself, unlike a flat brightness/opacity bias (the
// previous approach) which raised every pixel's value uniformly and made
// the whole selection look hazier everywhere instead of actually growing.
function morphMask(sourceImg: HTMLImageElement, w: number, h: number, expandPx: number): HTMLCanvasElement {
  if (expandPx === 0) {
    const copy = document.createElement("canvas");
    copy.width = w;
    copy.height = h;
    copy.getContext("2d")!.drawImage(sourceImg, 0, 0, w, h);
    return copy;
  }
  // A blur of radius R moves the 12/243 threshold crossing by ~1.7*R px
  // (measured empirically on a straight edge), so scale the blur radius
  // down to make the slider's px value track the actual boundary shift.
  const blurred = blurEdgeReplicate(sourceImg, w, h, Math.abs(expandPx) * 0.6);
  const bctx = blurred.getContext("2d")!;
  const imgData = bctx.getImageData(0, 0, w, h);
  const d = imgData.data;
  const threshold = expandPx > 0 ? 12 : 243;
  for (let i = 0; i < d.length; i += 4) {
    const v = d[i] > threshold ? 255 : 0;
    d[i] = v;
    d[i + 1] = v;
    d[i + 2] = v;
  }
  bctx.putImageData(imgData, 0, 0);
  return blurred;
}

// Two-stage refinement: first move the mask boundary via true dilate/erode
// (morphMask), then apply the soft edge (plain blur, no thresholding) on
// top of the repositioned boundary. Keeps "Expand/shrink" (where the edge
// sits) and "Soften" (how gradual the edge is) fully independent.
function refineSamMask(sourceImg: HTMLImageElement, w: number, h: number, featherPx: number, expandBias: number): HTMLCanvasElement {
  const morphed = morphMask(sourceImg, w, h, expandBias);
  return blurEdgeReplicate(morphed, w, h, featherPx);
}

// Live preview: cheap enough to re-run on every slider tick since it's
// pure client-side canvas work. Paints the small passive sidebar indicator
// always, and the bigger overlay canvas too whenever the overlay is open.
// Skipped entirely (canvas left blank, showing the plain original image
// underneath) when the Show-mask toggle is off.
function renderSamMaskPreview() {
  paintSamMaskCanvas(samMaskCanvasEl);
  if (!samOverlayEl.hidden) paintSamMaskCanvas(samOverlayMaskCanvasEl);
}

function hexToRgb(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(hex) || [];
  return [parseInt(m[1] || "00", 16), parseInt(m[2] || "ff", 16), parseInt(m[3] || "00", 16)];
}

// Draws two layers, in order: (1) the committed Cut/Keep accumulator — an
// OPAQUE checkerboard (alpha mode) or solid fill (color mode), since the
// whole point is to preview what will actually disappear, not just tint
// it; (2) the live, not-yet-committed selection as the usual pink tint on
// top. Both read their own alpha straight from the (possibly soft-edged)
// grayscale mask value, so Soften/Expand still show up as a graded edge.
function paintSamMaskCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!w || !h || !samShowMaskToggleEl.checked) return;

  const out = ctx.createImageData(w, h);

  // Live (uncommitted) selection drawn FIRST — it's just a preview of what
  // a future Cut/Keep/Generate would do with it. The committed cut
  // accumulator is drawn SECOND, unconditionally overwriting, so it always
  // wins visually: a pixel that's already been cut must stay looking cut
  // even if a later, unrelated click's segment happens to overlap it (e.g.
  // a soft/blurred edge from a "hair" pick bleeding over an already-cut
  // "face" zone) — otherwise the committed hole would flicker back to
  // looking uncut just from picking a new, unrelated selection nearby.
  if (samRawMaskImg && samPoints.length > 0) {
    const refined = refineSamMask(samRawMaskImg, w, h, Number(samSoftenEl.value), Number(samExpandEl.value));
    const rctx = refined.getContext("2d")!;
    const data = rctx.getImageData(0, 0, w, h);
    const invert = samInvertOn;
    // The small sidebar indicator is shown at thumbnail size against
    // whatever colors the source photo happens to have, where a subtle
    // tint can all but disappear — stronger alpha there than in the big
    // overlay (which is already plenty visible at full size).
    const alphaFactor = canvas === samMaskCanvasEl ? 0.6 : 0.45;
    for (let i = 0; i < data.data.length; i += 4) {
      let a = data.data[i];
      if (invert) a = 255 - a;
      if (a === 0) continue;
      out.data[i] = 255;
      out.data[i + 1] = 92;
      out.data[i + 2] = 138;
      out.data[i + 3] = Math.round(a * alphaFactor);
    }
  }

  if (samCutAccumCanvas) {
    const resized = document.createElement("canvas");
    resized.width = w;
    resized.height = h;
    resized.getContext("2d")!.drawImage(samCutAccumCanvas, 0, 0, w, h);
    const cutData = resized.getContext("2d")!.getImageData(0, 0, w, h).data;
    const isColor = document.querySelector<HTMLInputElement>('input[name="sam-cut-mode"]:checked')?.value === "color";
    const [cr, cg, cb] = isColor ? hexToRgb(samCutColorEl.value) : [0, 0, 0];
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4;
        const a = cutData[i];
        if (a === 0) continue;
        if (isColor) {
          out.data[i] = cr;
          out.data[i + 1] = cg;
          out.data[i + 2] = cb;
        } else {
          const light = ((px >> 3) + (py >> 3)) % 2 === 0;
          const v = light ? 200 : 150;
          out.data[i] = v;
          out.data[i + 1] = v;
          out.data[i + 2] = v;
        }
        out.data[i + 3] = a;
      }
    }
  }

  ctx.putImageData(out, 0, 0);
}

// Bakes the current soften/expand refinement (NOT invert — sent as a
// separate flag so the server composites deterministically off the same
// un-inverted mask this preview is built from) into a PNG blob for upload
// at Generate time. Returns null if no selection is active.
async function bakeSamMaskForGenerate(): Promise<Blob | null> {
  if (samPoints.length === 0 || !samRawMaskImg || !samNaturalW || !samNaturalH) return null;
  const refined = refineSamMask(samRawMaskImg, samNaturalW, samNaturalH, Number(samSoftenEl.value), Number(samExpandEl.value));
  return new Promise((resolve, reject) => {
    refined.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Failed to encode mask"));
    }, "image/png");
  });
}

function showInitImagePreview(file: File) {
  initFileKind = "image";
  if (videoUploadedPath) discardVideoUpload(videoUploadedPath);
  videoUploadedPath = null;
  videoProbeInfo = null;
  videoProbeHintEl.textContent = "";
  resetVideoRange();
  resetSamSelection(); // new/replaced image invalidates any prior embedding
  initVideoPreviewEl.hidden = true;
  initVideoPreviewEl.src = "";
  const src = URL.createObjectURL(file);
  initImagePreviewEl.src = src;
  initImagePreviewEl.hidden = false;
  initImageHintEl.hidden = true;
  initImageDropEl.classList.add("has-image");
  // A dropped/picked image is unlikely to match one of the fixed aspect
  // presets — force custom so the backend fits the output to the source's
  // own ratio (budgeted server-side) instead of stretching/cropping to 1:1.
  setAspect("custom");
  updateVideoOnlyFieldsVisibility();
  updateSamToggleVisibility();
  syncOutputVisibilityForMode();
}

// Fires right after a video file is picked/dropped into the shared drop
// zone — mirrors showInitImagePreview but shows the <video> preview and
// kicks off the upload+probe (duration/fps/long-clip warning) immediately
// rather than deferring it to Generate, same as the old dedicated video
// drop zone did.
function showInitVideoPreview(file: File) {
  initFileKind = "video";
  initImagePreviewEl.hidden = true;
  initImagePreviewEl.src = "";
  initVideoPreviewEl.src = URL.createObjectURL(file);
  initVideoPreviewEl.hidden = false;
  initImageHintEl.hidden = true;
  initImageDropEl.classList.add("has-image");
  updateVideoOnlyFieldsVisibility();
  updateSamToggleVisibility(); // magic wand is image-only (Phase 1) — hides for video
  uploadAndProbeVideo(file);
}

function clearInitImagePreview() {
  initFileKind = null;
  initImagePreviewEl.hidden = true;
  initImagePreviewEl.src = "";
  initVideoPreviewEl.hidden = true;
  initVideoPreviewEl.src = "";
  initImageHintEl.hidden = false;
  initImageDropEl.classList.remove("has-image");
  if (videoUploadedPath) discardVideoUpload(videoUploadedPath);
  videoUploadedPath = null;
  videoProbeInfo = null;
  videoProbeHintEl.textContent = "";
  resetVideoRange();
  updateVideoOnlyFieldsVisibility();
  updateSamToggleVisibility();
  syncOutputVisibilityForMode();
}

function onInitImageChange() {
  const file = initImageEl.files?.[0];
  if (!file) {
    clearInitImagePreview();
    return;
  }
  if (file.type.startsWith("video/")) {
    showInitVideoPreview(file);
    return;
  }
  showInitImagePreview(file);
}

function setInitImageFile(file: File) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  initImageEl.files = transfer.files;
  showInitImagePreview(file);
}

function setInitVideoFile(file: File) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  initImageEl.files = transfer.files;
  showInitVideoPreview(file);
}

// Works whether the drop zone is empty or already holds an image — pasting
// always overwrites, same as dragging a new file in.
async function pasteInitImage() {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (!type) continue;
      const blob = await item.getType(type);
      const file = new File([blob], "pasted." + (type.split("/")[1] || "png"), { type });
      setInitImageFile(file);
      return;
    }
    setGenStatus("Clipboard has no image to paste.", "status-error");
  } catch (err) {
    setGenStatus("Paste failed: " + (err as Error).message, "status-error");
  }
}

function showStyleImagePreview(file: File) {
  const src = URL.createObjectURL(file);
  styleImagePreviewEl.src = src;
  styleImagePreviewEl.hidden = false;
  styleImageHintEl.hidden = true;
  styleImageDropEl.classList.add("has-image");
  styleImageClearBtnEl.hidden = false;
  styleImageDownloadBtnEl.hidden = false;
  styleImageCopyBtnEl.hidden = false;
}

function clearStyleImagePreview() {
  styleImagePreviewEl.hidden = true;
  styleImageHintEl.hidden = false;
  styleImageDropEl.classList.remove("has-image");
  styleImageClearBtnEl.hidden = true;
  styleImageDownloadBtnEl.hidden = true;
  styleImageCopyBtnEl.hidden = true;
}

function setStyleImageFile(file: File) {
  const transfer = new DataTransfer();
  transfer.items.add(file);
  styleImageEl.files = transfer.files;
  showStyleImagePreview(file);
}

async function onGalleryClick() {
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("open_outputs_folder");
  } catch (err) {
    setGenStatus(
      "Open gallery only works inside the BlobVision app window, not a plain browser tab.",
      "status-error",
    );
    console.error(err);
  }
}

function filenameFromUrl(url: string): string {
  try {
    return new URL(url).pathname.split("/").pop() || "blobvision.png";
  } catch {
    return "blobvision.png";
  }
}

async function fetchAsBlob(url: string): Promise<Blob> {
  const res = await fetch(url);
  if (!res.ok) throw new Error("Failed to fetch image (" + res.status + ")");
  return res.blob();
}

// Fetch-then-blob-URL rather than a plain <a href download>: the image URLs
// here are cross-origin (http://127.0.0.1:8420, a different origin from the
// app itself), and browsers/webviews silently ignore the `download`
// attribute on cross-origin URLs for security reasons — clicking such a
// link just navigates the whole window to the raw image instead of saving
// it (this actually happened: it replaced the entire app UI with the image
// on a black background, no way back except force-closing the app). A blob:
// URL created from a fetched Blob is always same-origin to the page that
// created it, so `download` works correctly on it regardless of where the
// bytes originally came from.
async function downloadImage(url: string) {
  try {
    const blob = await fetchAsBlob(url);
    const blobUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = filenameFromUrl(url);
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 2000);
  } catch (err) {
    setGenStatus("Download failed: " + (err as Error).message, "status-error");
  }
}

// Writes the image straight to the OS clipboard as image/png so it can be
// pasted directly into other apps (chat, social media, etc.) without saving
// a file first.
async function copyImageToClipboard(url: string) {
  try {
    const blob = await fetchAsBlob(url);
    await navigator.clipboard.write([new ClipboardItem({ [blob.type || "image/png"]: blob })]);
  } catch (err) {
    setGenStatus("Copy failed: " + (err as Error).message, "status-error");
  }
}

// Feeds the current output straight back into the shared drop zone as a new
// img2img/video source — chains a result into the next generation without a
// manual download-then-drag-back round trip. Works for any family (they all
// share #init-image-drop) and for either output kind: an image output
// becomes a new init image, a video output becomes a new source video
// (re-running a family's own video pipeline on its own prior output is a
// legitimate use, not specially blocked).
async function useOutputAsInit() {
  try {
    if (hasOutputVideo()) {
      const blob = await fetchAsBlob(outputVideoEl.src);
      const file = new File([blob], filenameFromUrl(outputVideoEl.src), { type: blob.type || "video/mp4" });
      setInitVideoFile(file);
    } else if (hasOutputImage()) {
      const blob = await fetchAsBlob(outputImageEl.src);
      const file = new File([blob], filenameFromUrl(outputImageEl.src), { type: blob.type || "image/png" });
      setInitImageFile(file);
    } else {
      setGenStatus("Nothing to use — no output yet.", "status-error");
    }
  } catch (err) {
    setGenStatus("Use as init failed: " + (err as Error).message, "status-error");
  }
}

function openLightbox(src: string) {
  lightboxImageEl.src = src;
  lightboxEl.hidden = false;
}

function closeLightbox() {
  lightboxEl.hidden = true;
}

// --- warlock's grimoire ---
// A built-in catalogue of cryptic SD1.x-era prompt fragments (grimoire-data.ts),
// browsable from two places that share the same merged data: a compact ▾
// popup anchored to the Prompt/Negative Prompt fields (pick-only, stays open
// across multiple picks), and a full-screen overlay (GRIMOIRE button) where
// the user can add/remove their own entries per category. Custom entries
// persist in localStorage — this is pure client UI state, no backend
// involvement needed.

type GrimoireKind = "prompt" | "negative";

interface StoredGrimoire {
  prompt: Record<string, string[]>;
  negative: Record<string, string[]>;
}

const GRIMOIRE_STORAGE_KEY = "blobvision.grimoire.custom.v1";

function loadCustomGrimoire(): StoredGrimoire {
  try {
    const raw = localStorage.getItem(GRIMOIRE_STORAGE_KEY);
    if (!raw) return { prompt: {}, negative: {} };
    const parsed = JSON.parse(raw);
    return { prompt: parsed.prompt ?? {}, negative: parsed.negative ?? {} };
  } catch {
    return { prompt: {}, negative: {} };
  }
}

function saveCustomGrimoire(data: StoredGrimoire) {
  localStorage.setItem(GRIMOIRE_STORAGE_KEY, JSON.stringify(data));
}

let customGrimoire = loadCustomGrimoire();

function baseGrimoireCategories(kind: GrimoireKind): GrimoireCategory[] {
  return kind === "prompt" ? GRIMOIRE_PROMPT_CATEGORIES : GRIMOIRE_NEGATIVE_CATEGORIES;
}

interface MergedGrimoireEntry {
  text: string;
  custom: boolean;
}

interface MergedGrimoireCategory {
  name: string;
  entries: MergedGrimoireEntry[];
}

function mergedGrimoireCategories(kind: GrimoireKind): MergedGrimoireCategory[] {
  const customByCategory = customGrimoire[kind];
  return baseGrimoireCategories(kind).map((cat) => {
    const custom = customByCategory[cat.name] ?? [];
    return {
      name: cat.name,
      entries: [
        ...cat.entries.map((text) => ({ text, custom: false })),
        ...custom.map((text) => ({ text, custom: true })),
      ],
    };
  });
}

function addCustomGrimoireEntry(kind: GrimoireKind, category: string, text: string) {
  const trimmed = text.trim();
  if (!trimmed) return;
  const byCategory = customGrimoire[kind];
  const list = byCategory[category] ?? (byCategory[category] = []);
  if (list.includes(trimmed)) return;
  list.push(trimmed);
  saveCustomGrimoire(customGrimoire);
}

function removeCustomGrimoireEntry(kind: GrimoireKind, category: string, text: string) {
  const list = customGrimoire[kind][category];
  if (!list) return;
  customGrimoire[kind][category] = list.filter((t) => t !== text);
  saveCustomGrimoire(customGrimoire);
}

// Appends `text` to a prompt-like field, comma-separated.
function appendToField(el: HTMLTextAreaElement | HTMLInputElement, text: string) {
  const current = el.value.trim();
  el.value = current ? current + ", " + text : text;
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

// --- shared anchored popup menu (FORMAT button + both Grimoire ▾ pickers) ---

let openMenuEl: HTMLDivElement | null = null;
let openMenuCleanup: (() => void) | null = null;

function closeAnchoredMenu() {
  if (openMenuEl) {
    openMenuEl.remove();
    openMenuEl = null;
  }
  if (openMenuCleanup) {
    openMenuCleanup();
    openMenuCleanup = null;
  }
}

// Positions an absolutely-placed popup under `anchor`; closes on outside
// click, Escape, or re-clicking the same anchor. `build` populates the menu
// contents fresh each time it opens.
function openAnchoredMenu(anchor: HTMLElement, build: (menu: HTMLDivElement) => void) {
  const reopening = openMenuEl?.dataset.anchorFor === anchor.id;
  closeAnchoredMenu();
  if (reopening) return;

  const menu = document.createElement("div");
  menu.className = "popup-menu";
  menu.dataset.anchorFor = anchor.id;
  build(menu);
  document.body.appendChild(menu);

  const anchorRect = anchor.getBoundingClientRect();
  menu.style.position = "fixed";
  menu.style.top = anchorRect.bottom + 4 + "px";
  menu.style.left = anchorRect.left + "px";
  const menuRect = menu.getBoundingClientRect();
  if (menuRect.right > window.innerWidth - 8) {
    menu.style.left = Math.max(8, window.innerWidth - menuRect.width - 8) + "px";
  }
  openMenuEl = menu;

  const onOutsideClick = (e: MouseEvent) => {
    if (menu.contains(e.target as Node) || anchor.contains(e.target as Node)) return;
    closeAnchoredMenu();
  };
  const onEscape = (e: KeyboardEvent) => {
    if (e.key === "Escape") closeAnchoredMenu();
  };
  // Deferred so the click that opened the menu doesn't immediately close it.
  setTimeout(() => document.addEventListener("click", onOutsideClick), 0);
  document.addEventListener("keydown", onEscape);
  openMenuCleanup = () => {
    document.removeEventListener("click", onOutsideClick);
    document.removeEventListener("keydown", onEscape);
  };
}

function renderGrimoirePicker(kind: GrimoireKind, targetEl: HTMLTextAreaElement | HTMLInputElement, menu: HTMLDivElement) {
  mergedGrimoireCategories(kind).forEach((cat) => {
    const heading = document.createElement("div");
    heading.className = "popup-menu-heading";
    heading.textContent = cat.name;
    menu.appendChild(heading);
    cat.entries.forEach((entry) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "popup-menu-item";
      item.textContent = entry.text;
      item.addEventListener("click", () => appendToField(targetEl, entry.text));
      menu.appendChild(item);
    });
  });
}

const FORMAT_OPTIONS: { value: string; label: string }[] = [
  { value: "1:1", label: "1:1" },
  { value: "16:9", label: "16:9" },
  { value: "9:16", label: "9:16" },
  { value: "custom", label: "custom" },
];

function renderFormatMenu(menu: HTMLDivElement) {
  FORMAT_OPTIONS.forEach((opt) => {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "popup-menu-item";
    if (opt.value === currentAspect()) item.classList.add("active");
    item.disabled = opt.value === "custom" && formatCustomDisabled;
    item.textContent = opt.label;
    item.addEventListener("click", () => {
      setAspect(opt.value);
      closeAnchoredMenu();
    });
    menu.appendChild(item);
  });
}

// --- full grimoire overlay ---

let grimoireActiveKind: GrimoireKind = "prompt";

function openGrimoireOverlay() {
  renderGrimoireOverlay();
  grimoireOverlayEl.hidden = false;
}

function closeGrimoireOverlay() {
  grimoireOverlayEl.hidden = true;
}

function renderGrimoireOverlay() {
  grimoireTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.kind === grimoireActiveKind));
  grimoireListEl.innerHTML = "";
  mergedGrimoireCategories(grimoireActiveKind).forEach((cat) => {
    const section = document.createElement("div");
    section.className = "grimoire-category";

    const heading = document.createElement("h3");
    heading.textContent = cat.name;
    section.appendChild(heading);

    const chipsEl = document.createElement("div");
    chipsEl.className = "grimoire-chips";
    cat.entries.forEach((entry) => {
      const chip = document.createElement("span");
      chip.className = "grimoire-chip" + (entry.custom ? " grimoire-chip-custom" : "");
      const label = document.createElement("span");
      label.textContent = entry.text;
      chip.appendChild(label);
      if (entry.custom) {
        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "grimoire-chip-remove";
        removeBtn.title = "Remove";
        removeBtn.textContent = "×";
        removeBtn.addEventListener("click", () => {
          removeCustomGrimoireEntry(grimoireActiveKind, cat.name, entry.text);
          renderGrimoireOverlay();
        });
        chip.appendChild(removeBtn);
      }
      chipsEl.appendChild(chip);
    });
    section.appendChild(chipsEl);

    const addRow = document.createElement("div");
    addRow.className = "grimoire-add-row";
    const addInput = document.createElement("input");
    addInput.type = "text";
    addInput.placeholder = "Add your own...";
    const addBtn = document.createElement("button");
    addBtn.type = "button";
    addBtn.textContent = "ADD";
    const submit = () => {
      if (!addInput.value.trim()) return;
      addCustomGrimoireEntry(grimoireActiveKind, cat.name, addInput.value);
      addInput.value = "";
      renderGrimoireOverlay();
    };
    addBtn.addEventListener("click", submit);
    addInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        submit();
      }
    });
    addRow.appendChild(addInput);
    addRow.appendChild(addBtn);
    section.appendChild(addRow);

    grimoireListEl.appendChild(section);
  });
}

// --- meme studio ---

function setStudioStatus(text: string, cls: "status-pending" | "status-ok" | "status-error") {
  studioStatusEl.hidden = false;
  studioStatusEl.textContent = text;
  studioStatusEl.className = "status " + cls;
}

function studioHasImage(): boolean {
  return !!studioImageEl.src;
}

function currentStudioAspect(): StudioAspect {
  return (Array.from(studioAspectRadios).find((r) => r.checked)?.value as StudioAspect) ?? "custom";
}

function enterStudioMode() {
  studioMode = true;
  layoutEl.hidden = true;
  studioEl.hidden = false;
  studioBtnEl.classList.add("active");
  familyBtns.forEach((btn) => btn.classList.remove("active"));
  resizeStudioViewport();
  loadStudioThumbnails(); // always refresh — stay in sync with anything added/removed on disk since last visit
}

function exitStudioMode() {
  if (!studioMode) return;
  studioMode = false;
  studioEl.hidden = true;
  layoutEl.hidden = false;
  studioBtnEl.classList.remove("active");
  disarmStudioSticker();
}

function makeStudioThumb(outputUrl: string, filename: string): HTMLImageElement {
  const thumb = document.createElement("img");
  thumb.className = "studio-thumb";
  // Must match #studio-image's crossorigin="anonymous" (index.html) for the
  // SAME URL — browsers cache CORS and non-CORS responses to an identical
  // URL separately, and can serve back the wrong (opaque, canvas-tainting)
  // one once both request modes have been used for it. Loading every URL in
  // CORS mode everywhere it appears keeps that cache from ever splitting.
  thumb.crossOrigin = "anonymous";
  thumb.alt = filename;
  // Deleted-from-disk (or otherwise unreachable) outputs show an empty
  // dashed slot instead of the browser's default broken-image icon.
  thumb.addEventListener("error", () => {
    thumb.classList.add("studio-thumb-empty");
    thumb.removeAttribute("src");
  });
  thumb.src = API_BASE + outputUrl;
  thumb.addEventListener("click", () => {
    if (!thumb.classList.contains("studio-thumb-empty")) loadStudioImage(outputUrl);
  });
  return thumb;
}

function prependStudioThumbnail(outputUrl: string, filename: string) {
  studioStripEl.insertBefore(makeStudioThumb(outputUrl, filename), studioStripEl.firstChild);
}

// Populates the thumbnail strip from every family's output folder, newest
// first. If nothing is loaded into the main preview yet, defaults to the
// single most recent output across all of them — "the last image produced
// in any mode."
async function loadStudioThumbnails() {
  try {
    const items: { filename: string; output_url: string }[] = await fetchJson("/outputs/recent?limit=60");
    studioStripEl.innerHTML = "";
    for (const item of items) studioStripEl.appendChild(makeStudioThumb(item.output_url, item.filename));
    if (!studioHasImage() && items.length) loadStudioImage(items[0].output_url);
  } catch (err) {
    console.error("Failed to load recent outputs:", err);
  }
}

// Resets crop/zoom/pan and text position to a clean slate — carrying a crop
// meant for a previous, differently-sized image over to a new one would be
// meaningless at best.
// Crop/zoom/rotate/pan/text-offset only — used whenever a new base image
// loads, since a fresh image has no reason to inherit the previous one's
// framing. Text STYLE (rotate/scale/contour/colors) deliberately survives a
// new image load; only resetStudioFullState() (after CREATE MEME) touches
// that too.
function resetStudioCrop() {
  studioZoom = 1;
  studioPanX = 0;
  studioPanY = 0;
  studioImageRotate = 0;
  studioZoomEl.value = "1";
  studioZoomValueEl.textContent = "1.00";
  studioImageRotateEl.value = "0";
  studioImageRotateValueEl.textContent = "0";
  studioTopTextOffset.x = 0;
  studioTopTextOffset.y = 0;
  studioBottomTextOffset.x = 0;
  studioBottomTextOffset.y = 0;
  studioUpscaleApplyCount = 0;
}

function resetStudioTextStyle(
  style: StudioTextStyle,
  rotateScrub: ScrubberController,
  scaleScrub: ScrubberController,
  contourScrub: ScrubberController,
  fillInput: HTMLInputElement,
  strokeInput: HTMLInputElement,
  swatchEl: HTMLDivElement,
) {
  style.rotate = 0;
  style.scale = 1;
  style.contour = 0.08;
  style.fillColor = "#ffffff";
  style.strokeColor = "#000000";
  rotateScrub.reset(0);
  scaleScrub.reset(1);
  contourScrub.reset(0.08);
  fillInput.value = style.fillColor;
  strokeInput.value = style.strokeColor;
  swatchEl.style.setProperty("--fill-color", style.fillColor);
  swatchEl.style.setProperty("--stroke-color", style.strokeColor);
}

// Everything back to defaults, including text styling/colors and the
// aspect (back to "custom") — used after CREATE MEME, since the just-baked
// image is a brand new starting point, not something to keep cropping with
// leftover settings from the caption that's now permanently part of it.
function resetStudioFullState() {
  resetStudioCrop();
  resetStudioTextStyle(
    studioTopStyle, studioTopRotateScrub, studioTopScaleScrub, studioTopContourScrub,
    studioTopFillColorEl, studioTopStrokeColorEl, studioTopColorSwatchEl,
  );
  resetStudioTextStyle(
    studioBottomStyle, studioBottomRotateScrub, studioBottomScaleScrub, studioBottomContourScrub,
    studioBottomFillColorEl, studioBottomStrokeColorEl, studioBottomColorSwatchEl,
  );
  const customRadio = Array.from(studioAspectRadios).find((r) => r.value === "custom");
  if (customRadio) customRadio.checked = true;
  studioAspect = "custom";
}

function loadStudioImage(outputUrl: string) {
  resetStudioCrop();
  studioImageEl.src = API_BASE + outputUrl + "?t=" + Date.now();
}

function clearStudioImage() {
  studioImageEl.src = "";
  studioNaturalW = 0;
  studioNaturalH = 0;
  clearStudioScratchLayers(); // also resets undo history; calls updateStudioImageBg(), clearing bg + hiding rotate handles
  disarmStudioSticker();
  resetStudioCrop();
  studioTopTextEl.value = "";
  studioBottomTextEl.value = "";
  updateStudioTextOverlays();
}

async function pasteStudioImage() {
  try {
    const items = await navigator.clipboard.read();
    for (const item of items) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (!type) continue;
      const blob = await item.getType(type);
      resetStudioCrop();
      studioImageEl.src = URL.createObjectURL(blob);
      return;
    }
    setStudioStatus("Clipboard has no image to paste.", "status-error");
  } catch (err) {
    setStudioStatus("Paste failed: " + (err as Error).message, "status-error");
  }
}

// Fits a box of the target aspect ratio inside the available canvas space,
// centered — the same math object-fit:contain does, just applied to a plain
// div via explicit pixel sizing (CSS aspect-ratio doesn't reliably size a
// flex child constrained on both axes the way this needs).
function resizeStudioViewport() {
  if (studioEl.hidden) return;
  const canvasRect = studioCanvasEl.getBoundingClientRect();
  const availW = canvasRect.width - 16;
  const availH = canvasRect.height - 16;
  if (availW <= 0 || availH <= 0) return;
  const [rw, rh] =
    studioAspect === "custom" ? [studioNaturalW || 1, studioNaturalH || 1] : STUDIO_ASPECT_RATIOS[studioAspect];
  let w = availW;
  let h = (w * rh) / rw;
  if (h > availH) {
    h = availH;
    w = (h * rw) / rh;
  }
  studioViewportEl.style.width = w + "px";
  studioViewportEl.style.height = h + "px";
  updateStudioImageBg();
  updateStudioTextOverlays();
}

// The un-zoomed cover-fit scale (box vs. natural image, no studioZoom
// factor) — used ONLY to convert a natural-pixel PAN offset into a
// screen-pixel position, kept deliberately separate from studioEffectiveScale
// (which DOES include zoom, for SIZE). Mixing the two into one "scale" used
// for both position and size was the bug behind "zooming a panned-off-center
// layer makes it fly off toward one edge instead of growing in place": the
// pan offset got re-multiplied by the (changing) zoom factor every time the
// zoom slider moved, so the layer's on-screen center visibly slid further
// from the viewport center as zoom increased. Position now depends only on
// pan and the base cover-fit (viewport/image aspect) — never on zoom — so
// the layer's own on-screen center is fixed by pan alone, and zoom purely
// resizes the box around that fixed point.
function studioCoverBase(boxW: number, boxH: number, natW: number, natH: number): number {
  return natW && natH ? Math.max(boxW / natW, boxH / natH) : 1;
}

// The single source of truth for how the image is SIZED, shared by the live
// preview (updateStudioImageBg), the pan-drag handler, and the export
// (drawStudioForeground) — computing it different ways in each place was
// the actual bug behind two earlier reported symptoms (rotating making the
// image look zoomed in, and the pan range feeling too short): apparent zoom
// depends ONLY on the zoom slider and the box's own cover-fit, full stop,
// with no rotation term mixed in.
function studioEffectiveScale(boxW: number, boxH: number, natW: number, natH: number): number {
  return studioCoverBase(boxW, boxH, natW, natH) * studioZoom;
}

// Screen position (viewport-relative) of the layer's own center, updated by
// updateStudioImageBg() on every render — read by the rotate-handle drag
// handler to turn mouse position into an angle around the layer's actual
// current center, wherever pan has put it.
let studioLayerCenterX = 0;
let studioLayerCenterY = 0;

// Renders the image into the viewport as a box sized to EXACTLY the scaled
// image (natW*scale x natH*scale, background-size 100% 100% — no cover/
// contain math needed once the box already matches the content 1:1),
// centered at (viewport center − pan*coverBase) and rotated around its OWN
// 50%/50%. Rotating a rectangle around its own center can never open a gap
// *inside* the rectangle regardless of angle — the only gap that can appear
// is between the (possibly small/rotated) box and the viewport's edges,
// which #studio-image-backdrop is there to fill. This replaces an earlier
// design that kept #studio-image-bg oversized (via a "rotated cover box"
// safety margin) and panned via background-position within it: that
// approach implicitly assumed rotation pivoted around the OVERSIZED box's
// own center, which stopped being true once rotation was changed to pivot
// around the panned image's center instead — the safety margin was no
// longer guaranteed sufficient, and showed up as the backdrop appearing to
// "bite into" a corner of the foreground at some pan/zoom/rotation
// combinations. Sizing the box to the image itself sidesteps the whole
// problem instead of re-deriving a bigger margin.
//
// Because the box's own center (screen position: viewport center minus
// pan*coverBase) doesn't depend on rotation OR zoom, panning translates the
// image linearly on screen regardless of the current rotation angle — the
// mousemove handler below no longer needs to un-rotate the mouse delta
// before applying it (an earlier version did, back when the box's rotation
// pivot was fixed at the viewport center and rotating it visibly coupled
// panX/panY into both screen axes; that coupling is gone now that pivot ==
// content center) — and zoom no longer drags that center around either (see
// studioCoverBase() above). Kept in sync with drawStudioForeground() below,
// which draws the equivalent composition onto the export/patchwork canvases
// using the same pivot formula.
function updateStudioImageBg() {
  if (!studioImageEl.src) {
    studioImageBgEl.style.backgroundImage = "";
    studioImageBackdropEl.style.backgroundImage = "";
    studioImageBgEl.classList.remove("has-image");
    return;
  }
  const url = 'url("' + studioImageEl.src + '")';
  studioImageBgEl.style.backgroundImage = url;
  studioImageBgEl.classList.add("has-image");
  // The blurred backdrop always mirrors the ACTIVE image now — the SNAP
  // patchwork is a separate layer (#studio-patchwork-layer) stacked on top
  // of it, so stamping something no longer has to steal the blur away from
  // whatever the patchwork doesn't cover.
  studioImageBackdropEl.style.backgroundImage = url;
  if (!studioNaturalW || !studioNaturalH) return;
  const viewportRect = studioViewportEl.getBoundingClientRect();
  if (!viewportRect.width || !viewportRect.height) return;
  const scale = studioEffectiveScale(viewportRect.width, viewportRect.height, studioNaturalW, studioNaturalH);
  const posBase = studioCoverBase(viewportRect.width, viewportRect.height, studioNaturalW, studioNaturalH);
  const bgW = studioNaturalW * scale;
  const bgH = studioNaturalH * scale;
  const centerX = viewportRect.width / 2 - studioPanX * posBase;
  const centerY = viewportRect.height / 2 - studioPanY * posBase;
  studioLayerCenterX = centerX;
  studioLayerCenterY = centerY;
  studioImageBgEl.style.width = bgW + "px";
  studioImageBgEl.style.height = bgH + "px";
  studioImageBgEl.style.left = centerX - bgW / 2 + "px";
  studioImageBgEl.style.top = centerY - bgH / 2 + "px";
  studioImageBgEl.style.backgroundSize = "100% 100%";
  studioImageBgEl.style.transform = "rotate(" + studioImageRotate + "deg)";
}

function setStudioZoom(z: number) {
  studioZoom = clamp(z, 0.1, 4);
  studioZoomEl.value = String(studioZoom);
  studioZoomValueEl.textContent = studioZoom.toFixed(2);
  updateStudioImageBg();
}

function setStudioImageRotate(deg: number) {
  studioImageRotate = clamp(deg, -180, 180);
  studioImageRotateEl.value = String(studioImageRotate);
  studioImageRotateValueEl.textContent = String(Math.round(studioImageRotate));
  updateStudioImageBg();
}

function currentStudioUpscaleScale(): number {
  if (studioUpscale4xEl.checked) return 4;
  if (studioUpscale2xEl.checked) return 2;
  return 0;
}

let studioMeasureCanvas: HTMLCanvasElement | null = null;
const STUDIO_LINE_HEIGHT_MULT = 1.15;

function studioMeasureCtx(): CanvasRenderingContext2D | null {
  if (!studioMeasureCanvas) studioMeasureCanvas = document.createElement("canvas");
  return studioMeasureCanvas.getContext("2d");
}

// Greedy word-wrap at a given font size (ctx.font must already be set).
function wrapTextLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const lines: string[] = [];
  let current = words[0];
  for (let i = 1; i < words.length; i++) {
    const test = current + " " + words[i];
    if (ctx.measureText(test).width <= maxWidth) current = test;
    else {
      lines.push(current);
      current = words[i];
    }
  }
  lines.push(current);
  return lines;
}

// Picks the largest font size (down to minSize) whose word-wrapped layout
// fits within maxWidth x maxHeight, and returns the actual wrapped lines at
// that size. Used for BOTH the CSS preview (which lets the browser re-wrap
// the same text at this font-size — very likely the same breaks, since it's
// the same text/width/font) and the canvas export (which draws these exact
// lines) — previously the preview wrapped freely at a fixed size while the
// export only ever shrank to fit one line, so text that wrapped to 2 lines
// on screen would silently shrink to 1 line (or overflow) once baked.
function fitWrappedText(
  text: string,
  maxWidth: number,
  maxHeight: number,
  maxFontSize: number,
  minSize = 10,
): { size: number; lines: string[] } {
  const ctx = studioMeasureCtx();
  if (!ctx) return { size: maxFontSize, lines: [text] };
  let size = maxFontSize;
  let lines = [text];
  while (size > minSize) {
    ctx.font = '700 ' + size + 'px Impact, "Arial Narrow Bold", sans-serif';
    lines = wrapTextLines(ctx, text, maxWidth);
    if (lines.length * size * STUDIO_LINE_HEIGHT_MULT <= maxHeight) break;
    size -= 2;
  }
  return { size, lines };
}

function applyTextTransform(el: HTMLDivElement, offset: { x: number; y: number }, style: StudioTextStyle) {
  const rect = studioViewportEl.getBoundingClientRect();
  el.style.transform =
    "translate(" + offset.x * rect.width + "px, " + offset.y * rect.height + "px)" +
    " rotate(" + style.rotate + "deg) scale(" + style.scale + ")";
}

// Live preview: sizes the CSS overlay text to fit the viewport (the actual
// visible crop box, not the possibly letterboxed/undersized <img>) using
// the same 90%-width / 15%-height fractions drawMemeText() uses for the
// baked canvas version, so preview and final output match. Also applies
// per-text color/contour/rotate/scale/drag-offset and toggles pointer-events
// so empty overlays don't block panning underneath.
function updateStudioTextOverlays() {
  const topText = studioTopTextEl.value.trim().toUpperCase();
  const bottomText = studioBottomTextEl.value.trim().toUpperCase();
  studioTextTopEl.textContent = topText;
  studioTextBottomEl.textContent = bottomText;
  studioTextTopEl.classList.toggle("has-text", !!topText);
  studioTextBottomEl.classList.toggle("has-text", !!bottomText);
  const rect = studioViewportEl.getBoundingClientRect();
  if (!rect.width || !rect.height) return;
  const maxWidth = rect.width * 0.9;
  const maxHeight = rect.height * 0.3;
  const maxFontSize = rect.height * 0.15;
  const topSize = fitWrappedText(topText || "X", maxWidth, maxHeight, maxFontSize).size * studioTopStyle.scale;
  const bottomSize = fitWrappedText(bottomText || "X", maxWidth, maxHeight, maxFontSize).size * studioBottomStyle.scale;
  studioTextTopEl.style.fontSize = topSize + "px";
  studioTextBottomEl.style.fontSize = bottomSize + "px";
  studioTextTopEl.style.color = studioTopStyle.fillColor;
  studioTextTopEl.style.webkitTextStroke = Math.max(0, studioTopStyle.contour * topSize) + "px " + studioTopStyle.strokeColor;
  studioTextBottomEl.style.color = studioBottomStyle.fillColor;
  studioTextBottomEl.style.webkitTextStroke =
    Math.max(0, studioBottomStyle.contour * bottomSize) + "px " + studioBottomStyle.strokeColor;
  applyTextTransform(studioTextTopEl, studioTopTextOffset, studioTopStyle);
  applyTextTransform(studioTextBottomEl, studioBottomTextOffset, studioBottomStyle);
}

// Free click-and-drag positioning directly on the canvas — no separate x/y
// controls, just grab the text and move it (e.g. to tuck it inside a speech
// bubble sticker). Unclamped, same free-positioning philosophy as panning
// the image itself.
function wireStudioTextDrag(el: HTMLDivElement, offset: { x: number; y: number }, style: StudioTextStyle) {
  let dragging = false;
  let startClientX = 0;
  let startClientY = 0;
  let startOffX = 0;
  let startOffY = 0;
  el.addEventListener("mousedown", (e) => {
    if (!el.classList.contains("has-text")) return;
    dragging = true;
    startClientX = e.clientX;
    startClientY = e.clientY;
    startOffX = offset.x;
    startOffY = offset.y;
    e.stopPropagation();
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = studioViewportEl.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    offset.x = startOffX + (e.clientX - startClientX) / rect.width;
    offset.y = startOffY + (e.clientY - startClientY) / rect.height;
    applyTextTransform(el, offset, style);
  });
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
}

// Drag-to-scrub + click-to-type number control (rotate/scale/contour): drag
// horizontally to change the value, or click without moving to type an
// exact one. Shared by all 6 instances (rotate/scale/contour x top/bottom).
function wireScrubber(
  el: HTMLDivElement,
  opts: { min: number; max: number; step: number; initial: number; decimals: number; onChange: (v: number) => void },
): ScrubberController {
  const valueEl = el.querySelector<HTMLSpanElement>(".studio-scrub-value")!;
  let value = opts.initial;
  const fmt = (v: number) => v.toFixed(opts.decimals);
  valueEl.textContent = fmt(value);

  let dragging = false;
  let moved = false;
  let startX = 0;
  let startValue = 0;

  el.addEventListener("mousedown", (e) => {
    dragging = true;
    moved = false;
    startX = e.clientX;
    startValue = value;
    e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const dx = e.clientX - startX;
    if (Math.abs(dx) > 3) moved = true;
    if (!moved) return;
    const sensitivity = (opts.max - opts.min) / 200; // full range over a 200px drag
    value = clamp(startValue + dx * sensitivity, opts.min, opts.max);
    value = Math.round(value / opts.step) * opts.step;
    valueEl.textContent = fmt(value);
    opts.onChange(value);
  });
  window.addEventListener("mouseup", () => {
    if (dragging && !moved) startScrubberEdit();
    dragging = false;
  });

  function startScrubberEdit() {
    const input = document.createElement("input");
    input.type = "number";
    input.className = "studio-scrub-input";
    input.value = String(value);
    input.step = String(opts.step);
    valueEl.replaceWith(input);
    input.focus();
    input.select();
    const commit = () => {
      const parsed = Number(input.value);
      if (!Number.isNaN(parsed)) {
        value = clamp(parsed, opts.min, opts.max);
        opts.onChange(value);
      }
      valueEl.textContent = fmt(value);
      input.replaceWith(valueEl);
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") input.blur();
      if (e.key === "Escape") {
        input.value = String(value);
        input.blur();
      }
    });
  }

  return {
    reset(v: number) {
      value = clamp(v, opts.min, opts.max);
      valueEl.textContent = fmt(value);
    },
  };
}

// Two real, separately-clickable squares (Photoshop-style overlapping
// foreground/background swatches) instead of one div with click-position
// math — the earlier single-div version made both regions open the same
// (stroke) color picker in practice, since the two 1x1px native color
// inputs sat at identical unstyled positions and only one was ever
// reachable. -bg = contour/stroke, -fg = text/fill.
function wireColorSwatch(
  swatchEl: HTMLDivElement,
  fillInput: HTMLInputElement,
  strokeInput: HTMLInputElement,
  style: StudioTextStyle,
) {
  const bgEl = swatchEl.querySelector<HTMLDivElement>(".studio-color-swatch-bg")!;
  const fgEl = swatchEl.querySelector<HTMLDivElement>(".studio-color-swatch-fg")!;
  swatchEl.style.setProperty("--fill-color", style.fillColor);
  swatchEl.style.setProperty("--stroke-color", style.strokeColor);
  bgEl.addEventListener("click", (e) => {
    e.stopPropagation();
    strokeInput.click();
  });
  fgEl.addEventListener("click", (e) => {
    e.stopPropagation();
    fillInput.click();
  });
  fillInput.addEventListener("input", () => {
    style.fillColor = fillInput.value;
    swatchEl.style.setProperty("--fill-color", style.fillColor);
    updateStudioTextOverlays();
  });
  strokeInput.addEventListener("input", () => {
    style.strokeColor = strokeInput.value;
    swatchEl.style.setProperty("--stroke-color", style.strokeColor);
    updateStudioTextOverlays();
  });
}

// Mirrors updateStudioImageBg() onto a canvas: draws the WHOLE natural image
// at the same scale, pivoted (translate + rotate) around the same
// pan-adjusted center used for the box's own position there, then lets the
// canvas's own bounds clip whatever falls outside w x h — no explicit
// source-rect cropping needed, so there's nothing to clamp or go wrong when
// pan/zoom pushes the image partway (or fully) off the frame.
function drawStudioForeground(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  w: number,
  h: number,
  natW: number = studioNaturalW,
  natH: number = studioNaturalH,
  rotateDeg: number = studioImageRotate,
) {
  const scale = studioEffectiveScale(w, h, natW, natH);
  const posBase = studioCoverBase(w, h, natW, natH);
  const pivotX = w / 2 - studioPanX * posBase;
  const pivotY = h / 2 - studioPanY * posBase;
  const fullW = natW * scale;
  const fullH = natH * scale;
  ctx.save();
  ctx.translate(pivotX, pivotY);
  ctx.rotate((rotateDeg * Math.PI) / 180);
  ctx.drawImage(img, -fullW / 2, -fullH / 2, fullW, fullH);
  ctx.restore();
}

// Caps exported memes to a normal social-sharing size instead of baking at
// whatever the source's native resolution happens to be (an x4-upscaled
// output would otherwise export as a multi-thousand-pixel file) — but never
// upscales a small "custom" (uncropped) source past its own native size.
function studioExportSize(): { w: number; h: number } {
  if (studioAspect === "custom") {
    const nw = studioNaturalW || 1;
    const nh = studioNaturalH || 1;
    const scale = Math.min(1, STUDIO_EXPORT_MAX_DIM / Math.max(nw, nh));
    return { w: Math.round(nw * scale), h: Math.round(nh * scale) };
  }
  const [rw, rh] = STUDIO_ASPECT_RATIOS[studioAspect];
  if (rw >= rh) return { w: STUDIO_EXPORT_MAX_DIM, h: Math.round((STUDIO_EXPORT_MAX_DIM * rh) / rw) };
  return { w: Math.round((STUDIO_EXPORT_MAX_DIM * rw) / rh), h: STUDIO_EXPORT_MAX_DIM };
}

// Draws possibly-multiple wrapped lines, stacked away from the anchor edge
// (top text grows downward from its top anchor, bottom text grows upward
// from its bottom anchor) — matching how the CSS overlay's auto-height box
// naturally grows in updateStudioTextOverlays(), so a caption that wrapped
// to 2 lines in the live preview still wraps to 2 lines here instead of
// silently collapsing onto one (over)long line.
function drawMemeText(
  ctx: CanvasRenderingContext2D,
  text: string,
  w: number,
  h: number,
  position: "top" | "bottom",
  offset: { x: number; y: number },
  style: StudioTextStyle,
) {
  if (!text) return;
  const { size: fitSize, lines } = fitWrappedText(text, w * 0.9, h * 0.3, h * 0.15);
  const size = fitSize * style.scale;
  const lineHeight = size * STUDIO_LINE_HEIGHT_MULT;
  ctx.save();
  const anchorX = w / 2 + offset.x * w;
  const anchorY = (position === "top" ? h * 0.03 : h * 0.97) + offset.y * h;
  ctx.translate(anchorX, anchorY);
  ctx.rotate((style.rotate * Math.PI) / 180);
  ctx.font = '700 ' + size + 'px Impact, "Arial Narrow Bold", sans-serif';
  ctx.textAlign = "center";
  ctx.lineJoin = "round";
  const lineWidth = style.contour * size;
  ctx.fillStyle = style.fillColor;
  const n = lines.length;
  for (let i = 0; i < n; i++) {
    let y: number;
    if (position === "top") {
      ctx.textBaseline = "top";
      y = i * lineHeight;
    } else {
      ctx.textBaseline = "bottom";
      y = (i - (n - 1)) * lineHeight;
    }
    if (lineWidth > 0) {
      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = style.strokeColor;
      ctx.strokeText(lines[i], 0, y);
    }
    ctx.fillText(lines[i], 0, y);
  }
  ctx.restore();
}

function flashStudioCanvas() {
  studioFlashEl.classList.remove("flash");
  void studioFlashEl.offsetWidth; // restart the CSS animation
  studioFlashEl.classList.add("flash");
}

// Mirrors #studio-image-backdrop's CSS (plain cover-fit, no zoom/pan/
// rotation, scaled up 15% and blurred) so a free-dragged or rotated
// foreground that doesn't fully cover the export canvas reveals the same
// blurred echo in the baked file instead of a hard transparent/black edge.
function drawStudioBackdrop(ctx: CanvasRenderingContext2D, img: HTMLImageElement, w: number, h: number) {
  const coverScale = Math.max(w / img.naturalWidth, h / img.naturalHeight) * 1.15;
  const bw = img.naturalWidth * coverScale;
  const bh = img.naturalHeight * coverScale;
  ctx.save();
  ctx.filter = "blur(" + Math.round(w * 0.035) + "px)";
  ctx.drawImage(img, (w - bw) / 2, (h - bh) / 2, bw, bh);
  ctx.restore();
}

// Sized to the current export dimensions so SNAP always burns in at full
// export quality — resized (preserving existing content, just rescaled)
// rather than reset if the aspect ratio changes between snaps.
function ensureStudioPatchworkCanvas(w: number, h: number): HTMLCanvasElement {
  if (!studioPatchworkCanvas) {
    studioPatchworkCanvas = document.createElement("canvas");
    studioPatchworkCanvas.width = w;
    studioPatchworkCanvas.height = h;
  } else if (studioPatchworkCanvas.width !== w || studioPatchworkCanvas.height !== h) {
    const resized = document.createElement("canvas");
    resized.width = w;
    resized.height = h;
    resized.getContext("2d")!.drawImage(studioPatchworkCanvas, 0, 0, w, h);
    studioPatchworkCanvas = resized;
  }
  return studioPatchworkCanvas;
}

// Sized to the current export dimensions, same resize-preserving behavior
// as ensureStudioPatchworkCanvas() above but for the sticker layer.
function ensureStudioStickerCanvas(w: number, h: number): HTMLCanvasElement {
  if (!studioStickerCanvas) {
    studioStickerCanvas = document.createElement("canvas");
    studioStickerCanvas.width = w;
    studioStickerCanvas.height = h;
  } else if (studioStickerCanvas.width !== w || studioStickerCanvas.height !== h) {
    const resized = document.createElement("canvas");
    resized.width = w;
    resized.height = h;
    resized.getContext("2d")!.drawImage(studioStickerCanvas, 0, 0, w, h);
    studioStickerCanvas = resized;
  }
  return studioStickerCanvas;
}

// Resets BOTH scratch canvases (patchwork + stickers) and their shared undo
// history together — they're conceptually one "scratch state" that starts
// fresh at the same two points: the X-clear button and right after CREATE
// MEME bakes them in.
function clearStudioScratchLayers() {
  studioPatchworkCanvas = null;
  studioPatchworkLayerEl.style.backgroundImage = "";
  studioStickerCanvas = null;
  studioStickerLayerEl.style.backgroundImage = "";
  studioUndoStack = [];
  updateStudioUndoButton();
  updateStudioImageBg();
}

// Re-renders a scratch canvas's current pixels onto its display layer —
// shared by SNAP, sticker placement, and Undo, all of which mutate a canvas
// directly and then need the visible layer to catch up.
async function refreshStudioScratchDisplay(canvas: HTMLCanvasElement | null, layerEl: HTMLDivElement) {
  if (!canvas) {
    layerEl.style.backgroundImage = "";
    return;
  }
  const blob: Blob = await new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Layer render failed."))), "image/png"),
  );
  layerEl.style.backgroundImage = 'url("' + URL.createObjectURL(blob) + '")';
}

// Snapshots BOTH scratch canvases' CURRENT pixels before a mutating op (SNAP
// or a sticker stamp) so Undo can restore them together as one step — a
// `null` field means that canvas didn't exist yet at that point. Capped at
// STUDIO_UNDO_LIMIT; no redo, and nothing else in the studio mutates these
// canvases.
async function pushStudioUndo() {
  const entry: StudioUndoEntry = {
    patchwork: studioPatchworkCanvas ? await createImageBitmap(studioPatchworkCanvas) : null,
    sticker: studioStickerCanvas ? await createImageBitmap(studioStickerCanvas) : null,
  };
  studioUndoStack.push(entry);
  if (studioUndoStack.length > STUDIO_UNDO_LIMIT) studioUndoStack.shift();
  updateStudioUndoButton();
}

async function restoreStudioScratchSnapshot(
  snapshot: ImageBitmap | null,
  ensure: (w: number, h: number) => HTMLCanvasElement,
  setCanvas: (c: HTMLCanvasElement | null) => void,
  layerEl: HTMLDivElement,
) {
  if (!snapshot) {
    setCanvas(null);
    layerEl.style.backgroundImage = "";
    return;
  }
  const canvas = ensure(snapshot.width, snapshot.height);
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(snapshot, 0, 0);
  await refreshStudioScratchDisplay(canvas, layerEl);
}

async function undoStudioLastAction() {
  if (!studioUndoStack.length) return;
  const entry = studioUndoStack.pop()!;
  await restoreStudioScratchSnapshot(
    entry.patchwork, ensureStudioPatchworkCanvas, (c) => { studioPatchworkCanvas = c; }, studioPatchworkLayerEl,
  );
  await restoreStudioScratchSnapshot(
    entry.sticker, ensureStudioStickerCanvas, (c) => { studioStickerCanvas = c; }, studioStickerLayerEl,
  );
  updateStudioUndoButton();
  setStudioStatus("Undid last snap/sticker.", "status-ok");
}

function updateStudioUndoButton() {
  studioUndoBtnEl.disabled = studioUndoStack.length === 0;
}

// Stamps the active image's CURRENT on-screen composition (pan/zoom/
// rotation, cropped to the current aspect) onto the persistent patchwork
// canvas, then shows that patchwork on #studio-patchwork-layer — a separate
// layer stacked ON TOP of the always-blurred #studio-image-backdrop, opaque
// only where a stamp was actually drawn. The active image itself is
// untouched, so it stays in front and fully manipulable — repeated snaps
// build up a layered composite behind it, with the blur still showing
// through anywhere the patchwork doesn't cover.
async function snapStudioImage() {
  if (!studioHasImage()) {
    setStudioStatus("No image loaded to snap.", "status-error");
    return;
  }
  await studioImageEl.decode();
  const { w: exportW, h: exportH } = studioExportSize();
  await pushStudioUndo();
  const canvas = ensureStudioPatchworkCanvas(exportW, exportH);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  drawStudioForeground(ctx, studioImageEl, exportW, exportH);
  await refreshStudioScratchDisplay(canvas, studioPatchworkLayerEl);
  setStudioStatus("Snapped to background.", "status-ok");
}

// Screen (viewport-relative fraction) -> scratch-canvas pixel coordinates.
// The viewport and the scratch canvases always share the same aspect (both
// track studioAspect/studioExportSize()), so this is a plain proportional
// mapping — no crop/pan/zoom math needed, unlike the foreground image.
function studioViewportToCanvasPoint(
  clientX: number, clientY: number, canvasW: number, canvasH: number,
): { x: number; y: number } {
  const rect = studioViewportEl.getBoundingClientRect();
  return { x: ((clientX - rect.left) / rect.width) * canvasW, y: ((clientY - rect.top) / rect.height) * canvasH };
}

// Stamps the currently-armed sticker onto the STICKER canvas (not the
// patchwork) at the given screen point, at the current size/rotation — the
// sticker layer renders in front of the active foreground (see
// #studio-sticker-layer's DOM position), so a placed sticker stays visible
// instead of being covered up by the photo. Same "bake immediately, no
// live/editable object" model as SNAP — composes with it naturally since
// Undo snapshots both canvases together.
async function placeStudioSticker(clientX: number, clientY: number) {
  if (!studioActiveSticker) return;
  const { w: exportW, h: exportH } = studioExportSize();
  await pushStudioUndo();
  const canvas = ensureStudioStickerCanvas(exportW, exportH);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { x, y } = studioViewportToCanvasPoint(clientX, clientY, exportW, exportH);
  const sizePx = (studioStickerSizePercent / 100) * Math.min(exportW, exportH);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate((studioStickerRotateDeg * Math.PI) / 180);
  if (studioStickerMirrored) ctx.scale(-1, 1);
  if (studioActiveSticker.kind === "emoji") {
    ctx.font = sizePx + 'px "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(studioActiveSticker.glyph, 0, 0);
  } else {
    const activeShapeId = studioActiveSticker.id;
    const shape = STUDIO_SHAPE_STICKERS.find((s) => s.id === activeShapeId)!;
    shape.draw(ctx, sizePx);
  }
  ctx.restore();
  await refreshStudioScratchDisplay(canvas, studioStickerLayerEl);
  setStudioStatus("Sticker placed.", "status-ok");
}

function setStudioStickerSize(v: number) {
  studioStickerSizePercent = clamp(v, 5, 60);
  studioStickerSizeEl.value = String(studioStickerSizePercent);
  studioStickerSizeValueEl.textContent = Math.round(studioStickerSizePercent) + "%";
}

function setStudioStickerRotate(deg: number) {
  studioStickerRotateDeg = clamp(deg, -180, 180);
  studioStickerRotateEl.value = String(studioStickerRotateDeg);
  studioStickerRotateValueEl.textContent = String(Math.round(studioStickerRotateDeg));
  updateStudioStickerPreview();
}

// A toggle, not a one-shot action — stays on for every subsequent placement
// (any sticker) until switched off again, same "sticky until you change it"
// treatment as size/rotate.
function toggleStudioStickerMirror() {
  studioStickerMirrored = !studioStickerMirrored;
  studioStickerMirrorBtnEl.classList.toggle("active", studioStickerMirrored);
  updateStudioStickerPreview();
}

function ensureStudioStickerPreviewEl(): HTMLDivElement {
  if (!studioStickerPreviewEl) {
    studioStickerPreviewEl = document.createElement("div");
    studioStickerPreviewEl.className = "studio-sticker-cursor-preview";
    document.body.appendChild(studioStickerPreviewEl);
  }
  return studioStickerPreviewEl;
}

// clientX/clientY omitted just re-applies the current size/rotation to
// wherever the preview already is (e.g. after adjusting the size slider).
function updateStudioStickerPreview(clientX?: number, clientY?: number) {
  if (!studioActiveSticker) return;
  const el = ensureStudioStickerPreviewEl();
  if (studioActiveSticker.kind === "emoji") {
    el.textContent = studioActiveSticker.glyph;
  } else {
    const activeShapeId = studioActiveSticker.id;
    el.innerHTML = STUDIO_SHAPE_STICKERS.find((s) => s.id === activeShapeId)!.svg;
  }
  const rect = studioViewportEl.getBoundingClientRect();
  const sizePx = (studioStickerSizePercent / 100) * Math.min(rect.width, rect.height);
  el.style.width = sizePx + "px";
  el.style.height = sizePx + "px";
  el.style.fontSize = sizePx + "px";
  el.style.transform =
    "translate(-50%, -50%) rotate(" + studioStickerRotateDeg + "deg) scaleX(" + (studioStickerMirrored ? -1 : 1) + ")";
  if (clientX !== undefined && clientY !== undefined) {
    el.style.left = clientX + "px";
    el.style.top = clientY + "px";
  }
  el.style.display = "block";
}

function hideStudioStickerPreview() {
  if (studioStickerPreviewEl) studioStickerPreviewEl.style.display = "none";
}

function armStudioSticker(sticker: StudioSticker) {
  studioActiveSticker = sticker;
  studioViewportEl.classList.add("sticker-armed");
  const key = studioStickerKey(sticker);
  document.querySelectorAll<HTMLButtonElement>(".studio-sticker-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.stickerKey === key);
  });
  updateStudioStickerPreview();
}

function disarmStudioSticker() {
  studioActiveSticker = null;
  studioViewportEl.classList.remove("sticker-armed");
  document.querySelectorAll<HTMLButtonElement>(".studio-sticker-btn").forEach((btn) => {
    btn.classList.remove("active");
  });
  hideStudioStickerPreview();
}

// Draws the already-loaded <img> onto an offscreen canvas and exports it as
// a blob — used both to send the current source to /studio/upscale and, via
// createMeme(), to bake the final meme. Never re-fetches the image's URL: a
// fetch() on a URL the browser already loaded via <img src> unreliably
// fails outright in some webviews (crossorigin="anonymous" on #studio-image
// in index.html is what keeps this canvas untainted instead).
async function canvasBlobFromImage(img: HTMLImageElement): Promise<Blob> {
  await img.decode();
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported.");
  ctx.drawImage(img, 0, 0);
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Failed to render image."))), "image/png"),
  );
}

// Sends the current studio source to Real-ESRGAN and returns the sharper
// result as a blob — used only to refine detail (studioExportSize() still
// caps the final export, so this never makes the exported file bigger).
async function upscaleStudioSource(scale: number): Promise<Blob> {
  const blob = await canvasBlobFromImage(studioImageEl);
  const body = new FormData();
  body.set("image", blob, "source.png");
  body.set("scale", String(scale));
  const res = await fetch(API_BASE + "/studio/upscale", { method: "POST", body });
  if (!res.ok) throw new Error("Upscale failed (" + res.status + ")");
  return res.blob();
}

// Flips the ACTIVE image's own pixels horizontally and bakes that straight
// into studioImageEl.src (same "commit immediately" pattern as
// applyStudioUpscale() below) — not a live CSS transform, so it composes
// for free with everything downstream that already reads natural pixels
// (pan/zoom/rotate math, SNAP, export) without any special-casing.
async function mirrorStudioImage() {
  if (!studioHasImage()) {
    setStudioStatus("No image loaded to mirror.", "status-error");
    return;
  }
  await studioImageEl.decode();
  const canvas = document.createElement("canvas");
  canvas.width = studioImageEl.naturalWidth;
  canvas.height = studioImageEl.naturalHeight;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(studioImageEl, 0, 0);
  const blob: Blob = await new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Mirror failed."))), "image/png"),
  );
  studioImageEl.src = URL.createObjectURL(blob); // "load" listener handles naturalW/H + resizeStudioViewport()
  setStudioStatus("Mirrored.", "status-ok");
}

// A standalone, explicit action (Apply button) rather than something that
// silently happens inside CREATE MEME — lets you see the refined result
// before committing to it, and re-upscaling an already-upscaled image adds
// negligible real detail while burning GPU time, so it's capped per image
// (resetStudioCrop(), called on every new image load, resets the counter).
async function applyStudioUpscale() {
  if (!studioHasImage()) {
    setStudioStatus("No image loaded to upscale.", "status-error");
    return;
  }
  const scale = currentStudioUpscaleScale();
  if (!scale) {
    setStudioStatus("Pick Upscale x2 or x4 first.", "status-error");
    return;
  }
  if (studioUpscaleApplyCount >= STUDIO_UPSCALE_APPLY_LIMIT) {
    setStudioStatus(
      "Whoa, slow down — this image's already been upscaled " + studioUpscaleApplyCount +
        "x. Load a fresh image to upscale again.",
      "status-error",
    );
    return;
  }
  studioUpscaleApplyBtnEl.disabled = true;
  setStudioStatus("Refining image (x" + scale + ")...", "status-pending");
  try {
    const blob = await upscaleStudioSource(scale);
    studioImageEl.src = URL.createObjectURL(blob); // "load" listener handles naturalW/H + resizeStudioViewport()
    studioUpscaleApplyCount++;
    setStudioStatus("Upscaled x" + scale + " applied.", "status-ok");
  } catch (err) {
    setStudioStatus("Upscale failed: " + (err as Error).message, "status-error");
  } finally {
    studioUpscaleApplyBtnEl.disabled = false;
  }
}

async function createMeme() {
  if (!studioHasImage()) {
    setStudioStatus("No image loaded to add text to.", "status-error");
    return;
  }
  const topText = studioTopTextEl.value.trim().toUpperCase();
  const bottomText = studioBottomTextEl.value.trim().toUpperCase();
  if (!topText && !bottomText && !studioPatchworkCanvas && !studioStickerCanvas) {
    setStudioStatus("Type a top or bottom phrase first.", "status-error");
    return;
  }

  studioCreateBtnEl.disabled = true;
  try {
    await studioImageEl.decode();
    const { w: exportW, h: exportH } = studioExportSize();
    const canvas = document.createElement("canvas");
    canvas.width = exportW;
    canvas.height = exportH;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas not supported.");
    // Same layer stack as the live preview: blurred fallback at the bottom,
    // crisp SNAP patchwork on top of it (only opaque where actually
    // stamped), the active foreground on top of both, then placed stickers
    // on top of the foreground, then the meme captions on top of everything.
    drawStudioBackdrop(ctx, studioImageEl, exportW, exportH);
    if (studioPatchworkCanvas) {
      ctx.drawImage(ensureStudioPatchworkCanvas(exportW, exportH), 0, 0, exportW, exportH);
    }
    drawStudioForeground(ctx, studioImageEl, exportW, exportH);
    if (studioStickerCanvas) {
      ctx.drawImage(ensureStudioStickerCanvas(exportW, exportH), 0, 0, exportW, exportH);
    }
    drawMemeText(ctx, topText, exportW, exportH, "top", studioTopTextOffset, studioTopStyle);
    drawMemeText(ctx, bottomText, exportW, exportH, "bottom", studioBottomTextOffset, studioBottomStyle);

    flashStudioCanvas();

    const blob: Blob = await new Promise((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Failed to render meme."))), "image/png"),
    );

    const body = new FormData();
    body.set("image", blob, "meme.png");
    const result = await fetchJson("/studio/save", { method: "POST", body });

    clearStudioScratchLayers(); // baked into the export above — start fresh, including undo history
    resetStudioFullState();
    loadStudioImage(result.output_url);
    studioTopTextEl.value = "";
    studioBottomTextEl.value = "";
    updateStudioTextOverlays();
    prependStudioThumbnail(result.output_url, result.output_url);
    setStudioStatus("meme created", "status-ok");
  } catch (err) {
    setStudioStatus("Meme creation failed: " + (err as Error).message, "status-error");
  } finally {
    studioCreateBtnEl.disabled = false;
  }
}

function onPromptKeydown(e: KeyboardEvent) {
  // Plain Enter submits (matches a chat-input convention); Shift+Enter still
  // inserts a newline for multi-line prompts.
  if (e.key === "Enter" && !e.shiftKey && promptEl.value.trim()) {
    e.preventDefault();
    if (!generateBtn.disabled) onGenerate();
  }
}

generateBtn.disabled = true;
generateBtn.addEventListener("click", onGenerate);
promptEl.addEventListener("keydown", onPromptKeydown);
modeRadios.forEach((r) => r.addEventListener("change", onModeChange));
iterationsEl.addEventListener("input", () => (iterationsValueEl.textContent = iterationsEl.value));
denoiseEl.addEventListener("input", () => (denoiseValueEl.textContent = Number(denoiseEl.value).toFixed(2)));
ddIntensityEl.addEventListener("input", () => (ddIntensityValueEl.textContent = ddIntensityEl.value));
upscale2xEl.addEventListener("change", () => {
  if (upscale2xEl.checked) upscale4xEl.checked = false;
});
upscale4xEl.addEventListener("change", () => {
  if (upscale4xEl.checked) upscale2xEl.checked = false;
});
stylePresetEl.addEventListener("change", () => {
  stylePresetToggleEl.checked = !!stylePresetEl.value;
  onStylePresetChange();
});
stylePresetToggleEl.addEventListener("change", () => {
  if (stylePresetToggleEl.checked && !stylePresetEl.value) {
    const firstPreset = Array.from(stylePresetEl.options).find((o) => o.value);
    if (firstPreset) stylePresetEl.value = firstPreset.value;
  }
  onStylePresetChange();
});
stylePresetStrengthEl.addEventListener(
  "input",
  () => (stylePresetStrengthValueEl.textContent = Number(stylePresetStrengthEl.value).toFixed(2)),
);
styleStrengthEl.addEventListener(
  "input",
  () => (styleStrengthValueEl.textContent = Number(styleStrengthEl.value).toFixed(2)),
);
contentWeightEl.addEventListener(
  "input",
  () => (contentWeightValueEl.textContent = Number(contentWeightEl.value).toFixed(2)),
);
styleStepsEl.addEventListener("input", () => (styleStepsValueEl.textContent = styleStepsEl.value));
styleImageEl.addEventListener("change", () => {
  const file = styleImageEl.files?.[0];
  if (!file) clearStyleImagePreview();
  else showStyleImagePreview(file);
});
styleImagePreviewEl.addEventListener("click", () => openLightbox(styleImagePreviewEl.src));
styleImageClearBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  styleImageEl.value = "";
  clearStyleImagePreview();
});
styleImageDownloadBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  downloadImage(styleImagePreviewEl.src);
});
styleImageCopyBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  copyImageToClipboard(styleImagePreviewEl.src);
});
initImageEl.addEventListener("change", onInitImageChange);
// Clicking the init image opens the same overlay as "Magic wand" — no
// separate plain-zoom lightbox for this one, since the overlay is a
// strict superset (zoomed view + all the SAM tools); openSamOverlay()
// already no-ops unless an image is actually loaded. Video preview still
// uses its own click-to-play handling untouched.
initImagePreviewEl.addEventListener("click", openSamOverlay);
initImageClearBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!hasInitFile()) {
    setGenStatus("Nothing to remove — img2img is empty.", "status-error");
    return;
  }
  initImageEl.value = "";
  clearInitImagePreview();
});
initImagePasteBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  pasteInitImage();
});
initImageDownloadBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!hasInitFile()) {
    setGenStatus("Nothing to download — img2img is empty.", "status-error");
    return;
  }
  downloadImage(hasInitVideo() ? initVideoPreviewEl.src : initImagePreviewEl.src);
});
initImageCopyBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (hasInitVideo()) {
    setGenStatus("Video can't be copied to clipboard — use download instead.", "status-error");
    return;
  }
  if (!hasInitImage()) {
    setGenStatus("Nothing to copy — img2img is empty.", "status-error");
    return;
  }
  copyImageToClipboard(initImagePreviewEl.src);
});
videoRangeHandleFromEl.addEventListener("pointerdown", (e) => startVideoRangeDrag("from", e));
videoRangeHandleToEl.addEventListener("pointerdown", (e) => startVideoRangeDrag("to", e));
samWandBtnEl.addEventListener("click", openSamOverlay);
samOverlayCloseBtnEl.addEventListener("click", closeSamOverlay);
samOverlayEl.addEventListener("click", (e) => {
  if (e.target === samOverlayEl) closeSamOverlay(); // backdrop only — keeps the selection
});
samOverlayMaskCanvasEl.addEventListener("click", onSamOverlayCanvasClick);
samClearBtnEl.addEventListener("click", clearSamMask);
samOverlayClearBtnEl.addEventListener("click", unselectSam);
samOverlayResetBtnEl.addEventListener("click", resetCutMask);
samInvertBtnEl.addEventListener("click", () => {
  samInvertOn = !samInvertOn;
  samInvertBtnEl.classList.toggle("active", samInvertOn);
  renderSamMaskPreview();
});
samCutBtnEl.addEventListener("click", () => void commitCutMask(false));
samKeepBtnEl.addEventListener("click", () => void commitCutMask(true));
samExportCutBtnEl.addEventListener("click", () => void exportCutFromOutput());

// Highlights whichever preset swatch matches the color currently loaded in
// the native picker — including after the picker itself is used directly,
// so picking exactly green/blue/pink there still lights up the matching
// preset instead of looking unrelated to it. No match (a genuinely custom
// color) just leaves all three unlit; the native swatch already shows its
// own color as feedback.
function syncActiveSwatch() {
  const current = samCutColorEl.value.toLowerCase();
  samColorSwatchBtns.forEach((btn) => {
    btn.classList.toggle("active", (btn.dataset.color || "").toLowerCase() === current);
  });
}

function updateSamCutColorRowVisibility() {
  const isColor = document.querySelector<HTMLInputElement>('input[name="sam-cut-mode"]:checked')?.value === "color";
  samCutColorRowEl.hidden = !isColor;
}

samColorSwatchBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    samCutColorEl.value = btn.dataset.color || samCutColorEl.value;
    syncActiveSwatch();
    renderSamMaskPreview();
  });
});
samCutColorEl.addEventListener("input", () => {
  syncActiveSwatch();
  renderSamMaskPreview();
});
samCutModeRadios.forEach((radio) => radio.addEventListener("change", () => {
  updateSamCutColorRowVisibility();
  renderSamMaskPreview();
}));
samShowMaskToggleEl.addEventListener("change", renderSamMaskPreview);
// Small +/- badge cursor over the overlay canvas while Ctrl is held, so
// add-vs-subtract is visible before you click, not just after (mirrors
// onSamOverlayCanvasClick's own e.ctrlKey check).
document.addEventListener("keydown", (e) => {
  if (e.key === "Control" && !samOverlayEl.hidden) samOverlayMaskCanvasEl.classList.add("sam-subtract-cursor");
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Control") samOverlayMaskCanvasEl.classList.remove("sam-subtract-cursor");
});
// Segment size only takes effect on the NEXT click (it picks which of
// SAM2's per-point candidate masks a future segment() call uses) — it
// deliberately does NOT re-render/re-fetch the current selection, so the
// user can pick big segments first and switch to a finer size to refine
// without losing what's already selected.
samSizeEl.addEventListener("input", () => {
  const v = Number(samSizeEl.value);
  samSizeValueEl.textContent = v < -0.33 ? "Fine" : v > 0.33 ? "Coarse" : "Balanced";
});
samSoftenEl.addEventListener("input", () => {
  samSoftenValueEl.textContent = samSoftenEl.value;
  renderSamMaskPreview();
});
samExpandEl.addEventListener("input", () => {
  samExpandValueEl.textContent = samExpandEl.value;
  renderSamMaskPreview();
});
window.addEventListener("resize", () => {
  if (samPoints.length > 0) {
    positionSamCanvas();
    if (!samOverlayEl.hidden) positionSamOverlayCanvas();
  }
});
videoCodecsInstallBtnEl.addEventListener("click", async () => {
  videoCodecsInstallBtnEl.disabled = true;
  videoCodecsInstallBtnEl.textContent = "Installing… (~150MB, may take a minute)";
  try {
    await fetchJson("/video/codecs/install", { method: "POST" });
  } catch (err) {
    setGenStatus("Codecs install failed: " + (err as Error).message, "status-error");
    videoCodecsInstallBtnEl.disabled = false;
    videoCodecsInstallBtnEl.textContent = "Install codecs";
    return;
  }
  const poll = async () => {
    await refreshVideoCodecsGate();
    if (videoCodecsReady) {
      videoCodecsInstallBtnEl.textContent = "Install codecs";
      return;
    }
    setTimeout(poll, 3000);
  };
  poll();
});
galleryBtnEl.addEventListener("click", onGalleryClick);
// Always visible/enabled (not just during a video job) — a single global
// cancel flag on the backend (blobvision_cancel) is checked by every long
// generation loop (VQGAN, DeepDream, Style Transfer, and each family's
// video pipeline), so this same endpoint cancels whatever's currently
// running, image or video. Harmless to click when nothing is running: the
// flag just gets cleared again at the start of the next generate() call.
cancelBtnEl.addEventListener("click", async () => {
  try {
    await fetchJson("/cancel", { method: "POST" });
  } catch (err) {
    setGenStatus("Cancel failed: " + (err as Error).message, "status-error");
  }
});
outputImageEl.addEventListener("click", () => openLightbox(outputImageEl.src));
outputClearBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!hasOutputImage() && !hasOutputVideo()) {
    setGenStatus("Nothing to remove — no output yet.", "status-error");
    return;
  }
  clearOutput();
});
useAsInitBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  useOutputAsInit();
});
downloadBtnEl.addEventListener("click", () => {
  if (hasOutputVideo()) {
    downloadImage(outputVideoEl.src); // works for any URL, not just images — just triggers an <a download>
    return;
  }
  if (!hasOutputImage()) {
    setGenStatus("Nothing to download — no output yet.", "status-error");
    return;
  }
  downloadImage(outputImageEl.src);
});
copyBtnEl.addEventListener("click", () => {
  if (hasOutputVideo()) {
    setGenStatus("Video results can't be copied to clipboard — use download instead.", "status-error");
    return;
  }
  if (!hasOutputImage()) {
    setGenStatus("Nothing to copy — no output yet.", "status-error");
    return;
  }
  copyImageToClipboard(outputImageEl.src);
});
lightboxEl.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!lightboxEl.hidden) closeLightbox();
  else if (!grimoireOverlayEl.hidden) closeGrimoireOverlay();
  else if (!samOverlayEl.hidden) closeSamOverlay();
  else if (studioActiveSticker) disarmStudioSticker();
});
formatBtnEl.addEventListener("click", () => openAnchoredMenu(formatBtnEl, renderFormatMenu));
promptGrimoireBtnEl.addEventListener("click", () =>
  openAnchoredMenu(promptGrimoireBtnEl, (menu) => renderGrimoirePicker("prompt", promptEl, menu)),
);
negativeGrimoireBtnEl.addEventListener("click", () =>
  openAnchoredMenu(negativeGrimoireBtnEl, (menu) => renderGrimoirePicker("negative", negativePromptEl, menu)),
);
grimoireBtnEl.addEventListener("click", openGrimoireOverlay);
grimoireCloseBtnEl.addEventListener("click", closeGrimoireOverlay);
grimoireOverlayEl.addEventListener("click", (e) => {
  if (e.target === grimoireOverlayEl) closeGrimoireOverlay();
});
grimoireTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const kind = tab.dataset.kind as GrimoireKind | undefined;
    if (kind) {
      grimoireActiveKind = kind;
      renderGrimoireOverlay();
    }
  });
});
familyBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    const family = btn.dataset.family as Family | undefined;
    if (family && !btn.disabled) switchFamily(family);
  });
});

// Drag-and-drop onto the img2img zone — mirrors the Gradio sketch panel's
// drag-drop upload (blob-sketch-image / blob-source-image / dd-sketch-image).
// Also accepts video/* for all three families — see setInitVideoFile.
["dragenter", "dragover"].forEach((evt) =>
  initImageDropEl.addEventListener(evt, (e) => {
    e.preventDefault();
    initImageDropEl.classList.add("drop-zone-active");
  }),
);
["dragleave", "dragend"].forEach((evt) =>
  initImageDropEl.addEventListener(evt, () => initImageDropEl.classList.remove("drop-zone-active")),
);
initImageDropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  initImageDropEl.classList.remove("drop-zone-active");
  const file = e.dataTransfer?.files?.[0];
  if (!file) return;
  if (file.type.startsWith("image/")) {
    setInitImageFile(file);
  } else if (file.type.startsWith("video/")) {
    setInitVideoFile(file);
  }
});

// Style reference drop zone (classic Style Transfer only).
["dragenter", "dragover"].forEach((evt) =>
  styleImageDropEl.addEventListener(evt, (e) => {
    e.preventDefault();
    styleImageDropEl.classList.add("drop-zone-active");
  }),
);
["dragleave", "dragend"].forEach((evt) =>
  styleImageDropEl.addEventListener(evt, () => styleImageDropEl.classList.remove("drop-zone-active")),
);
styleImageDropEl.addEventListener("drop", (e) => {
  e.preventDefault();
  styleImageDropEl.classList.remove("drop-zone-active");
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith("image/")) setStyleImageFile(file);
});

studioBtnEl.addEventListener("click", () => {
  if (studioMode) exitStudioMode();
  else enterStudioMode();
});
studioImageEl.addEventListener("load", () => {
  studioNaturalW = studioImageEl.naturalWidth;
  studioNaturalH = studioImageEl.naturalHeight;
  resizeStudioViewport();
});
studioAspectRadios.forEach((r) =>
  r.addEventListener("change", () => {
    studioAspect = currentStudioAspect();
    studioZoom = 1;
    studioPanX = 0;
    studioPanY = 0;
    studioZoomEl.value = "1";
    studioZoomValueEl.textContent = "1.00";
    resizeStudioViewport();
  }),
);
studioZoomEl.addEventListener("input", () => setStudioZoom(Number(studioZoomEl.value)));
studioImageRotateEl.addEventListener("input", () => setStudioImageRotate(Number(studioImageRotateEl.value)));
studioViewportEl.addEventListener("wheel", (e) => {
  if (!studioHasImage()) return;
  e.preventDefault();
  setStudioZoom(studioZoom + (e.deltaY > 0 ? -0.1 : 0.1));
}, { passive: false });
studioViewportEl.addEventListener("mousedown", (e) => {
  if (!studioHasImage()) return;
  if (studioActiveSticker) {
    e.preventDefault();
    placeStudioSticker(e.clientX, e.clientY);
    return;
  }
  studioPanning = true;
  studioViewportEl.classList.add("panning");
  studioPanStartClientX = e.clientX;
  studioPanStartClientY = e.clientY;
  studioPanStartPanX = studioPanX;
  studioPanStartPanY = studioPanY;
  e.preventDefault();
});
studioViewportEl.addEventListener("mousemove", (e) => {
  if (studioActiveSticker) updateStudioStickerPreview(e.clientX, e.clientY);
});
studioViewportEl.addEventListener("mouseleave", () => {
  if (studioActiveSticker) hideStudioStickerPreview();
});
// Angle (degrees) from the layer's current on-screen center to a client
// (viewport-relative) point — shared by handle mousedown (drag start) and
// mousemove (drag delta) so both read the exact same center.
function studioAngleFromLayerCenter(clientX: number, clientY: number): number {
  const rect = studioViewportEl.getBoundingClientRect();
  const cx = rect.left + studioLayerCenterX;
  const cy = rect.top + studioLayerCenterY;
  return (Math.atan2(clientY - cy, clientX - cx) * 180) / Math.PI;
}
document.querySelectorAll<HTMLElement>(".studio-rotate-handle").forEach((handle) => {
  handle.addEventListener("mousedown", (e) => {
    if (!studioHasImage()) return;
    e.preventDefault();
    e.stopPropagation(); // don't also start a viewport pan
    studioRotating = true;
    studioRotateStartAngle = studioAngleFromLayerCenter(e.clientX, e.clientY);
    studioRotateStartValue = studioImageRotate;
  });
});
window.addEventListener("mousemove", (e) => {
  if (studioRotating) {
    const angle = studioAngleFromLayerCenter(e.clientX, e.clientY);
    setStudioImageRotate(studioRotateStartValue + (angle - studioRotateStartAngle));
    return;
  }
  if (!studioPanning) return;
  const rect = studioViewportEl.getBoundingClientRect();
  const posBase = studioCoverBase(rect.width, rect.height, studioNaturalW, studioNaturalH);
  const dx = e.clientX - studioPanStartClientX;
  const dy = e.clientY - studioPanStartClientY;
  // No rotation compensation needed here: #studio-image-bg's rotation now
  // pivots around its OWN center (screen position viewport-center −
  // pan*posBase — see updateStudioImageBg), which doesn't depend on
  // rotation angle, so screen-space dx/dy maps straight onto panX/panY
  // regardless of how the image is currently rotated. (An earlier version
  // pivoted rotation around the fixed viewport center instead, which
  // visibly coupled panX/panY into both screen axes and needed an
  // inverse-rotation correction here — dragging straight up would otherwise
  // go diagonal at 45°. That coupling no longer exists, so the correction
  // was removed rather than left in as now-wrong dead weight.) Divides by
  // posBase (NOT the zoomed scale) for the same reason zoom uses it for
  // position above — keeps 1:1 screen-pixel tracking independent of zoom.
  // Free drag, no clamping — dragging past the image's own edge is fine;
  // #studio-image-backdrop is there specifically to fill that space.
  studioPanX = studioPanStartPanX - dx / posBase;
  studioPanY = studioPanStartPanY - dy / posBase;
  updateStudioImageBg();
});
window.addEventListener("mouseup", () => {
  studioRotating = false;
  if (studioPanning) {
    studioPanning = false;
    studioViewportEl.classList.remove("panning");
  }
});
wireStudioTextDrag(studioTextTopEl, studioTopTextOffset, studioTopStyle);
wireStudioTextDrag(studioTextBottomEl, studioBottomTextOffset, studioBottomStyle);
studioTopTextEl.addEventListener("input", updateStudioTextOverlays);
studioBottomTextEl.addEventListener("input", updateStudioTextOverlays);
studioTopRotateScrub = wireScrubber(studioTopRotateEl, {
  min: -45, max: 45, step: 1, initial: 0, decimals: 0,
  onChange: (v) => { studioTopStyle.rotate = v; updateStudioTextOverlays(); },
});
studioTopScaleScrub = wireScrubber(studioTopScaleEl, {
  min: 0.3, max: 3, step: 0.05, initial: 1, decimals: 2,
  onChange: (v) => { studioTopStyle.scale = v; updateStudioTextOverlays(); },
});
studioTopContourScrub = wireScrubber(studioTopContourEl, {
  min: 0, max: 0.25, step: 0.01, initial: 0.08, decimals: 2,
  onChange: (v) => { studioTopStyle.contour = v; updateStudioTextOverlays(); },
});
studioBottomRotateScrub = wireScrubber(studioBottomRotateEl, {
  min: -45, max: 45, step: 1, initial: 0, decimals: 0,
  onChange: (v) => { studioBottomStyle.rotate = v; updateStudioTextOverlays(); },
});
studioBottomScaleScrub = wireScrubber(studioBottomScaleEl, {
  min: 0.3, max: 3, step: 0.05, initial: 1, decimals: 2,
  onChange: (v) => { studioBottomStyle.scale = v; updateStudioTextOverlays(); },
});
studioBottomContourScrub = wireScrubber(studioBottomContourEl, {
  min: 0, max: 0.25, step: 0.01, initial: 0.08, decimals: 2,
  onChange: (v) => { studioBottomStyle.contour = v; updateStudioTextOverlays(); },
});
wireColorSwatch(studioTopColorSwatchEl, studioTopFillColorEl, studioTopStrokeColorEl, studioTopStyle);
wireColorSwatch(studioBottomColorSwatchEl, studioBottomFillColorEl, studioBottomStrokeColorEl, studioBottomStyle);
studioUpscale2xEl.addEventListener("change", () => {
  if (studioUpscale2xEl.checked) studioUpscale4xEl.checked = false;
});
studioUpscale4xEl.addEventListener("change", () => {
  if (studioUpscale4xEl.checked) studioUpscale2xEl.checked = false;
});
studioUpscaleApplyBtnEl.addEventListener("click", applyStudioUpscale);
studioMirrorBtnEl.addEventListener("click", mirrorStudioImage);
studioUndoBtnEl.addEventListener("click", undoStudioLastAction);
studioSnapBtnEl.addEventListener("click", snapStudioImage);
function studioStickerButtonClick(sticker: StudioSticker) {
  if (studioActiveSticker && studioStickerKey(studioActiveSticker) === studioStickerKey(sticker)) disarmStudioSticker();
  else armStudioSticker(sticker);
}
STUDIO_STICKERS.forEach((glyph) => {
  const sticker: StudioSticker = { kind: "emoji", glyph };
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "studio-sticker-btn";
  btn.textContent = glyph;
  btn.dataset.stickerKey = studioStickerKey(sticker);
  btn.title = glyph;
  btn.addEventListener("click", () => studioStickerButtonClick(sticker));
  studioStickerGridEl.appendChild(btn);
});
STUDIO_SHAPE_STICKERS.forEach((shapeDef) => {
  const sticker: StudioSticker = { kind: "shape", id: shapeDef.id };
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "studio-sticker-btn";
  btn.innerHTML = shapeDef.svg;
  btn.dataset.stickerKey = studioStickerKey(sticker);
  btn.title = shapeDef.label;
  btn.addEventListener("click", () => studioStickerButtonClick(sticker));
  studioStickerShapeGridEl.appendChild(btn);
});
studioStickerSizeEl.addEventListener("input", () => {
  setStudioStickerSize(Number(studioStickerSizeEl.value));
  updateStudioStickerPreview();
});
studioStickerRotateEl.addEventListener("input", () => setStudioStickerRotate(Number(studioStickerRotateEl.value)));
studioStickerMirrorBtnEl.addEventListener("click", toggleStudioStickerMirror);
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || !studioMode) return;
  // Space is the shortcut for SNAP, but only when focus isn't in a text
  // field — otherwise it should just type a space, not stamp the image.
  const active = document.activeElement;
  const isTyping = active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement;
  if (isTyping) return;
  e.preventDefault();
  snapStudioImage();
});
window.addEventListener("resize", () => {
  if (studioMode) resizeStudioViewport();
});
studioCreateBtnEl.addEventListener("click", createMeme);
studioClearBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!studioHasImage()) {
    setStudioStatus("Nothing to remove — no image loaded.", "status-error");
    return;
  }
  clearStudioImage();
});
studioPasteBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  pasteStudioImage();
});
studioCopyBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!studioHasImage()) {
    setStudioStatus("Nothing to copy — no image loaded.", "status-error");
    return;
  }
  copyImageToClipboard(studioImageEl.src);
});
studioDownloadBtnEl.addEventListener("click", (e) => {
  e.stopPropagation();
  if (!studioHasImage()) {
    setStudioStatus("Nothing to download — no image loaded.", "status-error");
    return;
  }
  downloadImage(studioImageEl.src);
});

denoiseEl.value = String(DEFAULT_DENOISE);
denoiseValueEl.textContent = DEFAULT_DENOISE.toFixed(2);
onModeChange();
waitForEngine();
