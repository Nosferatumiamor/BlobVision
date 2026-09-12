import { defineConfig } from "vite";

// Standard Tauri-recommended Vite setup: fixed dev port (must match
// `build.devUrl` in src-tauri/tauri.conf.json), strict so Tauri never opens a
// window pointed at the wrong port, and the dev server ignores src-tauri/ so a
// `cargo build` output doesn't trigger a frontend reload loop.
export default defineConfig({
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: {
      ignored: ["**/src-tauri/**"],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    minify: !process.env.TAURI_ENV_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
});
