import { defineConfig } from 'vite';
import { resolve } from 'path';

// https://vitejs.dev/guide/build.html#multi-page-app
export default defineConfig({
  // Dev server: proxy /api/* and the WebSocket upgrade to the running Rust backend.
  // Start the Rust server first (`cargo run --release`), then `npm run dev`.
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target:    'http://0.0.0.0:7272',
        changeOrigin: true,
        ws: true,   // forward WebSocket upgrades for /api/events
      },
    },
  },

  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        // Every HTML page that can be opened as a standalone tab
        main:             resolve(__dirname, 'index.html'),
        visualizer:       resolve(__dirname, 'visualizer.html'),
        analysis_viewer:  resolve(__dirname, 'analysis_viewer.html'),
        umbrella_viewer:  resolve(__dirname, 'umbrella_viewer.html'),
      },
    },
  },
});
