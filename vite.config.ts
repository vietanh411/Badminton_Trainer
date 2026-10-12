import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PORT = Number(process.env.PORT ?? 3001);

export default defineConfig({
  plugins: [react()],
  // MediaPipe ships its own wasm loader; let Vite serve it as a plain asset.
  optimizeDeps: { exclude: ["@mediapipe/tasks-vision"] },
  server: {
    port: 5173,
    proxy: { "/api": `http://localhost:${API_PORT}` },
  },
});
