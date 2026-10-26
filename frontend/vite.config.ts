/// <reference types="vitest/config" />
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

export default defineConfig(({ mode }) => {
  // BV_API_PROXY (env var or .env file) points the dev proxy at a backend on another port.
  const env = { ...loadEnv(mode, process.cwd(), ""), ...process.env };
  const target = env.BV_API_PROXY || "http://127.0.0.1:8000";
  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: { "/api": { target, changeOrigin: false } },
    },
    test: { environment: "node", include: ["src/**/*.test.ts"] },
  };
});
