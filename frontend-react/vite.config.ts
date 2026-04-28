import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/chat": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
      "/upload": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
      },
    },
  },
});
