import { defineConfig } from "vite";

export default defineConfig({
  // Set base path for GitHub Pages
  base: "./",

  build: {
    target: "es2020",
    outDir: "dist",
    assetsDir: "assets",
  },
});
