import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// base is relative so the build works under any GitHub Pages path
// (https://<user>.github.io/<repo>/) as well as at a domain root.
export default defineConfig({
  base: './',
  plugins: [svelte()],
  build: { target: 'esnext', outDir: 'dist' },
});
