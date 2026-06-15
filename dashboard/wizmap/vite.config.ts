import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

// // https://vitejs.dev/config/
// export default defineConfig({
//   plugins: [svelte()]
// });

// https://vitejs.dev/config/
export default defineConfig(({ command, mode }) => {
  if (command === 'serve') {
    // Development
    return {
      base: './',
      plugins: [svelte()]
    };
  } else if (command === 'build') {
    switch (mode) {
      case 'production': {
        // Production: standard web page (default mode)
        return {
          base: './',
          build: {
            outDir: 'dist'
          },
          plugins: [svelte()]
        };
      }

      case 'vercel': {
        // Production: for vercel demo
        return {
          base: './',
          build: {
            outDir: 'dist'
          },
          plugins: [svelte()]
        };
      }

      case 'github': {
        // Production: github page
        return {
          base: '/wizmap/',
          build: {
            outDir: 'gh-page'
          },
          plugins: [svelte()]
        };
      }

      default: {
        console.error(`Unknown production mode ${mode}`);
        return null;
      }
    }
  }
});
