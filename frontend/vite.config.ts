import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    // No proxy - frontend calls API directly via CORS
    // This keeps frontend and backend completely separate
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
});
