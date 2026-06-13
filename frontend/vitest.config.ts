import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Vitest takes precedence over vite.config.ts when both exist. Tests don't need the dev proxy or
// Tailwind, so this is a lean jsdom config: just the React transform + the component-test setup.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    css: false,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
