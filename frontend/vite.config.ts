import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      'raf': resolve(__dirname, 'node_modules/raf/index.js'),
      // 唯一正确路径：指向 rgbcolor 根目录的 index.js
      'rgbcolor': resolve(__dirname, 'node_modules/rgbcolor/index.js'),
      'resize-observer-polyfill': resolve(__dirname, 'node_modules/resize-observer-polyfill/dist/ResizeObserver.global.js')
    }
  },
  optimizeDeps: {
    include: ['raf', 'rgbcolor', 'resize-observer-polyfill'],
    esbuildOptions: {
      define: {
        global: 'globalThis'
      }
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
})