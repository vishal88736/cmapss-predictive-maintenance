import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    port: 5183,
    host: true,
    proxy: { '/api': { target: process.env.CMAPSS_API_PROXY || 'http://127.0.0.1:8000', changeOrigin: true } }
  }
})
