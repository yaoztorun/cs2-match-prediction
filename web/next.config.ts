import type { NextConfig } from "next";

/**
 * When NEXT_PUBLIC_API_BASE_URL is unset, the frontend calls the API
 * same-origin at /api/* and this rewrite proxies to the FastAPI backend
 * (API_PROXY_TARGET, dev default http://localhost:8000). This decouples the
 * frontend origin from the backend CORS allowlist — no backend config
 * changes needed regardless of which port the frontend serves on.
 * Setting NEXT_PUBLIC_API_BASE_URL switches to direct cross-origin calls.
 */
const API_PROXY_TARGET = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    if (process.env.NEXT_PUBLIC_API_BASE_URL) return [];
    return [
      {
        source: "/api/:path*",
        destination: `${API_PROXY_TARGET}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
