/**
 * Conservative V1 service worker (Phase 10A amendments #4/#5).
 *
 * Strategy, deliberately minimal:
 *   - `/api/` paths and ANY cross-origin request (the FastAPI backend):
 *     NEVER intercepted — network only. A prediction response is never
 *     cached or replayed as though newly computed.
 *   - Navigations / HTML: NETWORK FIRST. Offline fallback to a tiny
 *     pre-cached /offline page. A stale HTML document referencing chunks
 *     from a previous build is never served from cache while online.
 *   - /_next/static/ (content-addressed, immutable filenames): cache-first.
 *   - Manifest + placeholder icons: cache-first.
 *
 * The versioned cache name means every SW update deletes all previous
 * caches on activation.
 */
const CACHE_NAME = "cmi-shell-v1";
const PRECACHE = ["/offline", "/manifest.webmanifest", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)),
        ),
      )
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Backend traffic: never intercepted (network only, never cached).
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return;

  // Navigations: network first, offline fallback.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).catch(() =>
        caches
          .match("/offline")
          .then((cached) => cached ?? Response.error()),
      ),
    );
    return;
  }

  // Content-addressed Next static assets: cache first.
  if (url.pathname.startsWith("/_next/static/")) {
    event.respondWith(
      caches.match(request).then(
        (cached) =>
          cached ??
          fetch(request).then((response) => {
            if (response.ok) {
              const clone = response.clone();
              caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
            }
            return response;
          }),
      ),
    );
    return;
  }

  // Manifest / placeholder icons: cache first.
  if (PRECACHE.includes(url.pathname)) {
    event.respondWith(
      caches.match(request).then((cached) => cached ?? fetch(request)),
    );
  }
});
