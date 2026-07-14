package handlers

import (
	"net/http"
	"os"
	"path"
	"strings"
)

// StaticFileServer serves static files and handles SPA routing
func StaticFileServer(staticDir string) http.Handler {
	// Prefer dist/ subfolder if it exists (production build output)
	distDir := path.Join(staticDir, "dist")
	if info, err := os.Stat(distDir); err == nil && info.IsDir() {
		staticDir = distDir
	}
	fs := http.FileServer(http.Dir(staticDir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never serve index.html for API or embedded proxy routes
		// These should be handled by their respective handlers
		p := r.URL.Path
		if p == "/.well-known/change-password" {
			if r.Method != http.MethodGet && r.Method != http.MethodHead {
				http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
				return
			}
			http.Redirect(w, r, "/account/security", http.StatusFound)
			return
		}
		// Password managers probe a reserved nonexistent path to determine
		// whether this server returns real 404s. Do not let the SPA fallback
		// turn unknown well-known resources into misleading 200 responses.
		if strings.HasPrefix(p, "/.well-known/") {
			http.NotFound(w, r)
			return
		}
		// Never serve static files for proxy routes
		if strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/embedded/") ||
			strings.HasPrefix(p, "/metrics/") || strings.HasPrefix(p, "/public/") ||
			strings.HasPrefix(p, "/avatar/") || strings.HasPrefix(p, "/static/") ||
			p == "/logout" ||
			strings.HasPrefix(p, "/r/") {
			// These paths should have been handled by other handlers
			// If we reach here, it means the proxy failed or route not found
			w.Header().Set("Content-Type", "application/json")
			http.Error(w, `{"error":"Route not found","message":"This path should have been handled by a proxy"}`, http.StatusBadGateway)
			return
		}

		full := path.Join(staticDir, path.Clean(p))

		// Check if file exists
		info, err := os.Stat(full)
		if err == nil {
			// File exists
			if !info.IsDir() {
				// Hashed assets (Vite bundles) can be cached forever;
				// everything else (index.html) must be revalidated.
				setCacheHeaders(w, p)
				fs.ServeHTTP(w, r)
				return
			}
			// It's a directory, try index.html
			indexPath := path.Join(full, "index.html")
			if _, err := os.Stat(indexPath); err == nil {
				setNoCacheHeaders(w)
				http.ServeFile(w, r, indexPath)
				return
			}
		}

		// File doesn't exist or is directory without index.html
		// For SPA routing: serve index.html for routes without file extension
		if !strings.Contains(path.Base(p), ".") {
			setNoCacheHeaders(w)
			http.ServeFile(w, r, path.Join(staticDir, "index.html"))
			return
		}

		// Otherwise let the file server handle it (will return 404)
		fs.ServeHTTP(w, r)
	})
}

// setNoCacheHeaders prevents browser caching (used for index.html).
func setNoCacheHeaders(w http.ResponseWriter) {
	w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
	w.Header().Set("Pragma", "no-cache")
	w.Header().Set("Expires", "0")
}

// setCacheHeaders sets appropriate caching for static assets.
// Vite-hashed files (contain hash in name) can be cached aggressively.
func setCacheHeaders(w http.ResponseWriter, p string) {
	base := path.Base(p)
	// Vite bundles have content hashes like index-abc123.js
	if strings.Contains(base, "-") && (strings.HasSuffix(base, ".js") || strings.HasSuffix(base, ".css")) {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	}
	// All other static files: short cache with revalidation
}
