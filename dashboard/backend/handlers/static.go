package handlers

import (
	"net/http"
	"os"
	"path"
	"strings"
)

// StaticFileServer serves static files and handles SPA routing
func StaticFileServer(staticDir string) http.Handler {
	fs := http.FileServer(http.Dir(staticDir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never serve index.html for API or embedded proxy routes
		// These should be handled by their respective handlers
		p := r.URL.Path
		// Never serve static files for proxy routes or ChatUI API endpoints
		if strings.HasPrefix(p, "/api/") || strings.HasPrefix(p, "/embedded/") ||
			strings.HasPrefix(p, "/metrics/") || strings.HasPrefix(p, "/public/") ||
			strings.HasPrefix(p, "/avatar/") || strings.HasPrefix(p, "/_app/") ||
			strings.HasPrefix(p, "/_next/") || strings.HasPrefix(p, "/chatui/") ||
			p == "/conversation" || strings.HasPrefix(p, "/conversations") ||
			strings.HasPrefix(p, "/settings") || p == "/login" || p == "/logout" ||
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
				// It's a file, serve it
				fs.ServeHTTP(w, r)
				return
			}
			// It's a directory, try index.html
			indexPath := path.Join(full, "index.html")
			if _, err := os.Stat(indexPath); err == nil {
				http.ServeFile(w, r, indexPath)
				return
			}
		}

		// File doesn't exist or is directory without index.html
		// For SPA routing: serve index.html for routes without file extension
		if !strings.Contains(path.Base(p), ".") {
			http.ServeFile(w, r, path.Join(staticDir, "index.html"))
			return
		}

		// Otherwise let the file server handle it (will return 404)
		fs.ServeHTTP(w, r)
	})
}
