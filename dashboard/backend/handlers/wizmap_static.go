package handlers

import (
	"fmt"
	"html"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const wizMapEmbeddedPrefix = "/embedded/wizmap"

func WizMapStaticHandler(staticDir string) http.HandlerFunc {
	root, err := resolveWizMapStaticDir(staticDir)
	if err != nil {
		return func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			renderWizMapUnavailable(w, err)
		}
	}

	fs := http.FileServer(http.Dir(root))
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == wizMapEmbeddedPrefix {
			http.Redirect(w, r, wizMapEmbeddedPrefix+"/", http.StatusTemporaryRedirect)
			return
		}

		trimmed := strings.TrimPrefix(r.URL.Path, wizMapEmbeddedPrefix)
		if trimmed == "" {
			trimmed = "/"
		}
		cleaned := filepath.Clean(trimmed)
		if cleaned == "." {
			cleaned = "/"
		}
		if cleaned == "/" {
			setNoCacheHeaders(w)
			http.ServeFile(w, r, filepath.Join(root, "index.html"))
			return
		}

		targetPath := filepath.Join(root, strings.TrimPrefix(cleaned, "/"))
		info, statErr := os.Stat(targetPath)
		if statErr == nil && !info.IsDir() {
			setCacheHeaders(w, cleaned)
			r2 := r.Clone(r.Context())
			r2.URL.Path = cleaned
			fs.ServeHTTP(w, r2)
			return
		}

		if !strings.Contains(filepath.Base(cleaned), ".") {
			setNoCacheHeaders(w)
			http.ServeFile(w, r, filepath.Join(root, "index.html"))
			return
		}

		r2 := r.Clone(r.Context())
		r2.URL.Path = cleaned
		fs.ServeHTTP(w, r2)
	}
}

func resolveWizMapStaticDir(staticDir string) (string, error) {
	candidates := []string{
		filepath.Join(staticDir, "dist", "embedded", "wizmap"),
		filepath.Join(staticDir, "embedded", "wizmap"),
		filepath.Join(filepath.Dir(staticDir), "wizmap", "dist"),
	}
	for _, candidate := range candidates {
		indexPath := filepath.Join(candidate, "index.html")
		if info, err := os.Stat(indexPath); err == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	return "", fmt.Errorf("WizMap static bundle not found; run the dashboard frontend build to generate embedded assets")
}

func renderWizMapUnavailable(w http.ResponseWriter, err error) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusServiceUnavailable)
	_, _ = fmt.Fprintf(w, `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>WizMap unavailable</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f7fb; color: #172033; margin: 0; padding: 32px; }
      main { max-width: 720px; margin: 64px auto; background: white; border: 1px solid #dce3f0; border-radius: 20px; padding: 32px; box-shadow: 0 20px 60px rgba(20, 36, 68, 0.08); }
      h1 { margin-top: 0; font-size: 28px; }
      p { line-height: 1.6; }
      code { display: inline-block; background: #eef3ff; border-radius: 8px; padding: 4px 8px; }
    </style>
  </head>
  <body>
    <main>
      <h1>Knowledge Map is not available yet</h1>
      <p>The self-hosted WizMap static bundle is missing from this dashboard instance.</p>
      <p><code>%s</code></p>
    </main>
  </body>
</html>`, html.EscapeString(err.Error()))
}
