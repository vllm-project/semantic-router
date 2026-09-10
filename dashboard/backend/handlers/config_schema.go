package handlers

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

// ConfigSchemaHandler serves the exact generated Go Router contract consumed
// by this Dashboard build. It remains available before the managed Router is
// running, which lets setup, forms, and automation discover supported fields.
func ConfigSchemaHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		etag := configschema.ETag()
		w.Header().Set("Content-Type", "application/schema+json")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("ETag", etag)
		if r.Header.Get("If-None-Match") == etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}
		_, _ = w.Write(configschema.Document())
	}
}
