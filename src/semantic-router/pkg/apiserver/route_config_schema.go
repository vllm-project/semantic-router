//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func (s *ClassificationAPIServer) handleConfigSchema(w http.ResponseWriter, r *http.Request) {
	etag := configschema.ETag()
	w.Header().Set("Content-Type", "application/schema+json")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("ETag", etag)
	if r.Header.Get("If-None-Match") == etag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(configschema.Document())
}
