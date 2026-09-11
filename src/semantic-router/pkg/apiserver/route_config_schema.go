//go:build !windows && cgo

package apiserver

import (
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func (s *ClassificationAPIServer) handleConfigSchema(w http.ResponseWriter, r *http.Request) {
	representation, err := configschema.Render(configschema.ViewOptions{
		View:        r.URL.Query().Get("view"),
		Path:        r.URL.Query().Get("path"),
		SurfaceKind: r.URL.Query().Get("kind"),
		SurfaceName: r.URL.Query().Get("name"),
	})
	if err != nil {
		var viewError *configschema.ViewError
		if errors.As(err, &viewError) {
			http.Error(w, viewError.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, "Config schema is unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", representation.ContentType)
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("ETag", representation.ETag)
	if r.Header.Get("If-None-Match") == representation.ETag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(representation.Body)
}
