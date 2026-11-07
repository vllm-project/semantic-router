package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"time"
)

const modelCatalogTimeout = 30 * time.Second

// ModelCatalogSource supplies the canonical JSON emitted from the packaged
// model assets. Keeping this seam injectable lets the Dashboard consume the
// Python-owned catalog contract without copying parsing or compatibility
// policy into Go.
type ModelCatalogSource interface {
	Load(context.Context) ([]byte, error)
}

type packagedModelCatalogSource struct {
	cli packagedCLI
}

// NewPackagedModelCatalogSource reads every catalog channel packaged in the
// same runtime image as the Dashboard.
func NewPackagedModelCatalogSource(pythonPath string) ModelCatalogSource {
	return &packagedModelCatalogSource{cli: newPackagedCLI(pythonPath)}
}

func (source *packagedModelCatalogSource) Load(ctx context.Context) ([]byte, error) {
	return source.cli.run(ctx, modelCatalogTimeout, nil, "cli.model_catalog_export")
}

// ModelCatalogHandler exposes read-only built-in catalog metadata. All
// mutation and custom-config workflows remain on their existing endpoints.
func ModelCatalogHandler(source ModelCatalogSource) http.HandlerFunc {
	catalog := &packagedDocument{load: func(ctx context.Context) ([]byte, error) {
		raw, err := source.Load(ctx)
		if err != nil {
			return nil, err
		}
		return normalizeModelCatalogDocument(raw)
	}}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/catalog" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			writeModelCatalogError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET and HEAD are supported.")
			return
		}

		payload, err := catalog.get(r.Context())
		if err != nil {
			status := http.StatusServiceUnavailable
			code := "catalog_unavailable"
			message := "Built-in model catalog could not be loaded from the installed vllm-sr package."
			if errors.Is(err, errInvalidModelCatalogContract) {
				status = http.StatusBadGateway
				code = "catalog_contract_invalid"
				message = "The installed vllm-sr package returned an invalid model catalog contract."
			}
			writeModelCatalogError(w, status, code, message)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "private, max-age=60")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	}
}

func writeModelCatalogError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": code, "message": message})
}
