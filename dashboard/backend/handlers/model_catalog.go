package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

const (
	modelCatalogTimeout        = 10 * time.Second
	maxModelCatalogOutputBytes = 4 << 20
)

var errModelCatalogOutputTooLarge = errors.New("model catalog output exceeded the size limit")

// ModelCatalogSource supplies the canonical JSON emitted from the packaged
// model assets. Keeping this seam injectable lets the Dashboard consume the
// Python-owned catalog contract without copying parsing or compatibility
// policy into Go.
type ModelCatalogSource interface {
	Load(context.Context) ([]byte, error)
}

type packagedModelCatalogSource struct {
	pythonPath string
}

// NewPackagedModelCatalogSource reads every catalog channel packaged in the
// same runtime image as the Dashboard.
func NewPackagedModelCatalogSource(pythonPath string) ModelCatalogSource {
	pythonPath = strings.TrimSpace(pythonPath)
	if pythonPath == "" {
		pythonPath = "python3"
	}
	return &packagedModelCatalogSource{pythonPath: pythonPath}
}

func (source *packagedModelCatalogSource) Load(ctx context.Context) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, modelCatalogTimeout)
	defer cancel()
	workingDirectory, err := os.MkdirTemp("", "vllm-sr-model-catalog-")
	if err != nil {
		return nil, errors.New("model catalog working directory is unavailable")
	}
	defer os.RemoveAll(workingDirectory)

	command := exec.CommandContext( //nolint:gosec // The Python path is configured at Dashboard startup, never from an HTTP request.
		ctx,
		source.pythonPath,
		"-m",
		"cli.model_catalog_export",
	)
	// Execute in an empty directory so the catalog endpoint cannot be
	// contaminated by a process working directory containing config.yaml.
	command.Dir = workingDirectory
	command.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")
	stdout := &boundedCatalogBuffer{limit: maxModelCatalogOutputBytes}
	stderr := &boundedCatalogBuffer{limit: maxModelCatalogOutputBytes}
	command.Stdout = stdout
	command.Stderr = stderr
	if err := command.Run(); err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return nil, errors.New("model catalog command timed out")
		}
		return nil, fmt.Errorf("model catalog command failed: %w", err)
	}
	return stdout.Bytes(), nil
}

type boundedCatalogBuffer struct {
	bytes.Buffer
	limit int
}

func (buffer *boundedCatalogBuffer) Write(value []byte) (int, error) {
	if len(value) > buffer.limit-buffer.Len() {
		return 0, errModelCatalogOutputTooLarge
	}
	return buffer.Buffer.Write(value)
}

type modelCatalogService struct {
	source ModelCatalogSource
	mu     sync.RWMutex
	cache  []byte
}

func newModelCatalogService(source ModelCatalogSource) *modelCatalogService {
	return &modelCatalogService{source: source}
}

func (service *modelCatalogService) Catalog(ctx context.Context) ([]byte, error) {
	service.mu.RLock()
	if len(service.cache) > 0 {
		cached := append([]byte(nil), service.cache...)
		service.mu.RUnlock()
		return cached, nil
	}
	service.mu.RUnlock()

	raw, err := service.source.Load(ctx)
	if err != nil {
		return nil, err
	}
	normalized, err := normalizeModelCatalogDocument(raw)
	if err != nil {
		return nil, err
	}

	service.mu.Lock()
	if len(service.cache) == 0 {
		service.cache = append([]byte(nil), normalized...)
	}
	cached := append([]byte(nil), service.cache...)
	service.mu.Unlock()
	return cached, nil
}

// ModelCatalogHandler exposes read-only built-in catalog metadata. All
// mutation and custom-config workflows remain on their existing endpoints.
func ModelCatalogHandler(source ModelCatalogSource) http.HandlerFunc {
	service := newModelCatalogService(source)
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/catalog" {
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			writeModelCatalogError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only GET is supported.")
			return
		}

		payload, err := service.Catalog(r.Context())
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
