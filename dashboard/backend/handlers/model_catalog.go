package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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

var errInvalidModelCatalogContract = errors.New("invalid model catalog contract")

type modelCatalogEnvelope struct {
	Catalogs   []modelCatalogHeader      `json:"catalogs"`
	Models     []modelCatalogModelHeader `json:"models"`
	Configured json.RawMessage           `json:"configured,omitempty"`
}

type modelCatalogHeader struct {
	CatalogVersion string   `json:"catalog_version"`
	Channel        string   `json:"channel"`
	DefaultModel   string   `json:"default_model"`
	EnabledModels  []string `json:"enabled_models"`
}

type modelCatalogModelHeader struct {
	ID                  string                 `json:"id"`
	DisplayName         string                 `json:"display_name"`
	Description         string                 `json:"description"`
	Kind                string                 `json:"kind"`
	Family              string                 `json:"family"`
	Generation          int                    `json:"generation"`
	PolicyVersion       string                 `json:"policy_version"`
	Entrypoint          string                 `json:"entrypoint"`
	Recipe              string                 `json:"recipe"`
	CatalogVersion      string                 `json:"catalog_version"`
	Channel             string                 `json:"channel"`
	Compatible          *bool                  `json:"compatible"`
	CompatibilityReason string                 `json:"compatibility_reason"`
	EnabledByDefault    *bool                  `json:"enabled_by_default"`
	Default             *bool                  `json:"default"`
	Verification        modelVerificationHead  `json:"verification"`
	Protocols           []string               `json:"protocols"`
	Traits              []string               `json:"traits"`
	Roles               []modelCatalogRoleHead `json:"roles"`
}

type modelVerificationHead struct {
	Status      string `json:"status"`
	Authority   string `json:"authority"`
	AssetSHA256 string `json:"asset_sha256"`
}

type modelCatalogRoleHead struct {
	Name              string   `json:"name"`
	Required          *bool    `json:"required"`
	MinimumCandidates int      `json:"minimum_candidates"`
	Traits            []string `json:"traits"`
	RecommendedPool   []string `json:"recommended_pool"`
}

func normalizeModelCatalogDocument(raw []byte) ([]byte, error) {
	var envelope modelCatalogEnvelope
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&envelope); err != nil {
		return nil, fmt.Errorf("%w: invalid JSON", errInvalidModelCatalogContract)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("%w: trailing JSON", errInvalidModelCatalogContract)
	}
	if len(envelope.Catalogs) == 0 || len(envelope.Models) == 0 {
		return nil, fmt.Errorf("%w: empty catalog inventory", errInvalidModelCatalogContract)
	}
	for _, catalog := range envelope.Catalogs {
		if catalog.CatalogVersion == "" ||
			(catalog.Channel != "latest" && catalog.Channel != "release") ||
			catalog.DefaultModel == "" ||
			len(catalog.EnabledModels) == 0 {
			return nil, fmt.Errorf("%w: malformed catalog version", errInvalidModelCatalogContract)
		}
	}
	for _, model := range envelope.Models {
		if model.ID == "" ||
			model.DisplayName == "" ||
			model.Description == "" ||
			model.Kind != "virtual" ||
			model.Family == "" ||
			model.Generation < 1 ||
			model.PolicyVersion == "" ||
			model.Entrypoint == "" ||
			model.Recipe == "" ||
			model.CatalogVersion == "" ||
			(model.Channel != "latest" && model.Channel != "release") ||
			model.Compatible == nil ||
			model.CompatibilityReason == "" ||
			model.EnabledByDefault == nil ||
			model.Default == nil ||
			model.Verification.Status != "verified" ||
			model.Verification.Authority == "" ||
			!validModelCatalogDigest(model.Verification.AssetSHA256) ||
			len(model.Protocols) == 0 ||
			len(model.Traits) == 0 ||
			!validModelCatalogRoles(model.Roles) {
			return nil, fmt.Errorf("%w: malformed model metadata", errInvalidModelCatalogContract)
		}
	}
	// `configured` is interactive local-config state owned by the CLI. Never
	// expose it through Dashboard; marshal only the validated catalog DTOs.
	envelope.Configured = nil
	return json.Marshal(envelope)
}

func validModelCatalogDigest(value string) bool {
	if len(value) != len("sha256:")+64 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	for _, character := range value[len("sha256:"):] {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

func validModelCatalogRoles(roles []modelCatalogRoleHead) bool {
	if len(roles) == 0 {
		return false
	}
	for _, role := range roles {
		if role.Name == "" || role.Required == nil || role.MinimumCandidates < 1 || len(role.Traits) == 0 || len(role.RecommendedPool) == 0 {
			return false
		}
	}
	return true
}

func writeModelCatalogError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": code, "message": message})
}
