//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RouterConfigValidateRequest is the JSON body for POST /api/v1/config/validate.
type RouterConfigValidateRequest struct {
	YAML            string `json:"yaml"`
	CompareToActive bool   `json:"compare_to_active,omitempty"`
}

type RouterConfigValidateResponse struct {
	Valid           bool                `json:"valid"`
	ContractVersion string              `json:"contract_version"`
	NormalizedYAML  string              `json:"normalized_yaml"`
	Errors          []config.Diagnostic `json:"errors"`
	Warnings        []config.Diagnostic `json:"warnings"`
	Diff            *config.ConfigDiff  `json:"diff,omitempty"`
}

func (s *ClassificationAPIServer) handleConfigValidate(
	w http.ResponseWriter,
	r *http.Request,
) {
	var req RouterConfigValidateRequest
	if err := s.parseStrictJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return
	}

	opts := config.EvaluateOptions{CompareToActive: req.CompareToActive}
	if req.CompareToActive {
		opts.ActiveYAML = s.activeConfigSnapshotYAML()
	}
	result := config.Evaluate([]byte(req.YAML), opts)
	s.writeJSONResponse(w, http.StatusOK, RouterConfigValidateResponse{
		Valid:           result.Valid,
		ContractVersion: result.ContractVersion,
		NormalizedYAML:  result.NormalizedYAML,
		Errors:          scrubValidateDiagnostics(result.Errors),
		Warnings:        scrubValidateDiagnostics(result.Warnings),
		Diff:            result.Diff,
	})
}

func (s *ClassificationAPIServer) activeConfigSnapshotYAML() []byte {
	if s == nil || s.configPath == "" {
		return nil
	}
	paths := resolveConfigPersistencePaths(s.configPath)
	data, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		return nil
	}
	return data
}

func scrubValidateDiagnostics(diagnostics []config.Diagnostic) []config.Diagnostic {
	if diagnostics == nil {
		return []config.Diagnostic{}
	}
	out := make([]config.Diagnostic, len(diagnostics))
	for i, diagnostic := range diagnostics {
		diagnostic.Message = scrubSecretsInErrorMessage(diagnostic.Message)
		out[i] = diagnostic
	}
	return out
}
