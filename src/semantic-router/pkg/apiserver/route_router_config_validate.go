//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"strings"

	"gopkg.in/yaml.v3"
)

type RouterConfigValidateResponse struct {
	Valid          bool   `json:"valid"`
	NormalizedYAML string `json:"normalized_yaml"`
}

func (s *ClassificationAPIServer) handleConfigValidate(
	w http.ResponseWriter,
	r *http.Request,
) {
	var req RouterConfigUpdateRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return
	}
	doc, err := decodeYAMLDocument([]byte(req.YAML))
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"YAML_PARSE_ERROR",
			scrubSecretsInErrorMessage(err.Error()),
		)
		return
	}
	normalized, err := normalizeRouterConfigDocumentWithoutEnv(doc)
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusUnprocessableEntity,
			"CONFIG_VALIDATION_ERROR",
			scrubSecretsInErrorMessage(err.Error()),
		)
		return
	}
	normalized, err = s.redactNormalizedConfigYAML(r, normalized)
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusInternalServerError,
			"REDACTION_ERROR",
			err.Error(),
		)
		return
	}
	s.writeJSONResponse(w, http.StatusOK, RouterConfigValidateResponse{
		Valid:          true,
		NormalizedYAML: string(normalized),
	})
}

func (s *ClassificationAPIServer) redactNormalizedConfigYAML(
	r *http.Request,
	normalized []byte,
) ([]byte, error) {
	var value interface{}
	if err := yaml.Unmarshal(normalized, &value); err != nil {
		return nil, fmt.Errorf("failed to decode normalized config: %w", err)
	}
	redacted := s.maybeRedactConfigView(r, value)
	result, err := yaml.Marshal(redacted)
	if err != nil {
		return nil, fmt.Errorf("failed to encode redacted config: %w", err)
	}
	return result, nil
}
