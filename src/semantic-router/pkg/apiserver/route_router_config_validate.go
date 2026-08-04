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
	normalized, err = redactNormalizedConfigYAML(normalized)
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

func redactNormalizedConfigYAML(
	normalized []byte,
) ([]byte, error) {
	var value interface{}
	if err := yaml.Unmarshal(normalized, &value); err != nil {
		return nil, fmt.Errorf("failed to decode normalized config: %w", err)
	}
	// Validation accepts caller-controlled api_key_env names. Never return
	// resolved process credentials, even to principals that can view the active
	// config's secrets.
	redacted := redactSensitiveConfigValue(value)
	result, err := yaml.Marshal(redacted)
	if err != nil {
		return nil, fmt.Errorf("failed to encode redacted config: %w", err)
	}
	return result, nil
}
