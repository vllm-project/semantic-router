//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strings"
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
		s.writeErrorResponse(w, http.StatusBadRequest, "YAML_PARSE_ERROR", err.Error())
		return
	}
	normalized, err := normalizeRouterConfigDocument(doc)
	if err != nil {
		s.writeErrorResponse(w, http.StatusUnprocessableEntity, "CONFIG_VALIDATION_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, RouterConfigValidateResponse{
		Valid:          true,
		NormalizedYAML: string(normalized),
	})
}
