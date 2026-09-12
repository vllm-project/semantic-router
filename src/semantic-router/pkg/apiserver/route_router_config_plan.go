//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"net/http"
	"reflect"
	"strings"
)

type routerConfigPlanRequest struct {
	YAML string                   `json:"yaml"`
	Mode routerConfigMutationMode `json:"mode"`
}

type routerConfigPlanResponse struct {
	Valid         bool                     `json:"valid"`
	Mode          routerConfigMutationMode `json:"mode"`
	Changed       bool                     `json:"changed"`
	CurrentETag   string                   `json:"current_etag,omitempty"`
	CandidateETag string                   `json:"candidate_etag"`
}

// handleConfigPlan performs the exact merge/replace and hot-reload checks used
// by mutation endpoints, but does not write a file or reload the Router.
func (s *ClassificationAPIServer) handleConfigPlan(
	w http.ResponseWriter,
	r *http.Request,
) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	var req routerConfigPlanRequest
	if err := s.parseStrictJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	if strings.TrimSpace(req.YAML) == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "YAML content is required")
		return
	}
	if req.Mode != routerConfigMutationMerge && req.Mode != routerConfigMutationReplace {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_MODE", "mode must be merge or replace")
		return
	}

	doc, err := decodeYAMLDocument([]byte(req.YAML))
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "YAML_PARSE_ERROR", scrubSecretsInErrorMessage(err.Error()))
		return
	}
	paths := resolveConfigPersistencePaths(s.configPath)
	candidate, current, ok := s.prepareRouterConfigMutationPayload(w, doc, paths.sourcePath, req.Mode)
	if !ok {
		return
	}

	s.writeJSONResponse(w, http.StatusOK, routerConfigPlanResponse{
		Valid:         true,
		Mode:          req.Mode,
		Changed:       !equivalentConfigDocuments(current, candidate),
		CurrentETag:   configDocumentETag(current),
		CandidateETag: configDocumentETag(candidate),
	})
}

func equivalentConfigDocuments(current, candidate []byte) bool {
	if bytes.Equal(current, candidate) {
		return true
	}
	currentDocument, currentErr := decodeYAMLDocument(current)
	if currentErr != nil {
		return false
	}
	candidateDocument, candidateErr := decodeYAMLDocument(candidate)
	if candidateErr != nil {
		return false
	}
	return reflect.DeepEqual(currentDocument, candidateDocument)
}
