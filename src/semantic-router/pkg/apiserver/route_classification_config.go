//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

func (s *ClassificationAPIServer) handleGetConfig(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	doc, err := routingSurfaceDocument(cfg)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_ENCODE_ERROR", err.Error())
		return
	}

	s.writeJSONResponse(w, http.StatusOK, doc)
}

func (s *ClassificationAPIServer) handleUpdateConfig(w http.ResponseWriter, r *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_UNAVAILABLE", "Classification config not available")
		return
	}

	var payload map[string]interface{}
	if err := s.parseJSONRequest(r, &payload); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}
	if len(payload) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "request body cannot be empty")
		return
	}

	fullDoc, err := fullConfigDocument(cfg)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_ENCODE_ERROR", err.Error())
		return
	}

	merged := configDeepMerge(fullDoc, normalizeClassificationConfigPayload(payload))
	mergedYAML, err := yaml.Marshal(merged)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_ENCODE_ERROR", fmt.Sprintf("failed to encode merged config: %v", err))
		return
	}

	newCfg, err := config.ParseYAMLBytes(mergedYAML)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "CONFIG_PARSE_ERROR", err.Error())
		return
	}

	s.updateRuntimeConfig(newCfg)

	doc, err := routingSurfaceDocument(newCfg)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "CONFIG_ENCODE_ERROR", err.Error())
		return
	}
	s.writeJSONResponse(w, http.StatusOK, doc)
}

func routingSurfaceDocument(cfg *config.RouterConfig) (map[string]interface{}, error) {
	yamlBytes, err := dsl.EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to emit routing config: %w", err)
	}
	return decodeYAMLDocument(yamlBytes)
}

func fullConfigDocument(cfg *config.RouterConfig) (map[string]interface{}, error) {
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	yamlBytes, err := yaml.Marshal(canonical)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal current config: %w", err)
	}
	return decodeYAMLDocument(yamlBytes)
}

func decodeYAMLDocument(data []byte) (map[string]interface{}, error) {
	var doc map[string]interface{}
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("failed to decode YAML document: %w", err)
	}
	return doc, nil
}

func normalizeClassificationConfigPayload(payload map[string]interface{}) map[string]interface{} {
	if _, ok := payload["routing"]; ok {
		return payload
	}
	return map[string]interface{}{
		"routing": payload,
	}
}
