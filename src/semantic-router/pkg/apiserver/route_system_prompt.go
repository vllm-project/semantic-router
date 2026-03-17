//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type SystemPromptInfo struct {
	Category string `json:"category"`
	Prompt   string `json:"prompt"`
	Enabled  bool   `json:"enabled"`
	Mode     string `json:"mode"` // "replace" or "insert"
}

// SystemPromptsResponse represents the response for GET /config/system-prompts
type SystemPromptsResponse struct {
	SystemPrompts []SystemPromptInfo `json:"system_prompts"`
}

// SystemPromptUpdateRequest represents a request to update system prompt settings
type SystemPromptUpdateRequest struct {
	Category string `json:"category,omitempty"` // If empty, applies to all categories
	Enabled  *bool  `json:"enabled,omitempty"`  // true to enable, false to disable
	Mode     string `json:"mode,omitempty"`     // "replace" or "insert"
}

type systemPromptUpdateError struct {
	statusCode int
	message    string
}

func (e *systemPromptUpdateError) Error() string {
	return e.message
}

// handleGetSystemPrompts handles GET /config/system-prompts
func (s *ClassificationAPIServer) handleGetSystemPrompts(w http.ResponseWriter, _ *http.Request) {
	cfg := s.currentConfig()
	if cfg == nil {
		http.Error(w, "Configuration not available", http.StatusInternalServerError)
		return
	}

	s.writeSystemPromptsResponse(w, cfg)
}

// handleUpdateSystemPrompts handles PUT /config/system-prompts
func (s *ClassificationAPIServer) handleUpdateSystemPrompts(w http.ResponseWriter, r *http.Request) {
	var req SystemPromptUpdateRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if req.Enabled == nil && req.Mode == "" {
		http.Error(w, "either enabled or mode field is required", http.StatusBadRequest)
		return
	}

	// Validate mode if provided
	if req.Mode != "" && req.Mode != "replace" && req.Mode != "insert" {
		http.Error(w, "mode must be either 'replace' or 'insert'", http.StatusBadRequest)
		return
	}

	cfg := s.currentConfig()
	if cfg == nil {
		http.Error(w, "Configuration not available", http.StatusInternalServerError)
		return
	}

	newCfg := cloneRouterConfigWithDecisions(cfg)
	if err := applySystemPromptUpdates(newCfg, req); err != nil {
		http.Error(w, err.message, err.statusCode)
		return
	}

	s.updateRuntimeConfig(newCfg)
	s.writeSystemPromptsResponse(w, newCfg)
}

func cloneRouterConfigWithDecisions(cfg *config.RouterConfig) *config.RouterConfig {
	newCfg := *cfg
	newDecisions := make([]config.Decision, len(cfg.Decisions))
	copy(newDecisions, cfg.Decisions)
	newCfg.Decisions = newDecisions
	return &newCfg
}

func applySystemPromptUpdates(cfg *config.RouterConfig, req SystemPromptUpdateRequest) *systemPromptUpdateError {
	if req.Category == "" {
		return applySystemPromptUpdatesToAll(cfg.Decisions, req)
	}
	return applySystemPromptUpdateToNamedDecision(cfg.Decisions, req)
}

func applySystemPromptUpdatesToAll(
	decisions []config.Decision,
	req SystemPromptUpdateRequest,
) *systemPromptUpdateError {
	updated := false
	for i := range decisions {
		if !hasConfiguredSystemPrompt(decisions[i]) {
			continue
		}
		if applySystemPromptUpdate(&decisions[i], req) {
			updated = true
		}
	}
	if updated {
		return nil
	}
	return &systemPromptUpdateError{
		statusCode: http.StatusBadRequest,
		message:    "No decisions with system prompts found to update",
	}
}

func applySystemPromptUpdateToNamedDecision(
	decisions []config.Decision,
	req SystemPromptUpdateRequest,
) *systemPromptUpdateError {
	for i := range decisions {
		if decisions[i].Name != req.Category {
			continue
		}
		if !hasConfiguredSystemPrompt(decisions[i]) {
			return &systemPromptUpdateError{
				statusCode: http.StatusBadRequest,
				message:    fmt.Sprintf("Decision '%s' has no system prompt configured", req.Category),
			}
		}
		if applySystemPromptUpdate(&decisions[i], req) {
			return nil
		}
		return &systemPromptUpdateError{
			statusCode: http.StatusBadRequest,
			message:    fmt.Sprintf("Decision '%s' has no system prompt plugin configured", req.Category),
		}
	}
	return &systemPromptUpdateError{
		statusCode: http.StatusNotFound,
		message:    fmt.Sprintf("Decision '%s' not found", req.Category),
	}
}

func hasConfiguredSystemPrompt(decision config.Decision) bool {
	systemPromptConfig := decision.GetSystemPromptConfig()
	return systemPromptConfig != nil && systemPromptConfig.SystemPrompt != ""
}

func applySystemPromptUpdate(decision *config.Decision, req SystemPromptUpdateRequest) bool {
	for i := range decision.Plugins {
		if decision.Plugins[i].Type != "system_prompt" {
			continue
		}
		configMap := systemPromptPluginConfigMap(decision.Plugins[i].Configuration)
		if req.Enabled != nil {
			configMap["enabled"] = *req.Enabled
		}
		if req.Mode != "" {
			configMap["mode"] = req.Mode
		}
		payload, err := config.NewStructuredPayload(configMap)
		if err != nil {
			return false
		}
		decision.Plugins[i].Configuration = payload
		return true
	}
	return false
}

func systemPromptPluginConfigMap(rawConfig *config.StructuredPayload) map[string]interface{} {
	if rawConfig != nil {
		if configMap, err := rawConfig.AsStringMap(); err == nil {
			return configMap
		}
	}
	return make(map[string]interface{})
}

func buildSystemPromptsResponse(cfg *config.RouterConfig) SystemPromptsResponse {
	systemPrompts := make([]SystemPromptInfo, 0, len(cfg.Decisions))
	for _, decision := range cfg.Decisions {
		systemPromptConfig := decision.GetSystemPromptConfig()
		prompt := ""
		if systemPromptConfig != nil {
			prompt = systemPromptConfig.SystemPrompt
		}
		systemPrompts = append(systemPrompts, SystemPromptInfo{
			Category: decision.Name,
			Prompt:   prompt,
			Enabled:  decision.IsSystemPromptEnabled(),
			Mode:     decision.GetSystemPromptMode(),
		})
	}
	return SystemPromptsResponse{SystemPrompts: systemPrompts}
}

func (s *ClassificationAPIServer) writeSystemPromptsResponse(
	w http.ResponseWriter,
	cfg *config.RouterConfig,
) {
	response := buildSystemPromptsResponse(cfg)
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}
