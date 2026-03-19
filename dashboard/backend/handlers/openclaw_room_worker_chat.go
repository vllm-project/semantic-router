package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type openAIChatRequest struct {
	Model    string              `json:"model"`
	Messages []openAIChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	User     string              `json:"user,omitempty"`
}

type openAIChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

const (
	openClawPrimaryAgentID    = "vllm-sr"
	openClawPrimaryAgentModel = "openclaw:main"
)

func nestedObject(parent map[string]any, key string) map[string]any {
	if existing, ok := parent[key].(map[string]any); ok {
		return existing
	}
	created := map[string]any{}
	parent[key] = created
	return created
}

func enableGatewayEndpoint(config map[string]any, endpointKey string) bool {
	gateway := nestedObject(config, "gateway")
	httpCfg := nestedObject(gateway, "http")
	endpoints := nestedObject(httpCfg, "endpoints")
	endpoint := nestedObject(endpoints, endpointKey)
	if enabled, ok := endpoint["enabled"].(bool); ok && enabled {
		return false
	}
	endpoint["enabled"] = true
	return true
}

func (h *OpenClawHandler) workerConfigPath(worker ContainerEntry) string {
	if dataDir := strings.TrimSpace(worker.DataDir); dataDir != "" {
		return filepath.Join(dataDir, "openclaw.json")
	}
	return filepath.Join(h.containerDataDir(sanitizeContainerName(worker.Name)), "openclaw.json")
}

func (h *OpenClawHandler) ensureWorkerChatEndpoint(worker ContainerEntry) (bool, error) {
	if !h.canRepairWorkerChatEndpoint() {
		return false, nil
	}

	configPath := h.workerConfigPath(worker)
	data, err := os.ReadFile(configPath)
	if err != nil {
		return false, fmt.Errorf("unable to read worker config %q: %w", configPath, err)
	}

	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		return false, fmt.Errorf("invalid worker config %q: %w", configPath, err)
	}

	changed := false
	changed = enableGatewayEndpoint(cfg, "chatCompletions") || changed
	changed = enableGatewayEndpoint(cfg, "responses") || changed
	if changed {
		updated, err := json.MarshalIndent(cfg, "", "  ")
		if err != nil {
			return false, fmt.Errorf("failed to marshal worker config update: %w", err)
		}
		if err := os.WriteFile(configPath, updated, 0o644); err != nil {
			return false, fmt.Errorf("failed to persist worker config update: %w", err)
		}
	}

	if _, err := h.containerCombinedOutput("restart", worker.Name); err != nil {
		if changed {
			return false, fmt.Errorf("worker restart failed after endpoint update: %w", err)
		}
		return false, fmt.Errorf("worker restart failed during endpoint recovery: %w", err)
	}

	deadline := time.Now().Add(openClawWorkerEndpointRecoveryTimeout)
	for time.Now().Before(deadline) {
		if h.gatewayReachable(worker.Name, worker.Port) {
			return true, nil
		}
		time.Sleep(openClawWorkerEndpointRecoveryPollPeriod)
	}
	return true, nil
}

func (h *OpenClawHandler) queryWorkerChatEndpoint(
	targetBase string,
	endpoint string,
	token string,
	payload openAIChatRequest,
) (string, int, string, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return "", 0, "", err
	}

	url := strings.TrimRight(targetBase, "/") + endpoint
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(raw))
	if err != nil {
		return "", 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-OpenClaw-Agent-Id", openClawPrimaryAgentID)
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("X-OpenClaw-Token", token)
	}

	client := newOpenClawWorkerChatHTTPClient()
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, "", err
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)
	trimmedBody := strings.TrimSpace(string(body))

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		if trimmedBody == "" {
			trimmedBody = resp.Status
		}
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker chat request failed: %s", trimmedBody)
	}

	var parsed openAIChatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("invalid worker chat response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("%s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker returned no choices")
	}
	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if content == "" {
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker returned empty content")
	}
	return content, resp.StatusCode, trimmedBody, nil
}

func (h *OpenClawHandler) queryWorkerChat(worker ContainerEntry, systemPrompt, userPrompt string) (string, error) {
	return h.queryWorkerChatWithMessages(
		worker,
		"team-room:"+sanitizeContainerName(worker.Name),
		[]openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	)
}

func (h *OpenClawHandler) queryWorkerChatWithMessages(
	worker ContainerEntry,
	sessionUser string,
	messages []openAIChatMessage,
) (string, error) {
	targetBase, ok := h.TargetBaseForContainer(worker.Name)
	if !ok {
		return "", fmt.Errorf("worker %q is not registered", worker.Name)
	}
	token := strings.TrimSpace(h.GatewayTokenForContainer(worker.Name))

	payload := buildWorkerChatRequest(messages, sessionUser, false)

	attempt := func() (string, bool, error) {
		failures := make([]workerChatAttemptFailure, 0, len(workerChatEndpointCandidates))
		for _, endpoint := range workerChatEndpointCandidates {
			content, statusCode, body, err := h.queryWorkerChatEndpoint(targetBase, endpoint, token, payload)
			if err == nil {
				return content, false, nil
			}
			failures = append(failures, buildWorkerChatAttemptFailure(endpoint, statusCode, body, err))
		}
		return "", workerChatAllEndpointsMissing(failures), formatWorkerChatAttemptError("worker chat", failures)
	}

	content, allEndpointMissing, err := attempt()
	if err == nil {
		return content, nil
	}
	if !allEndpointMissing {
		return "", err
	}

	recovered, ensureErr := h.ensureWorkerChatEndpoint(worker)
	if ensureErr != nil {
		return "", fmt.Errorf("%w; automatic endpoint repair failed: %w", err, ensureErr)
	}
	if !recovered {
		return "", fmt.Errorf(
			"%w; worker endpoint recovery skipped (read-only mode). ensure gateway.http.endpoints.chatCompletions.enabled=true in %s",
			err,
			h.workerConfigPath(worker),
		)
	}

	content, _, retryErr := attempt()
	if retryErr != nil {
		return "", fmt.Errorf("%w; retry after endpoint repair failed: %w", err, retryErr)
	}
	return content, nil
}

func workerDisplayName(worker ContainerEntry) string {
	if name := strings.TrimSpace(worker.AgentName); name != "" {
		return name
	}
	return worker.Name
}
