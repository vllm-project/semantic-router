package handlers

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

// StreamChunkMessage represents a chunk update message for streaming
type StreamChunkMessage struct {
	Type      string `json:"type"`
	RoomID    string `json:"roomId"`
	MessageID string `json:"messageId"`
	Chunk     string `json:"chunk"`
	Done      bool   `json:"done"`
	Timestamp string `json:"timestamp"`
}

// openAIStreamChoice represents a streaming choice in OpenAI format
type openAIStreamChoice struct {
	Delta struct {
		Content string `json:"content"`
	} `json:"delta"`
	FinishReason string `json:"finish_reason"`
}

// openAIStreamResponse represents a streaming response chunk
type openAIStreamResponse struct {
	Choices []openAIStreamChoice `json:"choices"`
}

// StreamCallback is called for each chunk of streamed content
type StreamCallback func(chunk string, done bool)

func parseWorkerChatFallbackResponse(
	body []byte,
	statusCode int,
	onChunk StreamCallback,
) (string, int, string, error) {
	trimmedBody := strings.TrimSpace(string(body))

	var parsed openAIChatResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return "", statusCode, trimmedBody, fmt.Errorf("invalid worker chat response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		return "", statusCode, trimmedBody, fmt.Errorf("%s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		return "", statusCode, trimmedBody, fmt.Errorf("worker returned no choices")
	}
	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if content == "" {
		return "", statusCode, trimmedBody, fmt.Errorf("worker returned empty content")
	}
	if onChunk != nil {
		onChunk(content, true)
	}
	return content, statusCode, trimmedBody, nil
}

func readWorkerChatStream(
	body io.Reader,
	onChunk StreamCallback,
) (string, error) {
	var fullContent strings.Builder
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, ":") || !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
		if data == "[DONE]" {
			if onChunk != nil {
				onChunk("", true)
			}
			break
		}

		var chunk openAIStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("openclaw: failed to parse stream chunk: %v", err)
			continue
		}
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta.Content
		if delta != "" {
			fullContent.WriteString(delta)
			if onChunk != nil {
				onChunk(delta, false)
			}
		}
		if chunk.Choices[0].FinishReason != "" && onChunk != nil {
			onChunk("", true)
		}
	}

	if err := scanner.Err(); err != nil {
		return fullContent.String(), fmt.Errorf("stream read error: %w", err)
	}

	content := strings.TrimSpace(fullContent.String())
	if content == "" {
		return "", fmt.Errorf("worker returned empty streamed content")
	}
	return content, nil
}

// queryWorkerChatStreamEndpoint makes a streaming request to worker chat endpoint
func (h *OpenClawHandler) queryWorkerChatStreamEndpoint(
	targetBase string,
	endpoint string,
	token string,
	payload openAIChatRequest,
	onChunk StreamCallback,
) (string, int, string, error) {
	// Set stream to true
	payload.Stream = true

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
	req.Header.Set("Accept", "text/event-stream")
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
	defer resp.Body.Close()

	// Check for non-streaming error response
	contentType := resp.Header.Get("Content-Type")
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		body, _ := io.ReadAll(resp.Body)
		trimmedBody := strings.TrimSpace(string(body))
		if trimmedBody == "" {
			trimmedBody = resp.Status
		}
		return "", resp.StatusCode, trimmedBody, fmt.Errorf("worker chat stream request failed: %s", trimmedBody)
	}

	// If response is not streaming, fall back to non-streaming handling
	if !strings.Contains(contentType, "text/event-stream") {
		body, _ := io.ReadAll(resp.Body)
		return parseWorkerChatFallbackResponse(body, resp.StatusCode, onChunk)
	}

	content, err := readWorkerChatStream(resp.Body, onChunk)
	if err != nil {
		return content, resp.StatusCode, "", err
	}

	return content, resp.StatusCode, content, nil
}

func (h *OpenClawHandler) queryWorkerChatStreamWithMessages(
	worker ContainerEntry,
	sessionUser string,
	messages []openAIChatMessage,
	onChunk StreamCallback,
) (string, error) {
	targetBase, ok := h.TargetBaseForContainer(worker.Name)
	if !ok {
		return "", fmt.Errorf("worker %q is not registered", worker.Name)
	}
	token := strings.TrimSpace(h.GatewayTokenForContainer(worker.Name))

	payload := buildWorkerChatRequest(messages, sessionUser, true)

	attempt := func() (string, bool, error) {
		failures := make([]workerChatAttemptFailure, 0, len(workerChatEndpointCandidates))
		for _, endpoint := range workerChatEndpointCandidates {
			content, statusCode, body, err := h.queryWorkerChatStreamEndpoint(targetBase, endpoint, token, payload, onChunk)
			if err == nil {
				return content, false, nil
			}
			failures = append(failures, buildWorkerChatAttemptFailure(endpoint, statusCode, body, err))
		}
		return "", workerChatAllEndpointsMissing(failures), formatWorkerChatAttemptError("worker stream chat", failures)
	}

	content, allEndpointMissing, err := attempt()
	if err == nil {
		return content, nil
	}
	if !allEndpointMissing {
		return "", err
	}

	// Try to recover endpoint
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

// runWorkerReplyStream runs worker reply with streaming support
func (h *OpenClawHandler) runWorkerReplyStream(
	room ClawRoomEntry,
	team TeamEntry,
	teamMembers []ContainerEntry,
	worker ContainerEntry,
	messages []ClawRoomMessage,
	trigger ClawRoomMessage,
	delegatedBy *ClawRoomMessage,
	onChunk StreamCallback,
) (ClawRoomMessage, error) {
	content, err := h.queryWorkerChatStreamWithMessages(
		worker,
		roomScopedSessionUser(room, worker),
		buildRoomChatMessages(room, team, teamMembers, worker, messages, trigger, delegatedBy),
		onChunk,
	)
	if err != nil {
		return ClawRoomMessage{}, err
	}

	senderType := normalizeRoleKind(worker.RoleKind)
	if senderType != "leader" {
		senderType = "worker"
	}

	metadata := map[string]string{}
	if delegatedBy != nil {
		metadata["delegatedBy"] = delegatedBy.SenderID
	}

	return newRoomMessage(room, senderType, worker.Name, workerDisplayName(worker), content, metadata), nil
}
