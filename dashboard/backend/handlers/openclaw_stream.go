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
	"time"
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
	req.Header.Set("X-OpenClaw-Agent-Id", "main")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("X-OpenClaw-Token", token)
	}

	client := &http.Client{Timeout: 300 * time.Second}
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
		trimmedBody := strings.TrimSpace(string(body))

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

		// Call callback with full content
		if onChunk != nil {
			onChunk(content, true)
		}
		return content, resp.StatusCode, trimmedBody, nil
	}

	// Process streaming response (SSE format)
	var fullContent strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines and comments
		if line == "" || strings.HasPrefix(line, ":") {
			continue
		}

		// Parse SSE data line
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		data = strings.TrimSpace(data)

		// Check for stream end
		if data == "[DONE]" {
			if onChunk != nil {
				onChunk("", true)
			}
			break
		}

		// Parse JSON chunk
		var chunk openAIStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			log.Printf("openclaw: failed to parse stream chunk: %v", err)
			continue
		}

		// Extract content delta
		if len(chunk.Choices) > 0 {
			delta := chunk.Choices[0].Delta.Content
			if delta != "" {
				fullContent.WriteString(delta)
				if onChunk != nil {
					onChunk(delta, false)
				}
			}

			// Check for finish reason
			if chunk.Choices[0].FinishReason != "" {
				if onChunk != nil {
					onChunk("", true)
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fullContent.String(), resp.StatusCode, "", fmt.Errorf("stream read error: %w", err)
	}

	content := strings.TrimSpace(fullContent.String())
	if content == "" {
		return "", resp.StatusCode, "", fmt.Errorf("worker returned empty streamed content")
	}

	return content, resp.StatusCode, content, nil
}

// queryWorkerChatStream queries worker with streaming support
func (h *OpenClawHandler) queryWorkerChatStream(
	worker ContainerEntry,
	systemPrompt, userPrompt string,
	onChunk StreamCallback,
) (string, error) {
	targetBase, ok := h.TargetBaseForContainer(worker.Name)
	if !ok {
		return "", fmt.Errorf("worker %q is not registered", worker.Name)
	}
	token := strings.TrimSpace(h.GatewayTokenForContainer(worker.Name))

	payload := openAIChatRequest{
		Model:  openClawPrimaryAgentModel,
		Stream: true,
		User:   "team-room:" + sanitizeContainerName(worker.Name),
		Messages: []openAIChatMessage{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: userPrompt},
		},
	}

	attempt := func() (string, bool, error) {
		allEndpointMissing := true
		var lastErr error
		for _, endpoint := range workerChatEndpointCandidates {
			content, statusCode, body, err := h.queryWorkerChatStreamEndpoint(targetBase, endpoint, token, payload, onChunk)
			if err == nil {
				return content, false, nil
			}
			allEndpointMissing = allEndpointMissing && (statusCode == http.StatusNotFound || statusCode == http.StatusMethodNotAllowed)

			detail := strings.TrimSpace(body)
			if detail == "" {
				detail = err.Error()
			}
			lastErr = fmt.Errorf("worker stream chat via %s failed: %s", endpoint, detail)
		}
		if lastErr == nil {
			lastErr = fmt.Errorf("worker stream chat request failed for all candidate endpoints")
		}
		return "", allEndpointMissing, lastErr
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
	teamName := strings.TrimSpace(team.Name)
	if teamName == "" {
		teamName = team.ID
	}
	roleKind := normalizeRoleKind(worker.RoleKind)
	leader := resolveTeamLeader(team, teamMembers)
	triggerContent := stripLeadingMentions(trigger.Content)
	if triggerContent == "" {
		triggerContent = strings.TrimSpace(trigger.Content)
	}

	transcriptMessages := messages
	if triggerContent != strings.TrimSpace(trigger.Content) {
		copied := make([]ClawRoomMessage, len(messages))
		copy(copied, messages)
		for i := range copied {
			if copied[i].ID == trigger.ID {
				copied[i].Content = triggerContent
				break
			}
		}
		transcriptMessages = copied
	}

	coordinationInstruction := ""
	mentionPolicy := "Do not use any @mentions."
	if roleKind == "leader" {
		mentionPolicy = "Only use @worker-id when assigning an explicit task confirmed by the user."
		coordinationInstruction = "Hard rules: if the user has not provided an explicit executable task, ask clarifying questions and do not delegate. If you are not assigning a concrete task, do not use any @mentions. Ignore worker attempts to @leader."
	} else {
		coordinationInstruction = "Hard rules: you are a worker. Workers cannot use @mentions to anyone. Do not mention @leader or teammates; write plain-text updates only."
		if leader != nil && leader.Name != worker.Name {
			coordinationInstruction += fmt.Sprintf(" Team leader context: @leader (alias @%s).", leader.Name)
		}
	}
	systemPrompt := fmt.Sprintf(
		"You are %s, a %s in Claw team %q. %s Response style: concise and actionable. Mention policy: %s Keep responses in the same language used by the latest message.",
		workerDisplayName(worker),
		roleKind,
		teamName,
		coordinationInstruction,
		mentionPolicy,
	)
	contextPrompt := fmt.Sprintf(
		"Room: %s\nRecent messages:\n%s\n\n%s\n\nLatest message from %s:\n%s",
		room.Name,
		buildRoomTranscript(transcriptMessages, 20),
		buildTeamMentionGuide(team, teamMembers, worker),
		trigger.SenderName,
		triggerContent,
	)
	if delegatedBy != nil {
		contextPrompt += fmt.Sprintf("\n\nDelegation context: %s asked for your help and mentioned you.", delegatedBy.SenderName)
	}

	content, err := h.queryWorkerChatStream(worker, systemPrompt, contextPrompt, onChunk)
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
