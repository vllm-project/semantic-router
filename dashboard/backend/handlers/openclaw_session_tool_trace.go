package handlers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

const (
	openClawSessionStateRoot             = "/state/agents"
	openClawSessionPollInterval          = 400 * time.Millisecond
	openClawSessionPollTimeout           = 45 * time.Second
	openClawSessionToolTraceMaxLineBytes = 1024 * 1024
	roomMessageToolTraceMetadataKey      = "toolTrace"
)

type openClawSessionToolStep struct {
	ID        string `json:"id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments,omitempty"`
	Status    string `json:"status"`
	Result    string `json:"result,omitempty"`
	Error     string `json:"error,omitempty"`
}

type openClawSessionToolTracePayload struct {
	Revision int                       `json:"revision"`
	Steps    []openClawSessionToolStep `json:"steps"`
}

type openClawSessionIndexEntry struct {
	SessionFile string `json:"sessionFile"`
	SessionID   string `json:"sessionId"`
}

func openClawChatCompletionsSessionKeys(sessionUser string) []string {
	trimmedUser := strings.TrimSpace(sessionUser)
	keys := make([]string, 0, 2)
	if trimmedUser != "" {
		keys = append(keys,
			fmt.Sprintf("agent:%s:openai-user:%s", openClawPrimaryAgentID, trimmedUser),
			fmt.Sprintf("agent:%s:openresponses-user:%s", openClawPrimaryAgentID, trimmedUser),
		)
	}
	return keys
}

func openClawSessionsIndexPath() string {
	return fmt.Sprintf("%s/%s/sessions/sessions.json", openClawSessionStateRoot, openClawPrimaryAgentID)
}

func (h *OpenClawHandler) readOpenClawContainerFile(containerName, filePath string) ([]byte, error) {
	normalizedContainer := sanitizeContainerName(containerName)
	if normalizedContainer == "" {
		return nil, fmt.Errorf("container name is required")
	}
	trimmedPath := strings.TrimSpace(filePath)
	if trimmedPath == "" || strings.Contains(trimmedPath, "..") {
		return nil, fmt.Errorf("invalid container file path")
	}

	output, err := h.containerCombinedOutput(
		"exec",
		normalizedContainer,
		"sh",
		"-c",
		fmt.Sprintf("cat %s 2>/dev/null", shellQuote(trimmedPath)),
	)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'"'"'`) + "'"
}

func resolveOpenClawSessionFileFromIndex(indexData []byte, sessionKeys []string) (string, bool) {
	var index map[string]openClawSessionIndexEntry
	if err := json.Unmarshal(indexData, &index); err != nil {
		return "", false
	}
	for _, key := range sessionKeys {
		entry, ok := index[key]
		if !ok {
			continue
		}
		sessionFile := strings.TrimSpace(entry.SessionFile)
		if sessionFile != "" {
			return sessionFile, true
		}
	}
	return "", false
}

func openClawSessionToolTraceInitialOffset(baselineOffset, fileSize int64) int64 {
	if baselineOffset <= 0 {
		return 0
	}
	if baselineOffset > fileSize {
		return 0
	}
	return baselineOffset
}

func encodeRoomMessageToolTraceMetadata(steps []openClawSessionToolStep) (string, error) {
	if len(steps) == 0 {
		return "", nil
	}
	encoded, err := json.Marshal(steps)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func attachRoomMessageToolTraceMetadata(message *ClawRoomMessage, steps []openClawSessionToolStep) {
	if message == nil || len(steps) == 0 {
		return
	}
	encoded, err := encodeRoomMessageToolTraceMetadata(steps)
	if err != nil {
		log.Printf("openclaw: failed to encode tool trace metadata for message %s: %v", message.ID, err)
		return
	}
	if encoded == "" {
		return
	}
	if message.Metadata == nil {
		message.Metadata = map[string]string{}
	}
	message.Metadata[roomMessageToolTraceMetadataKey] = encoded
}

func splitOpenClawSessionToolTraceLines(data []byte) ([]string, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	scanner.Buffer(make([]byte, 64*1024), openClawSessionToolTraceMaxLineBytes)
	lines := make([]string, 0)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return lines, err
	}
	return lines, nil
}

func (h *OpenClawHandler) readOpenClawSessionToolTraceSteps(
	room ClawRoomEntry,
	worker ContainerEntry,
	sessionFile string,
	baselineOffset int64,
) []openClawSessionToolStep {
	sessionFile = strings.TrimSpace(sessionFile)
	if sessionFile == "" {
		sessionUser := roomScopedSessionUser(room, worker)
		sessionKeys := openClawChatCompletionsSessionKeys(sessionUser)
		resolved, err := h.resolveOpenClawSessionFile(worker.Name, sessionKeys)
		if err != nil {
			return nil
		}
		sessionFile = resolved
	}

	data, err := h.readOpenClawContainerFile(worker.Name, sessionFile)
	if err != nil {
		return nil
	}

	offset := openClawSessionToolTraceInitialOffset(baselineOffset, int64(len(data)))
	if offset >= int64(len(data)) {
		return nil
	}

	lines, err := splitOpenClawSessionToolTraceLines(data[offset:])
	if err != nil {
		log.Printf("openclaw: failed to parse session tool trace lines in %s: %v", sessionFile, err)
	}

	steps := make(map[string]openClawSessionToolStep)
	stepOrder := make([]string, 0)
	stepOrder, _ = parseOpenClawSessionToolTraceLines(lines, steps, stepOrder)
	return orderedOpenClawSessionToolSteps(steps, stepOrder)
}

func (h *OpenClawHandler) captureOpenClawSessionToolTraceBaseline(
	room ClawRoomEntry,
	worker ContainerEntry,
) (sessionFile string, offset int64) {
	if !h.canWatchOpenClawSessionToolTrace(worker) {
		return "", 0
	}
	sessionUser := roomScopedSessionUser(room, worker)
	sessionKeys := openClawChatCompletionsSessionKeys(sessionUser)
	resolved, err := h.resolveOpenClawSessionFile(worker.Name, sessionKeys)
	if err != nil {
		return "", 0
	}
	data, err := h.readOpenClawContainerFile(worker.Name, resolved)
	if err != nil {
		return resolved, 0
	}
	return resolved, int64(len(data))
}

func (h *OpenClawHandler) resolveOpenClawSessionFile(containerName string, sessionKeys []string) (string, error) {
	indexData, err := h.readOpenClawContainerFile(containerName, openClawSessionsIndexPath())
	if err != nil {
		return "", err
	}
	sessionFile, ok := resolveOpenClawSessionFileFromIndex(indexData, sessionKeys)
	if !ok {
		return "", fmt.Errorf("session file not found for keys %v", sessionKeys)
	}
	return sessionFile, nil
}

func parseOpenClawSessionToolTraceLines(
	lines []string,
	steps map[string]openClawSessionToolStep,
	order []string,
) ([]string, bool) {
	changed := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		nextOrder, lineChanged := parseOpenClawSessionToolTraceLine(line, steps, order)
		if lineChanged {
			changed = true
			order = nextOrder
		}
	}
	return order, changed
}

func parseOpenClawSessionToolTraceLine(
	line string,
	steps map[string]openClawSessionToolStep,
	order []string,
) ([]string, bool) {
	var record struct {
		Type    string `json:"type"`
		Message struct {
			Role       string            `json:"role"`
			Content    []json.RawMessage `json:"content"`
			ToolCallID string            `json:"toolCallId"`
			ToolName   string            `json:"toolName"`
			IsError    bool              `json:"isError"`
		} `json:"message"`
	}
	if err := json.Unmarshal([]byte(line), &record); err != nil || record.Type != "message" {
		return order, false
	}

	changed := false
	if record.Message.Role == "assistant" {
		for _, raw := range record.Message.Content {
			var part struct {
				Type      string          `json:"type"`
				ID        string          `json:"id"`
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			if err := json.Unmarshal(raw, &part); err != nil || part.Type != "toolCall" {
				continue
			}
			callID := strings.TrimSpace(part.ID)
			if callID == "" {
				continue
			}
			next := openClawSessionToolStep{
				ID:        callID,
				Name:      strings.TrimSpace(part.Name),
				Arguments: stringifyToolTraceJSON(part.Arguments),
				Status:    "running",
			}
			if existing, ok := steps[callID]; ok {
				if existing.Status == "completed" || existing.Status == "failed" {
					continue
				}
				if existing.Result != "" {
					next.Result = existing.Result
				}
				if existing.Error != "" {
					next.Error = existing.Error
				}
			}
			steps[callID] = next
			order = appendToolTraceOrder(order, callID)
			changed = true
		}
	}

	if record.Message.Role == "toolResult" {
		callID := strings.TrimSpace(record.Message.ToolCallID)
		if callID == "" {
			return order, changed
		}
		existing := steps[callID]
		existing.ID = callID
		if existing.Name == "" {
			existing.Name = strings.TrimSpace(record.Message.ToolName)
		}
		existing.Result = extractOpenClawToolResultText(record.Message.Content)
		if record.Message.IsError {
			existing.Status = "failed"
			if existing.Result == "" {
				existing.Error = "tool returned an error"
			} else {
				existing.Error = existing.Result
			}
		} else {
			existing.Status = "completed"
		}
		steps[callID] = existing
		order = appendToolTraceOrder(order, callID)
		changed = true
	}

	return order, changed
}

func appendToolTraceOrder(order []string, callID string) []string {
	for _, existing := range order {
		if existing == callID {
			return order
		}
	}
	return append(order, callID)
}

func stringifyToolTraceJSON(raw json.RawMessage) string {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return ""
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, raw); err != nil {
		return trimmed
	}
	return compact.String()
}

func extractOpenClawToolResultText(raw []json.RawMessage) string {
	parts := make([]string, 0, len(raw))
	for _, item := range raw {
		var part struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		if err := json.Unmarshal(item, &part); err != nil {
			continue
		}
		if strings.TrimSpace(part.Text) != "" {
			parts = append(parts, part.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func orderedOpenClawSessionToolSteps(
	steps map[string]openClawSessionToolStep,
	order []string,
) []openClawSessionToolStep {
	ordered := make([]openClawSessionToolStep, 0, len(order))
	for _, callID := range order {
		step, ok := steps[callID]
		if !ok {
			continue
		}
		ordered = append(ordered, step)
	}
	return ordered
}

func toolTraceUpdateCollaborationEvent(
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	payload openClawSessionToolTracePayload,
) ClawRoomCollaborationEvent {
	steps := payload.Steps
	payloadMap := map[string]any{
		"revision": payload.Revision,
		"steps":    steps,
	}
	return ClawRoomCollaborationEvent{
		Type:            WSTypeToolTraceUpdate,
		MessageID:       messageID,
		ParticipantType: "worker",
		ParticipantID:   worker.Name,
		SessionUser:     roomScopedSessionUser(room, worker),
		Payload:         payloadMap,
	}
}

func takeOpenClawSessionToolTraceSteps(messageID string) []openClawSessionToolStep {
	raw, ok := openClawSessionToolTraceStepRegistry.LoadAndDelete(messageID)
	if !ok {
		return nil
	}
	steps, ok := raw.([]openClawSessionToolStep)
	if !ok || len(steps) == 0 {
		return nil
	}
	return append([]openClawSessionToolStep(nil), steps...)
}

func (h *OpenClawHandler) attachOpenClawSessionToolTraceToReply(
	messageID string,
	done chan struct{},
	room ClawRoomEntry,
	worker ContainerEntry,
	sessionFile string,
	baselineOffset int64,
	reply *ClawRoomMessage,
) {
	finishOpenClawSessionToolTraceWatcher(messageID, done)
	steps := h.readOpenClawSessionToolTraceSteps(room, worker, sessionFile, baselineOffset)
	if len(steps) == 0 {
		steps = takeOpenClawSessionToolTraceSteps(messageID)
	} else {
		openClawSessionToolTraceStepRegistry.Delete(messageID)
	}
	if reply != nil {
		attachRoomMessageToolTraceMetadata(reply, steps)
	}
}

func (h *OpenClawHandler) startOpenClawSessionToolTraceWatcher(
	roomID string,
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	done <-chan struct{},
	sessionFile string,
	baselineOffset int64,
) context.CancelFunc {
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		defer cancel()
		<-done
	}()
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				log.Printf("openclaw: session tool trace watcher panic for %s: %v", messageID, recovered)
			}
		}()
		h.streamOpenClawSessionToolTrace(ctx, roomID, room, worker, messageID, sessionFile, baselineOffset)
	}()
	return cancel
}

var (
	openClawSessionToolTraceWatcherRegistry sync.Map
	openClawSessionToolTraceStepRegistry    sync.Map
)

func (h *OpenClawHandler) canWatchOpenClawSessionToolTrace(worker ContainerEntry) bool {
	containerName := sanitizeContainerName(worker.Name)
	if containerName == "" {
		return false
	}
	return h.containerRunning(containerName)
}

func (h *OpenClawHandler) bindOpenClawSessionToolTraceWatcher(
	roomID string,
	room ClawRoomEntry,
	worker ContainerEntry,
	messageID string,
	sessionFile string,
	baselineOffset int64,
) (done chan struct{}, cancel context.CancelFunc) {
	done = make(chan struct{})
	if !h.canWatchOpenClawSessionToolTrace(worker) {
		return done, func() {}
	}
	cancel = h.startOpenClawSessionToolTraceWatcher(
		roomID,
		room,
		worker,
		messageID,
		done,
		sessionFile,
		baselineOffset,
	)
	openClawSessionToolTraceWatcherRegistry.Store(messageID, cancel)
	return done, cancel
}

func finishOpenClawSessionToolTraceWatcher(messageID string, done chan struct{}) {
	if done != nil {
		close(done)
	}
	if raw, ok := openClawSessionToolTraceWatcherRegistry.LoadAndDelete(messageID); ok {
		if cancel, ok := raw.(context.CancelFunc); ok {
			cancel()
		}
	}
}
