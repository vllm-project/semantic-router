package agentmanagement

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/google/uuid"
)

const (
	maximumEventPayloadBytes     = 256 << 10
	maximumInlineToolResultBytes = 64 << 10
	maximumEventTextBytes        = 64 << 10
	maximumModelStepLabelBytes   = 256
	maximumModelStepDurationMS   = int64((24 * time.Hour) / time.Millisecond)
	maximumModelStepTokens       = int64(1<<53 - 1)
	minimumToolCredentialBytes   = 8
)

// NormalizeEventAppend is the single persistence boundary for the durable
// transcript. It accepts only the closed event vocabulary, enforces bounded
// payloads, and redacts credential-shaped fields from tool arguments/results.
// Runtime credentials are injected by transports only after these values are
// serialized and therefore can never be represented by an Agent event.
func NormalizeEventAppend(request EventAppend) (EventAppend, error) {
	if len(request.Payload) == 0 || len(request.Payload) > maximumEventPayloadBytes {
		return EventAppend{}, fmt.Errorf("%w: Agent event payload exceeds its bound", ErrInvalid)
	}
	if request.Origin != "control" && request.Origin != "worker" {
		return EventAppend{}, fmt.Errorf("%w: Agent event origin is invalid", ErrInvalid)
	}
	if (request.Origin == "worker") != (request.Fence != nil) {
		return EventAppend{}, fmt.Errorf("%w: Agent event fence is invalid", ErrInvalid)
	}

	payload, err := normalizeEventPayload(request.Type, request.Payload)
	if err != nil {
		return EventAppend{}, err
	}
	encoded, err := json.Marshal(payload)
	if err != nil || len(encoded) > maximumEventPayloadBytes {
		return EventAppend{}, fmt.Errorf("%w: Agent event cannot be encoded safely", ErrInvalid)
	}
	request.Payload = encoded
	return request, nil
}

func normalizeEventPayload(eventType EventType, raw json.RawMessage) (any, error) {
	switch eventType {
	case EventUserInput:
		value, err := decodeTranscript[UserInputEvent](raw)
		if err != nil {
			return EventAppend{}, fmt.Errorf("%w: Agent user input event is invalid", ErrInvalid)
		}
		normalized, err := NormalizeTurnInput(TurnInput(value))
		if err != nil {
			return EventAppend{}, fmt.Errorf("%w: Agent user input event is invalid", ErrInvalid)
		}
		value.Content = normalized.Content
		return value, nil
	case EventAssistantDelta:
		value, err := decodeTranscript[AssistantDeltaEvent](raw)
		if err != nil || uuid.Validate(value.ModelStepID) != nil || value.ChunkIndex < 0 ||
			validateAssistantDelta(value.Delta) != nil {
			return EventAppend{}, fmt.Errorf("%w: Agent assistant delta is invalid", ErrInvalid)
		}
		value.Delta.Text = sanitizeTranscriptText(value.Delta.Text, maximumEventTextBytes)
		return value, nil
	case EventModelStepSummary:
		value, err := decodeTranscript[ModelStepSummaryEvent](raw)
		if err != nil || validateModelStepSummary(value) != nil {
			return EventAppend{}, fmt.Errorf("%w: Agent model step summary is invalid", ErrInvalid)
		}
		return value, nil
	case EventToolRequest:
		value, err := decodeTranscript[ToolRequestEvent](raw)
		if err != nil || uuid.Validate(value.InvocationID) != nil || !canonicalToolName(value.ToolName) ||
			(value.Class != ToolRead && value.Class != ToolWrite && value.Class != ToolExecute) {
			return EventAppend{}, fmt.Errorf("%w: Agent tool request event is invalid", ErrInvalid)
		}
		value.Arguments, err = sanitizeTranscriptObject(value.Arguments, maximumInlineToolResultBytes)
		if err != nil {
			return EventAppend{}, err
		}
		return value, nil
	case EventToolResult:
		value, err := decodeTranscript[ToolResultEvent](raw)
		if err != nil || validateToolResultEvent(value) != nil {
			return EventAppend{}, fmt.Errorf("%w: Agent tool result event is invalid", ErrInvalid)
		}
		if len(value.Result) != 0 {
			value.Result, err = sanitizeTranscriptObject(value.Result, maximumInlineToolResultBytes)
			if err != nil {
				return EventAppend{}, err
			}
		}
		value.Error = sanitizeFailure(value.Error)
		return value, nil
	case EventProgress:
		value, err := decodeTranscript[ProgressEvent](raw)
		if err != nil || !validTranscriptLabel(value.Phase, 64) || strings.TrimSpace(value.Message) == "" {
			return EventAppend{}, fmt.Errorf("%w: Agent progress event is invalid", ErrInvalid)
		}
		value.Message = sanitizeTranscriptText(value.Message, 1024)
		return value, nil
	case EventContextCheckpoint:
		value, err := decodeTranscript[ContextCheckpointEvent](raw)
		if err != nil || uuid.Validate(value.CheckpointID) != nil || value.ThroughSequence < 1 {
			return EventAppend{}, fmt.Errorf("%w: Agent checkpoint event is invalid", ErrInvalid)
		}
		return value, nil
	case EventApprovalRequest:
		value, err := decodeTranscript[ApprovalRequestEvent](raw)
		if err != nil || uuid.Validate(value.PlanID) != nil || !validSHA256Digest(value.PlanDigest) ||
			value.PlanRevision < 1 || strings.TrimSpace(value.PlanETag) == "" || value.ExpiresAt.IsZero() {
			return EventAppend{}, fmt.Errorf("%w: Agent approval request event is invalid", ErrInvalid)
		}
		if value.Summary.Topology, err = sanitizeOptionalTranscriptObject(value.Summary.Topology); err != nil {
			return EventAppend{}, err
		}
		if value.Summary.Assignments, err = sanitizeOptionalTranscriptObject(value.Summary.Assignments); err != nil {
			return EventAppend{}, err
		}
		if value.Summary.GateResults, err = sanitizeOptionalTranscriptValue(value.Summary.GateResults); err != nil {
			return EventAppend{}, err
		}
		return value, nil
	case EventApprovalResult:
		value, err := decodeTranscript[ApprovalResultEvent](raw)
		if err != nil || uuid.Validate(value.PlanID) != nil ||
			(value.Status != "committed" && value.Status != "rejected" && value.Status != "expired" && value.Status != "failed") ||
			(value.OperationID != "" && uuid.Validate(value.OperationID) != nil) {
			return EventAppend{}, fmt.Errorf("%w: Agent approval result event is invalid", ErrInvalid)
		}
		return value, nil
	case EventCancellation:
		value, err := decodeTranscript[CancellationEvent](raw)
		if err != nil || value.RequestedAt.IsZero() {
			return EventAppend{}, fmt.Errorf("%w: Agent cancellation event is invalid", ErrInvalid)
		}
		return value, nil
	case EventTerminal:
		value, err := decodeTranscript[TerminalEvent](raw)
		if err != nil || (value.Status != TurnCompleted && value.Status != TurnFailed && value.Status != TurnCancelled) ||
			(value.Status == TurnFailed) != (value.Error != nil) {
			return EventAppend{}, fmt.Errorf("%w: Agent terminal event is invalid", ErrInvalid)
		}
		value.Error = sanitizeFailure(value.Error)
		return value, nil
	default:
		return nil, fmt.Errorf("%w: Agent event type is invalid", ErrInvalid)
	}
}

func decodeTranscript[T any](raw []byte) (T, error) {
	var value T
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return value, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			return value, fmt.Errorf("trailing Agent event content")
		}
		return value, err
	}
	return value, nil
}

func validateAssistantDelta(delta AssistantDelta) error {
	if delta.Kind != AssistantTextDelta || delta.Text == "" ||
		len(delta.Text) > maximumEventTextBytes || !utf8.ValidString(delta.Text) {
		return ErrInvalid
	}
	return nil
}

func validateModelStepSummary(value ModelStepSummaryEvent) error {
	if uuid.Validate(value.ModelStepID) != nil ||
		!validModelStepLabel(value.RequestID, true) ||
		!validModelStepLabel(value.SelectedRecipe, false) ||
		!validModelStepLabel(value.SelectedDecision, false) ||
		!validModelStepLabel(value.SelectedModel, false) ||
		!validModelStepLabel(value.SelectedAlgorithm, false) ||
		(value.ResponsePath != "" && !validModelStepResponsePath(value.ResponsePath)) ||
		value.LatencyMilliseconds < 0 || value.LatencyMilliseconds > maximumModelStepDurationMS ||
		(value.TTFTMilliseconds != nil && (*value.TTFTMilliseconds < 0 ||
			*value.TTFTMilliseconds > value.LatencyMilliseconds)) {
		return ErrInvalid
	}
	if value.Usage != nil && validateModelStepUsage(*value.Usage) != nil {
		return ErrInvalid
	}
	return nil
}

func validModelStepLabel(value string, required bool) bool {
	if value == "" {
		return !required
	}
	if !validTranscriptLabel(value, maximumModelStepLabelBytes) {
		return false
	}
	for _, character := range value {
		if unicode.IsControl(character) || !unicode.IsGraphic(character) {
			return false
		}
	}
	return true
}

func validModelStepResponsePath(value string) bool {
	switch value {
	case "upstream", "cache", "fast_response", "looper", "image_generation":
		return true
	default:
		return false
	}
}

func validateModelStepUsage(value ModelStepUsage) error {
	if !validModelStepTokenCount(value.InputTokens) ||
		!validModelStepTokenCount(value.OutputTokens) ||
		!validModelStepTokenCount(value.TotalTokens) ||
		value.InputTokens > maximumModelStepTokens-value.OutputTokens ||
		value.TotalTokens != value.InputTokens+value.OutputTokens {
		return ErrInvalid
	}
	for _, count := range []*int64{
		value.InputUncachedTokens,
		value.InputCacheReadTokens,
		value.InputCacheWriteTokens,
		value.OutputReasoningTokens,
		value.OutputOtherTokens,
	} {
		if count != nil && !validModelStepTokenCount(*count) {
			return ErrInvalid
		}
	}
	for _, count := range []*int64{
		value.InputUncachedTokens, value.InputCacheReadTokens, value.InputCacheWriteTokens,
	} {
		if count != nil && *count > value.InputTokens {
			return ErrInvalid
		}
	}
	for _, count := range []*int64{value.OutputReasoningTokens, value.OutputOtherTokens} {
		if count != nil && *count > value.OutputTokens {
			return ErrInvalid
		}
	}
	return nil
}

func validModelStepTokenCount(value int64) bool {
	return value >= 0 && value <= maximumModelStepTokens
}

func validateToolResultEvent(value ToolResultEvent) error {
	if uuid.Validate(value.InvocationID) != nil || !canonicalToolName(value.ToolName) ||
		(value.Status != "completed" && value.Status != "failed" && value.Status != "cancelled") ||
		(value.ArtifactID != "" && uuid.Validate(value.ArtifactID) != nil) {
		return ErrInvalid
	}
	if value.Status == "completed" {
		if value.Error != nil || (len(value.Result) == 0) == (value.ArtifactID == "") {
			return ErrInvalid
		}
	} else if len(value.Result) != 0 || value.ArtifactID != "" || value.Error == nil {
		return ErrInvalid
	}
	return nil
}

func sanitizeOptionalTranscriptObject(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	return sanitizeTranscriptObject(raw, maximumInlineToolResultBytes)
}

func sanitizeOptionalTranscriptValue(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	value, err := decodeBoundedJSON(raw, maximumValueDepth, maximumValueNodes)
	if err != nil {
		return nil, fmt.Errorf("%w: Agent transcript JSON is invalid", ErrInvalid)
	}
	return json.Marshal(redactTranscriptValue(value))
}

func sanitizeTranscriptObject(raw json.RawMessage, maximum int) (json.RawMessage, error) {
	return sanitizeTranscriptObjectWithSecrets(raw, maximum, nil)
}

// ScrubToolSecrets removes every exact occurrence of a transport-owned
// plaintext credential from a closed JSON tool object. Remote transports call
// this both before an invocation is made durable and before its result is
// returned. Secret values never become heuristics: every supplied byte string
// must be a scrub-safe UTF-8 credential and is replaced inside nested string
// values and object keys.
func ScrubToolSecrets(raw json.RawMessage, secrets ...[]byte) (json.RawMessage, error) {
	return sanitizeTranscriptObjectWithSecrets(raw, maximumInlineToolResultBytes, secrets)
}

func sanitizeTranscriptObjectWithSecrets(
	raw json.RawMessage, maximum int, secrets [][]byte,
) (json.RawMessage, error) {
	if len(raw) == 0 || len(raw) > maximum {
		return nil, fmt.Errorf("%w: inline Agent tool content exceeds its bound; use an Artifact", ErrInvalid)
	}
	value, err := decodeBoundedJSON(raw, maximumValueDepth, maximumValueNodes)
	if err != nil {
		return nil, fmt.Errorf("%w: Agent transcript JSON is invalid", ErrInvalid)
	}
	if _, object := value.(map[string]any); !object {
		return nil, fmt.Errorf("%w: Agent transcript tool content must be an object", ErrInvalid)
	}
	canonical, err := canonicalExactSecrets(secrets)
	if err != nil {
		return nil, err
	}
	marker := exactSecretMarker(canonical)
	exact, err := redactExactTranscriptValue(redactTranscriptValue(value), canonical, marker)
	if err != nil {
		return nil, err
	}
	encoded, err := json.Marshal(exact)
	if err != nil || len(encoded) > maximum {
		return nil, fmt.Errorf("%w: inline Agent tool content exceeds its bound; use an Artifact", ErrInvalid)
	}
	return encoded, nil
}

func canonicalExactSecrets(values [][]byte) ([]string, error) {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		if len(value) == 0 {
			continue
		}
		if len(value) < minimumToolCredentialBytes || !utf8.Valid(value) {
			return nil, fmt.Errorf("%w: tool credential cannot be scrubbed safely", ErrInvalid)
		}
		secret := string(value)
		if _, duplicate := seen[secret]; duplicate {
			continue
		}
		seen[secret] = struct{}{}
		result = append(result, secret)
	}
	sort.Slice(result, func(left, right int) bool {
		if len(result[left]) == len(result[right]) {
			return result[left] < result[right]
		}
		return len(result[left]) > len(result[right])
	})
	return result, nil
}

func exactSecretMarker(secrets []string) string {
	for _, candidate := range []string{"[redacted]", "<credential removed>", "credential removed", ""} {
		safe := true
		for _, secret := range secrets {
			if strings.Contains(candidate, secret) {
				safe = false
				break
			}
		}
		if safe {
			return candidate
		}
	}
	return ""
}

func redactExactTranscriptValue(value any, secrets []string, marker string) (any, error) {
	switch typed := value.(type) {
	case map[string]any:
		clean := make(map[string]any, len(typed))
		for key, item := range typed {
			cleanKey := replaceExactSecrets(key, secrets, marker)
			if _, collision := clean[cleanKey]; collision {
				return nil, fmt.Errorf("%w: exact credential redaction produced a duplicate tool field", ErrInvalid)
			}
			cleanItem, err := redactExactTranscriptValue(item, secrets, marker)
			if err != nil {
				return nil, err
			}
			clean[cleanKey] = cleanItem
		}
		return clean, nil
	case []any:
		clean := make([]any, len(typed))
		for index, item := range typed {
			var err error
			clean[index], err = redactExactTranscriptValue(item, secrets, marker)
			if err != nil {
				return nil, err
			}
		}
		return clean, nil
	case string:
		return replaceExactSecrets(typed, secrets, marker), nil
	default:
		return typed, nil
	}
}

func replaceExactSecrets(value string, secrets []string, marker string) string {
	for _, secret := range secrets {
		value = strings.ReplaceAll(value, secret, marker)
	}
	return value
}

func redactTranscriptValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		clean := make(map[string]any, len(typed))
		for key, item := range typed {
			if sensitiveTranscriptKey(key) {
				clean[key] = "[redacted]"
				continue
			}
			clean[key] = redactTranscriptValue(item)
		}
		return clean
	case []any:
		clean := make([]any, len(typed))
		for index, item := range typed {
			clean[index] = redactTranscriptValue(item)
		}
		return clean
	case string:
		lower := strings.ToLower(strings.TrimSpace(typed))
		if strings.HasPrefix(lower, "bearer ") || strings.HasPrefix(lower, "basic ") ||
			strings.HasPrefix(lower, "sk-") || strings.HasPrefix(lower, "vsr_") ||
			strings.HasPrefix(lower, "vsd_") {
			return "[redacted]"
		}
		return sanitizeTranscriptText(typed, maximumInlineToolResultBytes)
	default:
		return typed
	}
}

func sensitiveTranscriptKey(value string) bool {
	canonical := strings.Map(func(character rune) rune {
		if unicode.IsLetter(character) || unicode.IsDigit(character) {
			return unicode.ToLower(character)
		}
		return -1
	}, value)
	for _, suffix := range []string{"authorization", "cookie", "password", "secret", "credential", "apikey", "accesstoken", "refreshtoken"} {
		if canonical == suffix || strings.HasSuffix(canonical, suffix) {
			return true
		}
	}
	return false
}

func sanitizeFailure(value *Failure) *Failure {
	if value == nil {
		return nil
	}
	return &Failure{
		Code:      sanitizeTranscriptText(value.Code, 128),
		Message:   sanitizeTranscriptText(value.Message, 1024),
		Retryable: value.Retryable,
	}
}

func sanitizeTranscriptText(value string, maximum int) string {
	value = strings.Map(func(character rune) rune {
		if character == '\n' || character == '\t' {
			return character
		}
		if unicode.IsControl(character) || !unicode.IsGraphic(character) {
			return ' '
		}
		return character
	}, value)
	if len(value) <= maximum {
		return value
	}
	for maximum > 0 && !utf8.RuneStart(value[maximum]) {
		maximum--
	}
	return value[:maximum]
}

func validTranscriptLabel(value string, maximum int) bool {
	return value != "" && len(value) <= maximum && strings.TrimSpace(value) == value && utf8.ValidString(value)
}
