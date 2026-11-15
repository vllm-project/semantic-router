// Package codexreview reads Codex "Approve for me" approval reviews: the
// envelope Codex sends its reviewer about one planned agent action, and the
// verdict the reviewer answers. It only observes; nothing here routes,
// authorizes or masks a request.
package codexreview

import (
	"encoding/json"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The review envelope as Codex writes it.
const (
	judgePreamble      = "You are judging one planned coding-agent action"
	approvalStart      = ">>> APPROVAL REQUEST START"
	approvalEnd        = ">>> APPROVAL REQUEST END"
	plannedActionLabel = "Planned action JSON:"
	reviewedSession    = "Reviewed Codex session id:"
)

// Bounds on what a replay record keeps from a review.
const (
	maxIDBytes        = 128
	maxSummaryRunes   = 240
	maxCwdBytes       = 512
	maxArgumentBytes  = 16 << 10
	maxRationaleBytes = 2 << 10
)

// HasEnvelope reports whether request carries Codex's review envelope: the
// judge instructions, an approval request block with a planned action, and
// an output schema whose outcome is allow or deny. A request that only
// mentions reviews or approvals has none of them.
func HasEnvelope(request *llmprotocol.Request) bool {
	if request == nil || !hasJudgeInstructions(request) || !hasVerdictSchema(request.OutputFormat) {
		return false
	}
	block, ok := approvalBlock(userText(request))
	return ok && strings.Contains(block, plannedActionLabel)
}

// Observe returns the pending review of a request recognised by source. The
// body is the wire request, whose client_metadata the protocol codec drops.
func Observe(request *llmprotocol.Request, body []byte, source string) *store.Review {
	metadata := readClientMetadata(body)
	text := userText(request)
	review := &store.Review{
		SchemaVersion:        store.ReviewSchemaVersion,
		IdentificationSource: source,
		ReviewedSessionID:    boundedID(reviewedSessionID(text, metadata.sessionID)),
		ThreadID:             boundedID(metadata.threadID),
		TurnID:               boundedID(metadata.turnID),
		ParentTurnID:         boundedID(metadata.parentTurnID),
		Status:               store.ReviewStatusPending,
	}
	if block, ok := approvalBlock(text); ok {
		review.Action = plannedAction(block)
	}
	return review
}

func hasJudgeInstructions(request *llmprotocol.Request) bool {
	for _, block := range request.Instructions {
		if strings.HasPrefix(strings.TrimSpace(contentText(block.Content)), judgePreamble) {
			return true
		}
	}
	for _, message := range request.Messages {
		if message.Role != llmprotocol.RoleDeveloper && message.Role != llmprotocol.RoleSystem {
			continue
		}
		if strings.HasPrefix(strings.TrimSpace(contentText(message.Content)), judgePreamble) {
			return true
		}
	}
	return false
}

// hasVerdictSchema requires a JSON schema whose outcome property is the
// allow or deny enumeration.
func hasVerdictSchema(format llmprotocol.OutputFormat) bool {
	if format.Kind != llmprotocol.OutputJSONSchema || len(format.Schema) == 0 {
		return false
	}
	var schema struct {
		Properties struct {
			Outcome struct {
				Enum []string `json:"enum"`
			} `json:"outcome"`
		} `json:"properties"`
	}
	if json.Unmarshal(format.Schema, &schema) != nil {
		return false
	}
	enum := schema.Properties.Outcome.Enum
	return slices.Contains(enum, store.ReviewOutcomeAllow) && slices.Contains(enum, store.ReviewOutcomeDeny)
}

func userText(request *llmprotocol.Request) string {
	if request == nil {
		return ""
	}
	var text strings.Builder
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleUser {
			text.WriteString(contentText(message.Content))
		}
	}
	return text.String()
}

func contentText(content []llmprotocol.Content) string {
	var text strings.Builder
	for _, part := range content {
		if part.Kind == llmprotocol.ContentText {
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

// approvalBlock returns the last approval request block. Codex writes the
// transcript first, so an earlier start marker quoted in it is not the block.
func approvalBlock(text string) (string, bool) {
	start := strings.LastIndex(text, approvalStart)
	if start < 0 {
		return "", false
	}
	rest := text[start+len(approvalStart):]
	end := strings.Index(rest, approvalEnd)
	if end < 0 {
		return "", false
	}
	return rest[:end], true
}

func reviewedSessionID(text, fallback string) string {
	index := strings.LastIndex(text, reviewedSession)
	if index < 0 {
		return fallback
	}
	line, _, _ := strings.Cut(text[index+len(reviewedSession):], "\n")
	if id := strings.TrimSpace(line); id != "" {
		return id
	}
	return fallback
}

type clientMetadata struct {
	sessionID, threadID, turnID, parentTurnID string
}

// readClientMetadata reads Codex's correlation ids from client_metadata,
// falling back to the turn metadata JSON it also carries there.
func readClientMetadata(body []byte) clientMetadata {
	var wire struct {
		ClientMetadata map[string]string `json:"client_metadata"`
	}
	if json.Unmarshal(body, &wire) != nil || len(wire.ClientMetadata) == 0 {
		return clientMetadata{}
	}
	var turn map[string]any
	_ = json.Unmarshal([]byte(wire.ClientMetadata[headers.XCodexTurnMetadata]), &turn)
	pick := func(key string) string {
		if value := strings.TrimSpace(wire.ClientMetadata[key]); value != "" {
			return value
		}
		value, _ := turn[key].(string)
		return strings.TrimSpace(value)
	}
	return clientMetadata{
		sessionID: pick("session_id"), threadID: pick("thread_id"),
		turnID: pick("turn_id"), parentTurnID: pick("parent_turn_id"),
	}
}

func boundedID(value string) string {
	value, _ = truncate(value, maxIDBytes)
	return value
}

// truncate cuts s to at most max bytes on a rune boundary.
func truncate(s string, max int) (string, bool) {
	if len(s) <= max {
		return s, false
	}
	cut := max
	for cut > 0 && !utf8.RuneStart(s[cut]) {
		cut--
	}
	return s[:cut], true
}
