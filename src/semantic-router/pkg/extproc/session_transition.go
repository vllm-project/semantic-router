package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// populateSessionTransitionFields derives session-aware metadata from the
// request context. The neutral request is authoritative; optional retained
// object history augments it without changing protocol semantics.
func populateSessionTransitionFields(ctx *RequestContext) {
	if ctx == nil {
		return
	}

	if state := ctx.ResponseObjectState; state != nil {
		ctx.SessionID = state.ConversationID
		ctx.PreviousResponseID = state.PreviousResponseID
		history := state.ConversationHistory
		ctx.TurnIndex = len(history)
		for i := len(history) - 1; i >= 0; i-- {
			if history[i] != nil {
				ctx.PreviousModel = history[i].Model
				break
			}
		}
		ctx.HistoryTokenCount = historyTokensFromStoredResponses(history)
		populateLastSessionObservation(ctx)
		return
	}

	if sid := strings.TrimSpace(headerValueCI(ctx, headers.XSessionID)); sid != "" {
		ctx.SessionID = sid
	}

	if ctx.SessionID == "" {
		if sid := deriveSessionIDFromAnthropicSignals(ctx); sid != "" {
			ctx.SessionID = sid
		}
	}

	if ctx.SemanticRequest == nil || len(ctx.SemanticRequest.Messages) == 0 {
		if ctx.SessionID == "" {
			ctx.SessionID = deriveSessionIDFromRequestID(ctx)
		}
		return
	}

	populateSemanticSessionIDIfNeeded(ctx)

	ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionMessages(ctx.SemanticRequest.Messages)) - 1
	ctx.HistoryTokenCount = historyTokensFromSemanticMessages(ctx.SemanticRequest.Messages)
	// Populate PreviousModel for Chat Completions from the in-memory last-model
	// store, recorded at response time of the previous turn in this session.
	// (Response API derives PreviousModel from the conversation chain above.)
	if model, ok := sessiontelemetry.GetLastModel(routingSessionStateKey(ctx)); ok {
		ctx.PreviousModel = model
	}
	populateLastSessionObservation(ctx)
}

// populatePinnedSessionFromHeaders makes client-supplied Chat Completions
// session IDs available before full request parsing. Decision evaluation runs
// on the fast-extract path, so session-aware selection cannot wait for
// populateSessionTransitionFields.
func populatePinnedSessionFromHeaders(ctx *RequestContext) {
	if ctx == nil {
		return
	}
	if sid := strings.TrimSpace(headerValueCI(ctx, headers.XSessionID)); sid != "" {
		ctx.SessionID = sid
	}
	if ctx.SessionID == "" {
		return
	}
	populateLastSessionObservation(ctx)
}

func populateLastSessionObservation(ctx *RequestContext) {
	now := time.Now()
	stateSessionID := routingSessionStateKey(ctx)
	if snapshot, ok := sessiontelemetry.GetRouterSessionSnapshot(stateSessionID, now); ok {
		if ctx.PreviousModel == "" {
			ctx.PreviousModel = snapshot.CurrentModel
		}
		ctx.SessionIdleSeconds = snapshot.IdleFor.Seconds()
		ctx.SessionIdleKnown = true
		return
	}
	model, idleFor, ok := sessiontelemetry.GetLastModelInfo(stateSessionID, now)
	if !ok {
		return
	}
	if ctx.PreviousModel == "" {
		ctx.PreviousModel = model
	}
	ctx.SessionIdleSeconds = idleFor.Seconds()
	ctx.SessionIdleKnown = true
}

// deriveSessionIDFromAnthropicSignals returns a session-ID candidate
// from Anthropic-shape transport and body signals. Two sources, evaluated
// in order:
//
//  1. x-claude-code-session-id — per-conversation UUID emitted by the
//     Claude Code CLI on every /v1/messages request in a thread. Returned
//     verbatim so plugins can map sessions back to the client-declared
//     conversation; operators wanting privacy should run a hashing plugin
//     in front.
//  2. metadata.user_id — populated by the PR2 Anthropic inbound parser
//     into neutral request metadata. Returned with an "ant-md-" prefix
//     so the namespace is distinguishable from other derivation sources.
//
// Returns empty string when neither signal is present, leaving the caller
// to fall through to the chat-message fingerprint fallbacks.
func deriveSessionIDFromAnthropicSignals(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if sid := strings.TrimSpace(headerValueCI(ctx, headers.XClaudeCodeSessionID)); sid != "" {
		return sid
	}
	if ctx.SemanticRequest != nil {
		if uid := strings.TrimSpace(ctx.SemanticRequest.Metadata["user_id"]); uid != "" {
			return "ant-md-" + uid
		}
	}
	return ""
}

// populateSemanticSessionIDIfNeeded sets ctx.SessionID when not already pinned
// (for example by x-session-id). Kept separate to avoid deep nesting in
// populateSessionTransitionFields (nestif).
func populateSemanticSessionIDIfNeeded(ctx *RequestContext) {
	if ctx.SessionID != "" {
		return
	}
	userID := extractUserID(ctx)
	if userID != "" {
		ctx.SessionID = deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, userID)
		return
	}
	ctx.SessionID = deriveSessionIDFromSemanticStructure(ctx.SemanticRequest.Messages)
	if ctx.SessionID == "" {
		ctx.SessionID = deriveSessionIDFromRequestID(ctx)
	}
	if ctx.SessionID == "" {
		logging.Debugf("Session: could not derive session ID for chat completion request")
	}
}

func sessionTransitionMessages(messages []llmprotocol.Message) []sessiontelemetry.ChatMessage {
	converted := make([]sessiontelemetry.ChatMessage, len(messages))
	for i := range messages {
		converted[i] = sessiontelemetry.ChatMessage{
			Role:    string(messages[i].Role),
			Content: semanticText(messages[i].Content),
		}
	}
	return converted
}

func historyTokensFromStoredResponses(history []*responseapi.StoredResponse) int {
	total := 0
	for _, resp := range history {
		if resp == nil {
			continue
		}
		if resp.Usage != nil {
			total += resp.Usage.InputTokens + resp.Usage.OutputTokens
		} else {
			total += estimateStoredResponseTokens(resp)
		}
		if total >= historyTokensCap {
			return historyTokensCap
		}
	}
	return total
}

const historyTokensCap = 8192

func estimateStoredResponseTokens(resp *responseapi.StoredResponse) int {
	total := 0
	for _, item := range resp.Input {
		total += estimateTokenLength(extractContentFromInputItem(item))
	}

	if resp.OutputText != "" {
		return total + estimateTokenLength(resp.OutputText)
	}

	for _, item := range resp.Output {
		total += estimateTokenLength(extractContentFromOutputItem(item))
	}

	return total
}

func historyTokensFromSemanticMessages(messages []llmprotocol.Message) int {
	if len(messages) == 0 {
		return 0
	}
	prior := messages
	if messages[len(messages)-1].Role == llmprotocol.RoleUser {
		prior = messages[:len(messages)-1]
	}
	total := 0
	for _, msg := range prior {
		total += estimateTokenLength(semanticText(msg.Content))
		if total >= historyTokensCap {
			return historyTokensCap
		}
	}
	return total
}

func estimateTokenLength(text string) int {
	return len(text) / 4
}
