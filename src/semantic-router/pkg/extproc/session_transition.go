package extproc

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
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
	ctx.AuthenticatedPrincipal = authHeaderUserID(ctx)

	pinned, pinnedHeader := pinnedSessionID(ctx)
	if state := ctx.ResponseObjectState; state != nil {
		// The client session correlates routing and Replay across protocols;
		// retained-object lineage remains the fallback when it is not supplied.
		ctx.SessionID = pinned
		if ctx.SessionID != "" {
			ctx.SessionProvenance, ctx.SessionIDHeader = SessionProvenanceHeader, pinnedHeader
		} else {
			ctx.SessionID = state.SessionTrackingID
			ctx.SessionProvenance = SessionProvenanceResponseAPI
		}
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

	if pinned != "" {
		ctx.SessionID = pinned
		ctx.SessionProvenance, ctx.SessionIDHeader = SessionProvenanceHeader, pinnedHeader
	}
	// Protection may key on it later, after routing has edited the request.
	ctx.conversationStart()

	if ctx.SessionID == "" {
		if sid := deriveSessionIDFromAnthropicSignals(ctx); sid != "" {
			ctx.SessionID = sid
			ctx.SessionProvenance = SessionProvenanceAnthropicPromptCache
		}
	}

	if ctx.SemanticRequest == nil || len(ctx.SemanticRequest.Messages) == 0 {
		if ctx.SessionID == "" {
			setSessionIDFromRequestIDFallback(ctx)
		}
		return
	}

	populateSemanticSessionIDIfNeeded(ctx)

	ctx.TurnIndex = sessiontelemetry.ChatTurnNumber(sessionTransitionMessages(ctx.SemanticRequest.Messages)) - 1
	ctx.HistoryTokenCount = historyTokensFromConversation(originalConversation(ctx))
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
	ctx.AuthenticatedPrincipal = authHeaderUserID(ctx)
	if sid, header := pinnedSessionID(ctx); sid != "" {
		ctx.SessionID = sid
		ctx.SessionProvenance, ctx.SessionIDHeader = SessionProvenanceHeader, header
	}
	if ctx.SessionID == "" {
		return
	}
	populateLastSessionObservation(ctx)
}

// pinnedSessionID is the session id a request pins and the header that
// carries it: x-vsr-session-id, else the first client header
// global.router.headers.session_id_headers accepts.
func pinnedSessionID(ctx *RequestContext) (string, string) {
	names := ctx.sessionIDHeaders
	if len(names) == 0 {
		names = config.RouterHeadersConfig{}.SessionIDHeaderNames()
	}
	for _, name := range names {
		if sid := strings.TrimSpace(headerValueCI(ctx, name)); sid != "" {
			return sid, name
		}
	}
	return "", ""
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

// deriveSessionIDFromAnthropicSignals returns a session-ID candidate from
// the Anthropic body's metadata.user_id, which the Anthropic inbound parser
// puts into neutral request metadata. It is returned with an "ant-md-"
// prefix so the namespace is distinguishable from other derivation sources.
// Claude Code's x-claude-code-session-id is an accepted session header, so
// it pins the session before this runs.
//
// Returns empty string when the signal is absent, leaving the caller to
// fall through to the chat-message fingerprint fallbacks.
func deriveSessionIDFromAnthropicSignals(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	if ctx.SemanticRequest != nil {
		if uid := strings.TrimSpace(ctx.SemanticRequest.Metadata["user_id"]); uid != "" {
			return "ant-md-" + uid
		}
	}
	return ""
}

// populateSemanticSessionIDIfNeeded sets ctx.SessionID when not already pinned
// (for example by x-vsr-session-id). Kept separate to avoid deep nesting in
// populateSessionTransitionFields (nestif).
func populateSemanticSessionIDIfNeeded(ctx *RequestContext) {
	if ctx.SessionID != "" {
		return
	}
	userID := extractUserID(ctx)
	if userID != "" {
		ctx.SessionID = deriveSessionIDFromSemanticMessages(ctx.SemanticRequest.Messages, userID)
		ctx.SessionProvenance = SessionProvenanceMessageHash
		return
	}
	ctx.SessionID = ctx.conversationStart()
	if ctx.SessionID != "" {
		ctx.SessionProvenance = SessionProvenanceMessageHash
	}
	if ctx.SessionID == "" {
		setSessionIDFromRequestIDFallback(ctx)
	}
	if ctx.SessionID == "" {
		logging.Debugf("Session: could not derive session ID for chat completion request")
	}
}

// conversationStart is the request's conversation-start fingerprint
// (deriveConversationStartID), taken once from the request as the client sent
// it, so that later Router edits to the request cannot move it. A Responses
// request has none: its retained lineage already ties its turns together.
func (ctx *RequestContext) conversationStart() string {
	if ctx == nil || ctx.ResponseObjectState != nil {
		return ""
	}
	if !ctx.conversationStartTaken && ctx.SemanticRequest != nil {
		ctx.conversationStartTaken = true
		tenant := strings.TrimSpace(headerValueCI(ctx, headers.AuthzTenantID))
		ctx.conversationStartID = deriveConversationStartID(ctx.SemanticRequest, tenant)
	}
	return ctx.conversationStartID
}

// setSessionIDFromRequestIDFallback assigns ctx.SessionID and its
// SessionProvenanceRequestID provenance from the neutral request ID when a
// derived ID is available; a no-op otherwise. Extracted out of its two call
// sites above (unchanged behavior) so the request-ID fallback's branch is
// paid for once, keeping populateSessionTransitionFields within the
// repository's cyclomatic-complexity budget (agent-lint cyclop).
func setSessionIDFromRequestIDFallback(ctx *RequestContext) {
	if sid := deriveSessionIDFromRequestID(ctx); sid != "" {
		ctx.SessionID = sid
		ctx.SessionProvenance = SessionProvenanceRequestID
	}
}

// sessionPositionFacts derives the trajectory-position facts the conversation
// signal family exposes: the zero-based user turn index (tool continuations
// grouped with the user message that started them, matching the router-replay
// TurnIndex semantics) and the token size of the conversation history the
// request carries. Both mirror the session-transition fields the router records
// after routing, computed from the original request snapshot so routing edits
// cannot renumber turns, and both stay available at decision time —
// populateSessionTransitionFields runs later in the request path.
func sessionPositionFacts(ctx *RequestContext) (turnIndex int, historyTokens int) {
	if ctx == nil {
		return 0, 0
	}
	// A Responses continuation carries the retained responses' billed usage.
	// Counting it matches the post-routing record instead of re-estimating the
	// projection the object store prepended to the request.
	if state := ctx.ResponseObjectState; state != nil && len(state.ConversationHistory) > 0 {
		return replayUserTurnIndex(ctx), historyTokensFromStoredResponses(state.ConversationHistory)
	}
	history := originalConversation(ctx)
	if len(history.Messages) == 0 && len(history.Instructions) == 0 {
		return 0, 0
	}
	return replayUserTurnIndex(ctx), historyTokensFromConversation(history)
}

// originalConversation returns the request's pre-routing conversation with the
// instructions the client sent alongside it.
func originalConversation(ctx *RequestContext) contextcompression.ConversationHistory {
	if ctx.OriginalContextHistory != nil {
		return ctx.OriginalContextHistory.Conversation()
	}
	if ctx.SemanticRequest != nil {
		return contextcompression.ConversationHistory{
			Instructions: ctx.SemanticRequest.Instructions,
			Messages:     ctx.SemanticRequest.Messages,
		}
	}
	return contextcompression.ConversationHistory{}
}

// sessionTransitionMessages converts neutral messages for turn counting.
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
	}
	return total
}

func estimateStoredResponseTokens(resp *responseapi.StoredResponse) int {
	total := 0
	for _, item := range resp.Input {
		total += estimateTokenLength(extractContentFromInputItem(item))
		total += estimateToolItemTokens(item.Type, item.Arguments, derefToolInput(item.Input), string(item.Output))
	}

	if resp.OutputText != "" {
		return total + estimateTokenLength(resp.OutputText)
	}

	for _, item := range resp.Output {
		total += estimateTokenLength(extractContentFromOutputItem(item))
		total += estimateToolItemTokens(item.Type, item.Arguments, derefToolInput(item.Input), item.Output)
	}

	return total
}

// estimateToolItemTokens counts the payload of tool-call and tool-output
// items, whose text lives outside the content fields the generic extractors
// read. A historical tool result is the bulk of an agent session's history,
// so a session gate reading the estimate would otherwise never see the bytes
// the tool loop added.
func estimateToolItemTokens(itemType, arguments, customInput, output string) int {
	switch itemType {
	case "function_call":
		return estimateTokenLength(arguments)
	case "custom_tool_call":
		return estimateTokenLength(customInput)
	case "function_call_output", "custom_tool_call_output":
		return estimateTokenLength(toolOutputText(output))
	}
	return 0
}

func derefToolInput(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

// toolOutputText renders a tool output for the token estimate: a JSON string,
// an array of content parts, or any opaque payload by its raw bytes.
func toolOutputText(raw string) string {
	if raw == "" {
		return ""
	}
	var text string
	if err := json.Unmarshal([]byte(raw), &text); err == nil {
		return text
	}
	var parts []responseapi.ContentPart
	if err := json.Unmarshal([]byte(raw), &parts); err == nil {
		return contentPartsText(parts)
	}
	return raw
}

// contentPartsText joins the text of the content parts that legally carry it:
// an official function_call_output uses input_text parts, an output item uses
// output_text, and some providers send bare text. The output-side extractor
// reads only output_text, so reusing it here dropped every valid tool output
// sent as an input_text array.
func contentPartsText(parts []responseapi.ContentPart) string {
	var text strings.Builder
	for _, part := range parts {
		switch part.Type {
		case "text", "input_text", "output_text":
			text.WriteString(part.Text)
		}
	}
	return text.String()
}

func historyTokensFromSemanticMessages(messages []llmprotocol.Message) int {
	total := 0
	for _, msg := range historyMessagesBeforeCurrentTurn(messages) {
		total += contentBytes(msg.Content) / 4
	}
	return total
}

// historyTokensFromConversation estimates the history a request carries: the
// instructions plus every message except the current user turn. Instructions
// live outside the message list, so reading messages alone dropped the system
// and developer text every turn resends.
func historyTokensFromConversation(history contextcompression.ConversationHistory) int {
	total := 0
	for _, instruction := range history.Instructions {
		total += contentBytes(instruction.Content) / 4
	}
	return total + historyTokensFromSemanticMessages(history.Messages)
}

// historyMessagesBeforeCurrentTurn drops the current user message: the history
// the session signals measure ends before the turn being routed.
func historyMessagesBeforeCurrentTurn(messages []llmprotocol.Message) []llmprotocol.Message {
	if len(messages) == 0 {
		return nil
	}
	if messages[len(messages)-1].Role == llmprotocol.RoleUser {
		return messages[:len(messages)-1]
	}
	return messages
}

// contentBytes counts the bytes whose token size the history estimate charges
// to one content list: display text plus tool-call arguments plus the payload
// of tool-result blocks. semanticText alone reads only display text, so an
// agent session's tool loop — the bulk of its history — contributed nothing to
// the estimate. Counting bytes rather than materialising the join keeps the
// uncapped estimate allocation-free on long histories.
func contentBytes(contents []llmprotocol.Content) int {
	bytes, parts := 0, 0
	var walk func(contents []llmprotocol.Content)
	walk = func(contents []llmprotocol.Content) {
		for _, content := range contents {
			if content.Text != "" {
				bytes += len(content.Text)
				parts++
			}
			if content.ToolCall != nil && content.ToolCall.Arguments != "" {
				bytes += len(content.ToolCall.Arguments)
				parts++
			}
			if content.ToolResult != nil {
				walk(content.ToolResult.Content)
			}
		}
	}
	walk(contents)
	if parts > 1 {
		bytes += parts - 1 // the single-space separators a join would insert
	}
	return bytes
}

func estimateTokenLength(text string) int {
	return len(text) / 4
}
