package extproc

import (
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Cursor sends the calls of a custom OpenAI base URL from its servers with
// User-Agent Cursor/<version>. Its desktop's built-in browser carries
// Cursor/<version> inside a browser agent, which is not the harness, so only
// an agent that starts with it counts.
const cursorUserAgentPrefix = "cursor/"

// OpenClaw sends the generic agent of the OpenAI SDK it calls through and
// opens its system prompt with an HTML comment that names it, such as
// <!-- openclaw:attempt:STABLE -->. Its compaction carries no such comment:
// a tool-less call whose system prompt is its summary instructions, with its
// session in prompt_cache_key and that SDK's agent.
const (
	openClawInstructionPrefix = "<!-- openclaw:"
	openClawUserAgentPrefix   = "openai/js "
)

func isCursor(ctx *RequestContext) bool {
	return strings.HasPrefix(strings.ToLower(strings.TrimSpace(headerValueCI(ctx, headers.UserAgent))), cursorUserAgentPrefix)
}

func isOpenClaw(ctx *RequestContext) bool {
	if ctx == nil || ctx.SemanticRequest == nil {
		return false
	}
	request := ctx.SemanticRequest
	instructions := leadingInstructionText(request)
	if strings.HasPrefix(strings.ToLower(instructions), openClawInstructionPrefix) {
		return true
	}
	return len(request.Tools) == 0 && strings.HasPrefix(instructions, openClawCompactionPrompt) && openClawSession(ctx) != "" &&
		strings.HasPrefix(strings.ToLower(strings.TrimSpace(headerValueCI(ctx, headers.UserAgent))), openClawUserAgentPrefix)
}

// maxOpenClawSessionBytes bounds the session OpenClaw names in its
// prompt_cache_key.
const maxOpenClawSessionBytes = 128

// openClawSession is the session OpenClaw names in prompt_cache_key when its
// model entry turns prompt cache keys on: <session>:<compaction generation>
// on an agent turn, the bare session on a compaction. Empty for any other
// request, or a key too long to be a session id.
func openClawSession(ctx *RequestContext) string {
	if ctx == nil || ctx.SemanticRequest == nil {
		return ""
	}
	session, _, _ := strings.Cut(strings.TrimSpace(ctx.SemanticRequest.PromptCacheKey), ":")
	if session == "" || len(session) > maxOpenClawSessionBytes {
		return ""
	}
	return "openclaw-" + session
}

// derivedConversationID is the session Router Replay files a call of a
// harness that names none under. The conversation-start fingerprint reads
// only the first characters of the first user message, and Cursor opens that
// message with the same workspace preamble in every conversation, so this id
// reads the whole first user message, with the start of the system
// instructions, the request's user field and the authenticated tenant, keyed
// as that fingerprint is. Every call of one conversation resends them; only
// the hash is kept, and only Replay, tracing and run budgets read it, never
// protection or routing. A compaction that rewrites the conversation's opening starts a new
// id.
func derivedConversationID(ctx *RequestContext, harness string) string {
	if ctx == nil || ctx.SemanticRequest == nil {
		return ""
	}
	request := ctx.SemanticRequest
	system, user := leadingInstructionText(request), firstUserText(request)
	if system == "" && user == "" {
		return ""
	}
	var encoded strings.Builder
	for _, part := range []string{
		harness, strings.TrimSpace(headerValueCI(ctx, headers.AuthzTenantID)), request.EndUserID,
		leadingRunes(system, conversationStartRunes), user,
	} {
		fmt.Fprintf(&encoded, "%d:%s\n", len(part), part)
	}
	return harness + "-start-" + hex.EncodeToString(conversationStartDigest(encoded.String()))[:16]
}

// firstUserText is the text of a request's first user message.
func firstUserText(request *llmprotocol.Request) string {
	for _, message := range request.Messages {
		if message.Role == llmprotocol.RoleUser {
			return strings.TrimSpace(semanticText(message.Content))
		}
	}
	return ""
}

// requestImageCounts counts the images of a request: in its newest turn, the
// messages after the last assistant message (a picture the user attached or
// a tool returned), and in the messages before it, which a later call of the
// conversation keeps resending.
func requestImageCounts(request *llmprotocol.Request) (turn int, history int) {
	if request == nil {
		return 0, 0
	}
	lastAssistant := -1
	for i, message := range request.Messages {
		if message.Role == llmprotocol.RoleAssistant {
			lastAssistant = i
		}
	}
	for i, message := range request.Messages {
		count := countImages(message.Content)
		if i > lastAssistant {
			turn += count
		} else {
			history += count
		}
	}
	return turn, history
}

func countImages(contents []llmprotocol.Content) int {
	count := 0
	for _, content := range contents {
		switch content.Kind {
		case llmprotocol.ContentImage:
			count++
		case llmprotocol.ContentToolResult:
			if content.ToolResult != nil {
				count += countImages(content.ToolResult.Content)
			}
		}
	}
	return count
}
