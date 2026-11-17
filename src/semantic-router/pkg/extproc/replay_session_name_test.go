package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The shapes Claude Code 2.1.325 and Codex 0.166.0 were captured sending:
// Claude Code asks for a JSON object of one title field on an auxiliary
// request, and asks for one name field when /rename has no argument; Codex
// starts a thread whose turn metadata says thread_title.
const (
	claudeTitleSchema = `{"type":"json_schema","schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}`
	claudeNameSchema  = `{"type":"json_schema","schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}}`
)

func claudeSideRequest(format string) []byte {
	return []byte(`{"model":"vllm-sr/fast","max_tokens":4000,"messages":[{"role":"user","content":[{"type":"text","text":"<session>\nfix average\n</session>"}]}],"output_config":{"format":` + format + `}}`)
}

func codexThreadRequest(source string) []byte {
	return []byte(`{"model":"vllm-sr/auto","input":"Generate a title","client_metadata":{"x-codex-turn-metadata":"{\"session_id\":\"title-thread\",\"thread_id\":\"title-thread\",\"turn_id\":\"turn-1\",\"thread_source\":\"` + source + `\"}"}}`)
}

// A session-title request is recognised by what the harness marks, not by
// its prompt: Claude Code's auxiliary class with a title-only schema, or
// Codex's thread_title thread.
func TestSessionTitleRequestIsRecognisedAtIngress(t *testing.T) {
	router := &OpenAIRouter{Config: reviewEntrypointConfig()}
	for name, test := range map[string]struct {
		format  llmprotocol.WireFormat
		class   string
		body    []byte
		isTitle bool
	}{
		"claude code title":             {llmprotocol.AnthropicMessagesV1, "auxiliary", claudeSideRequest(claudeTitleSchema), true},
		"claude code rename generator":  {llmprotocol.AnthropicMessagesV1, "auxiliary", claudeSideRequest(claudeNameSchema), false},
		"title schema on a main turn":   {llmprotocol.AnthropicMessagesV1, "main", claudeSideRequest(claudeTitleSchema), false},
		"codex title thread":            {llmprotocol.OpenAIResponsesV1, "", codexThreadRequest("thread_title"), true},
		"codex thread the user started": {llmprotocol.OpenAIResponsesV1, "", codexThreadRequest("user"), false},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: test.format, RequestID: "title-ingress", TraceContext: t.Context()}
			ctx.Headers = map[string]string{headers.XClaudeCodeRequestClass: test.class}
			_, immediate := router.prepareProtocolRequest(test.body, ctx)
			require.Nil(t, immediate)
			require.Equal(t, test.isTitle, ctx.SessionTitleRequest)
		})
	}
}

func startSessionReplay(t *testing.T, capture string, ctxHeaders map[string]string, prepare func(*RequestContext)) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": capture, "capture_request_body": true, "capture_response_body": true}
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest, ctx.RequestID, ctx.Headers = false, "session-name", ctxHeaders
	prepare(ctx)
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/fast", "small", "")
	return router, ctx
}

// A Claude Code title request's answer names its session, buffered or
// streamed; the name is response text, kept only where Replay keeps it. A
// Codex title thread names a thread no field of it identifies, so its
// answer names nothing.
func TestSessionTitleAnswerNamesTheSession(t *testing.T) {
	claude := map[string]string{headers.XClaudeCodeSessionID: "cc-session"}
	codex := map[string]string{headers.CodexSessionID: "title-thread"}
	title := func(ctx *RequestContext) { ctx.SessionTitleRequest = true }
	codexTitle := func(ctx *RequestContext) {
		ctx.SessionTitleRequest, ctx.AgentTurn.ThreadSource = true, "thread_title"
	}
	for name, test := range map[string]struct {
		capture string
		headers map[string]string
		prepare func(*RequestContext)
		reply   func(*testing.T, *OpenAIRouter, *RequestContext)
		want    string
	}{
		"buffered": {config.PromptCaptureRaw, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(`{"title": " Average helper "}`), "")
		}, "Average helper"},
		"streamed in fragments": {config.PromptCaptureRaw, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			ctx.ExpectStreamingResponse = true
			chatStream(t, r, ctx, []string{`{"ti`, `tle":"Average`, ` helper"}`}, true)
		}, "Average helper"},
		"answer outside the schema": {config.PromptCaptureRaw, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer("Average helper"), "")
		}, ""},
		"without raw capture": {config.PromptCaptureNone, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(`{"title":"Average helper"}`), "")
		}, ""},
		"codex title thread": {config.PromptCaptureRaw, codex, codexTitle, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(`{"title":"Add median"}`), "")
		}, ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, test.capture, test.headers, test.prepare)
			test.reply(t, router, ctx)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, store.RequestKindSessionTitle, record.RequestKind)
			require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
			require.Equal(t, test.want, record.SessionName)
		})
	}
}

func renameEcho(name string) []llmprotocol.Content {
	text := func(value string) llmprotocol.Content {
		return llmprotocol.Content{Kind: llmprotocol.ContentText, Text: value}
	}
	return []llmprotocol.Content{
		text("<command-name>/rename</command-name>\n            <command-message>rename</command-message>\n            <command-args>" + name + "</command-args>\n"),
		text("<local-command-stdout>Session renamed to: " + name + "</local-command-stdout>\n"),
	}
}

// Claude Code tells the gateway of a /rename only in the history of the
// next request, as the command and its output; the last one names the
// session. The text is request text, kept only where Replay keeps it, and
// the same words from another client or without the command name nothing.
func TestSessionRenameNamesTheSession(t *testing.T) {
	claude := map[string]string{headers.XClaudeCodeSessionID: "cc-session"}
	history := func(messages ...[]llmprotocol.Content) func(*RequestContext) {
		return func(ctx *RequestContext) {
			for _, content := range messages {
				ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, llmprotocol.Message{Role: llmprotocol.RoleUser, Content: content})
			}
		}
	}
	stdoutOnly := renameEcho("Typed by hand")[1:]
	for name, test := range map[string]struct {
		capture string
		headers map[string]string
		prepare func(*RequestContext)
		want    string
	}{
		"last rename":         {config.PromptCaptureRaw, claude, history(renameEcho("Average helper"), renameEcho("notes-py-average")), "notes-py-average"},
		"without the command": {config.PromptCaptureRaw, claude, history(stdoutOnly), ""},
		"without raw capture": {config.PromptCaptureNone, claude, history(renameEcho("Average helper")), ""},
		"another client":      {config.PromptCaptureRaw, map[string]string{headers.XSessionID: "s"}, history(renameEcho("Average helper")), ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, test.capture, test.headers, test.prepare)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Empty(t, record.RequestKind)
			require.Equal(t, test.want, record.SessionName)
		})
	}
}
