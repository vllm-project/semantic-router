package extproc

import (
	"fmt"
	"strings"
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

// sessionNamesOff is a capture that keeps no text and turns session names off.
const sessionNamesOff = "names off"

func startSessionReplay(t *testing.T, capture string, ctxHeaders map[string]string, prepare func(*RequestContext)) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": capture, "capture_request_body": true, "capture_response_body": true}
	if capture == sessionNamesOff {
		plugin["prompt_capture"], plugin["capture_session_name"] = config.PromptCaptureNone, false
	}
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
// streamed, whatever the prompt capture, unless session names are off. A
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
		"under the default capture": {config.PromptCaptureNone, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(`{"title":"Average helper"}`), "")
		}, "Average helper"},
		"with session names off": {sessionNamesOff, claude, title, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
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
// session, whatever the prompt capture, unless session names are off. The
// same words from another client or without the command name nothing.
func TestSessionRenameNamesTheSession(t *testing.T) {
	claude := map[string]string{headers.XClaudeCodeSessionID: "cc-session"}
	// a conversation under way, so its first prompt does not name it
	history := func(messages ...[]llmprotocol.Content) func(*RequestContext) {
		return func(ctx *RequestContext) {
			ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, llmprotocol.Message{
				Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Done."}},
			})
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
		"last rename":               {config.PromptCaptureRaw, claude, history(renameEcho("Average helper"), renameEcho("notes-py-average")), "notes-py-average"},
		"without the command":       {config.PromptCaptureRaw, claude, history(stdoutOnly), ""},
		"under the default capture": {config.PromptCaptureNone, claude, history(renameEcho("Average helper")), "Average helper"},
		"with session names off":    {sessionNamesOff, claude, history(renameEcho("Average helper")), ""},
		"another client":            {config.PromptCaptureRaw, map[string]string{headers.XSessionID: "s"}, history(renameEcho("Average helper")), ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, test.capture, test.headers, test.prepare)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Empty(t, record.RequestKind)
			require.Equal(t, test.want, record.SessionName)
			require.False(t, record.SessionNameFromPrompt)
		})
	}
}

// A request that is no turn of its agent keeps its kind as the client marks
// it: a Claude Code auxiliary or compaction request by its class, a check by
// x-vsr-probe. A title request keeps its own kind and a main turn none.
func TestSideRequestKindIsRecorded(t *testing.T) {
	class := func(value string) map[string]string {
		return map[string]string{headers.XClaudeCodeSessionID: "cc-session", headers.XClaudeCodeRequestClass: value}
	}
	for name, test := range map[string]struct {
		headers map[string]string
		title   bool
		want    string
	}{
		"main turn":     {class("main"), false, ""},
		"auxiliary":     {class("auxiliary"), false, store.RequestKindAuxiliary},
		"compaction":    {class("Compaction"), false, store.RequestKindCompaction},
		"title request": {class("auxiliary"), true, store.RequestKindSessionTitle},
		"connect check": {map[string]string{headers.VSRProbe: "connect"}, false, store.RequestKindProbe},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, config.PromptCaptureNone, test.headers, func(ctx *RequestContext) {
				ctx.SessionTitleRequest = test.title
			})
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, test.want, record.RequestKind)
		})
	}
}

// A headless run, such as claude -p or codex exec, gets no title: its main
// agent's first prompt, shortened, names it. A later turn, a subagent's first
// turn, a side request and a configuration with session names off name
// nothing.
func TestFirstPromptNamesARunWithoutATitle(t *testing.T) {
	long := "Fix the average helper in notes.py <system-reminder>Today is Monday.</system-reminder>\n" +
		"and add a test " + strings.Repeat("for every case ", 12)
	prompt := func(extra ...llmprotocol.Message) func(*RequestContext) {
		return func(ctx *RequestContext) {
			messages := append([]llmprotocol.Message{}, extra...)
			messages = append(messages, llmprotocol.Message{
				Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: long}},
			})
			ctx.SemanticRequest.Messages = messages
		}
	}
	answered := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Done."}}}
	main := map[string]string{headers.XClaudeCodeSessionID: "cc-session"}
	subagent := map[string]string{headers.XClaudeCodeSessionID: "cc-session", headers.XClaudeCodeAgentID: "explore-1"}
	side := map[string]string{headers.XClaudeCodeSessionID: "cc-session", headers.XClaudeCodeRequestClass: "auxiliary"}
	// cut to 120 characters, its system reminder and line break gone
	named := "Fix the average helper in notes.py and add a test for every case for every case for every case for every case for every…"
	for name, test := range map[string]struct {
		capture string
		headers map[string]string
		prepare func(*RequestContext)
		want    string
	}{
		"first turn":             {config.PromptCaptureNone, main, prompt(), named},
		"later turn":             {config.PromptCaptureNone, main, prompt(answered), ""},
		"subagent's first turn":  {config.PromptCaptureNone, subagent, prompt(), ""},
		"side request":           {config.PromptCaptureNone, side, prompt(), ""},
		"with session names off": {sessionNamesOff, main, prompt(), ""},
		"another client":         {config.PromptCaptureNone, map[string]string{headers.XSessionID: "s"}, prompt(), ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, test.capture, test.headers, test.prepare)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, test.want, record.SessionName)
			require.Equal(t, test.want != "", record.SessionNameFromPrompt)
			require.LessOrEqual(t, len([]rune(record.SessionName)), maxSessionNameRunes)
		})
	}
}

// Claude Code drops a first turn the person stops while it thinks, so the
// prompt after it is a first prompt too. A name the harness gave still names
// the run over both, sent before or after them, a later /rename over an
// earlier title, and without one the earliest first prompt does.
func TestHarnessNameOutranksEveryFirstPrompt(t *testing.T) {
	claude := map[string]string{headers.XClaudeCodeSessionID: "cc-dropped"}
	user := func(text string) func(*RequestContext) {
		return func(ctx *RequestContext) {
			ctx.SemanticRequest.Messages = []llmprotocol.Message{{
				Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
			}}
		}
	}
	answered := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Hi there, friend."}}}
	title := func(ctx *RequestContext) { ctx.SessionTitleRequest = true }
	essay, hi := user("Write a 700-word essay about rivers."), user("Say hi in three words.")
	renamed := func(ctx *RequestContext) {
		hi(ctx)
		ctx.SemanticRequest.Messages = append(ctx.SemanticRequest.Messages, answered, llmprotocol.Message{
			Role: llmprotocol.RoleUser, Content: append(renameEcho("River notes"), llmprotocol.Content{Kind: llmprotocol.ContentText, Text: "Thanks."}),
		})
	}
	for name, test := range map[string]struct {
		requests []func(*RequestContext)
		want     string
	}{
		"title first":     {[]func(*RequestContext){title, essay, hi}, "Essay about rivers"},
		"title last":      {[]func(*RequestContext){essay, hi, title}, "Essay about rivers"},
		"renamed later":   {[]func(*RequestContext){title, essay, hi, renamed}, "River notes"},
		"no harness name": {[]func(*RequestContext){essay, hi}, "Write a 700-word essay about rivers."},
	} {
		t.Run(name, func(t *testing.T) {
			router, first := startSessionReplay(t, config.PromptCaptureNone, claude, test.requests[0])
			titled := func(ctx *RequestContext) {
				if ctx.SessionTitleRequest {
					runBufferedReply(t, router, ctx, 200, chatAnswer(`{"title":"Essay about rivers"}`), "")
				}
			}
			titled(first)
			for index, prepare := range test.requests[1:] {
				ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
				ctx.LooperRequest, ctx.RequestID, ctx.Headers = false, fmt.Sprintf("turn-%d", index+1), claude
				prepare(ctx)
				ctx.RouterReplayPluginConfig = first.RouterReplayPluginConfig
				router.startRouterReplay(ctx, "vllm-sr/fast", "small", "")
				titled(ctx)
			}
			var run routerreplay.RunResponse
			require.Equal(t, 200, replayAPIJSON(t, router, "/api/v1/observability/replays/run?session_id=cc-dropped", &run))
			require.Equal(t, test.want, run.Run.Name)
		})
	}
}

// Codex 0.166.0 titles a session on a thread of its own, whose request
// carries the first prompt after "User prompt:". The same prompt from the
// same client, just after its run's first turn, files the title thread under
// that run and its answer names it. Another client, another prompt or session
// names off leave it on its own thread, naming nothing.
func TestCodexTitleThreadNamesTheRunItTitles(t *testing.T) {
	first := "Add a median helper to notes.py"
	user := func(text string) func(*RequestContext) {
		return func(ctx *RequestContext) {
			ctx.SemanticRequest.Messages = []llmprotocol.Message{{
				Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
			}}
		}
	}
	codex := func(session, account string) map[string]string {
		return map[string]string{headers.CodexSessionID: session, "chatgpt-account-id": account, headers.Authorization: "Bearer login-" + account}
	}
	// the run's first turn, which Codex sends before its title thread
	main, mainCtx := startSessionReplay(t, config.PromptCaptureNone, codex("run-1", "acct-a"), user(first))
	mainRecord, ok := main.ReplayRecorder.GetRecord(mainCtx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, first, mainRecord.SessionName, "a first prompt names its run until a title does")

	titlePrompt := func(prompt string) func(*RequestContext) {
		return func(ctx *RequestContext) {
			user("Generate a concise, single-line task title.\n\nUser prompt:\n" + prompt)(ctx)
			ctx.SessionTitleRequest, ctx.AgentTurn.ThreadSource = true, "thread_title"
		}
	}
	for name, test := range map[string]struct {
		capture string
		account string
		prompt  string
		session string
		want    string
	}{
		"its run's title":         {config.PromptCaptureNone, "acct-a", first, "run-1", "Add median helper"},
		"its run's title as JSON": {config.PromptCaptureNone, "acct-a", first, "run-1", "Locate the median helper"},
		"another client":          {config.PromptCaptureNone, "acct-b", first, "title-thread", ""},
		"another prompt":          {config.PromptCaptureNone, "acct-a", "Something else", "title-thread", ""},
		"with session names off":  {sessionNamesOff, "acct-a", first, "title-thread", ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, test.capture, codex("title-thread", test.account), titlePrompt(test.prompt))
			answer := "Add median helper"
			if test.want == "Locate the median helper" {
				// as Codex 0.166.0 answers, with the structured output it asks for
				answer = `{"title":"Locate the median helper"}`
			}
			runBufferedReply(t, router, ctx, 200, chatAnswer(answer), "")
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, store.RequestKindSessionTitle, record.RequestKind)
			require.Equal(t, test.session, record.SessionID)
			require.Equal(t, test.want, record.SessionName)
		})
	}
}

// A capture that keeps no text still keeps a run's tool history: each tool
// call's name and call id, its result's status and the subagent's kind, but
// no argument, result or prompt text. capture_tool_names: false keeps none.
func TestDefaultCaptureKeepsToolNamesWithoutText(t *testing.T) {
	failed := true
	subagent := map[string]string{
		headers.XClaudeCodeSessionID: "cc-session", headers.XClaudeCodeAgentID: "explore-1", headers.XClaudeCodeAgentType: "Explore",
	}
	turn := func(ctx *RequestContext) {
		ctx.SemanticRequest.Messages = []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "private prompt"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
				ID: "call-1", Name: "Bash", Arguments: `{"command":"cat private.txt"}`,
			}}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{
				CallID: "call-1", IsError: &failed, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "private output"}},
			}}}},
		}
	}
	for name, test := range map[string]struct {
		capture string
		keeps   bool
	}{
		"default capture":   {config.PromptCaptureNone, true},
		"tool names off":    {"tool names off", false},
		"session names off": {sessionNamesOff, true},
	} {
		t.Run(name, func(t *testing.T) {
			capture := test.capture
			if capture == "tool names off" {
				capture = config.PromptCaptureNone
			}
			router, ctx := startToolNamesReplay(t, capture, test.capture == "tool names off", subagent, turn)
			ctx.SemanticResponse = &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{
					ID: "call-2", Name: "Read", Arguments: `{"file_path":"/private/notes.py"}`,
				}}},
			}}}
			router.attachRouterReplayResponse(ctx, []byte(`{"text":"ok"}`), true)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			if !test.keeps {
				require.Nil(t, record.ToolTrace)
				require.Empty(t, record.Agent.Kind)
				return
			}
			require.Equal(t, "Explore", record.Agent.Kind, "the subagent type Claude Code sends")
			require.NotNil(t, record.ToolTrace)
			var calls []string
			for _, step := range record.ToolTrace.Steps {
				require.Empty(t, step.Text+step.Arguments+step.RawArguments+step.Output+step.RawOutput, "no text: %+v", step)
				calls = append(calls, step.Source+" "+step.Type+" "+step.ToolName+" "+step.ToolCallID+" "+step.Status)
			}
			require.Equal(t, []string{
				"request assistant_tool_call Bash call-1 ",
				"request client_tool_result  call-1 failed",
				"response assistant_tool_call Read call-2 ",
			}, calls)
		})
	}
}

func startToolNamesReplay(t *testing.T, capture string, off bool, ctxHeaders map[string]string, prepare func(*RequestContext)) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	if !off {
		return startSessionReplay(t, capture, ctxHeaders, prepare)
	}
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": capture, "capture_tool_names": false}
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest, ctx.RequestID, ctx.Headers = false, "tool-names", ctxHeaders
	prepare(ctx)
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/fast", "small", "")
	return router, ctx
}

// A masking route keeps no tool history, as it keeps no request text.
func TestToolNamesAreNotKeptOnAMaskingRoute(t *testing.T) {
	ctx := &RequestContext{RouterReplayPluginConfig: &config.RouterReplayPluginConfig{Enabled: true}}
	require.True(t, replayKeepsToolNames(ctx))
	ctx.VSRSelectedDecision = &config.Decision{Plugins: []config.DecisionPlugin{{
		Type:          config.DecisionPluginMasking,
		Configuration: config.MustStructuredPayload(map[string]interface{}{"enabled": true, "signal": "personal"}),
	}}}
	require.False(t, replayKeepsToolNames(ctx))
}

// Under the default capture a turn that answers after tool calls still reads
// as the run's final answer: its answer's step is kept without its text.
func TestDefaultCaptureKeepsTheFinalAnswerStep(t *testing.T) {
	main := map[string]string{headers.XClaudeCodeSessionID: "cc-session"}
	history := func(ctx *RequestContext) {
		ctx.SemanticRequest.Messages = []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "private prompt"}}},
			{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: "Read"}}}},
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{CallID: "call-1"}}}},
		}
	}
	router, ctx := startSessionReplay(t, config.PromptCaptureNone, main, history)
	ctx.SemanticResponse = &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
		Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "private answer"}},
	}}}
	router.attachRouterReplayResponse(ctx, []byte(`{"text":"ok"}`), true)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.True(t, replayEndsWithFinalAnswer(record))
	for _, step := range record.ToolTrace.Steps {
		require.Empty(t, step.Text, "no text: %+v", step)
	}
}
