package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	// The agent Cursor's servers send with the calls of a custom OpenAI base
	// URL, and the browser built into the Cursor desktop app.
	cursorTestUserAgent        = "Cursor/1.0"
	cursorBrowserTestUserAgent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Cursor/3.27.22 Chrome/142.0.0.0 Electron/39.2.3 Safari/537.36"
	// OpenClaw 2026.11.5 sends the OpenAI SDK's agent and opens its system
	// prompt with this comment.
	openClawTestUserAgent = "OpenAI/JS 7.39.0"
	openClawTestSystem    = "<!-- openclaw:attempt:STABLE -->\nYou are a personal assistant running inside OpenClaw."
	// The system prompt of its compaction, which carries no such comment.
	openClawTestSummary = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant."
	// A one-pixel PNG.
	testImageBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)

// cursorCall is a Chat call shaped as Cursor sends one: streamed with usage,
// a user field, its agent tools, one system message and the conversation.
func cursorCall(firstUser string, followUp bool) string {
	call := `{"model":"vllm-sr/auto","stream":true,"stream_options":{"include_usage":true},"user":"cursor-account-1",` +
		`"tools":[{"type":"function","function":{"name":"Read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}},` +
		`{"type":"function","function":{"name":"Grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"}}}}}],` +
		`"messages":[{"role":"system","content":"You are an AI coding assistant. You operate in Cursor."},` +
		`{"role":"user","content":"<user_info>OS: linux</user_info> ` + firstUser + `"}`
	if followUp {
		call += `,{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"app/stats.py\"}"}}]},` +
			`{"role":"tool","tool_call_id":"call_1","content":"def median(values): ..."}`
	}
	return call + `]}`
}

func recordReplayCall(t *testing.T, router *OpenAIRouter, requestHeaders map[string]string, format llmprotocol.WireFormat, body string) store.Record {
	t.Helper()
	ctx := &RequestContext{SourceFormat: format, Headers: requestHeaders, RequestID: "call", TraceContext: t.Context()}
	_, immediate := router.prepareProtocolRequest([]byte(body), ctx)
	require.Nil(t, immediate)
	ctx.RouterReplayPluginConfig = router.Config.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/auto", "qwen", "all")
	require.NotEmpty(t, ctx.RouterReplayID)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	return record
}

func memoryReplayRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}},
		replayTestPlugin(map[string]interface{}{"enabled": true}))
	return replayRuntimeRouter(t, cfg)
}

func TestCursorAndOpenClawAreNamed(t *testing.T) {
	openClaw := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: openClawTestSystem}}},
		{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Run the tests."}}},
	}}
	instructions := &llmprotocol.Request{Instructions: []llmprotocol.InstructionBlock{
		{Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "  " + openClawTestSystem}}},
	}}
	other := &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "You talk about OpenClaw."}}},
	}}
	summary := func(key string) *llmprotocol.Request {
		return &llmprotocol.Request{PromptCacheKey: key, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: openClawTestSummary}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "The conversation."}}},
		}}
	}
	summaryWithTools := summary("3b1d9a4e")
	summaryWithTools.Tools = testTools
	for name, test := range map[string]struct {
		headers map[string]string
		request *llmprotocol.Request
		harness string
	}{
		"cursor's servers":             {map[string]string{headers.UserAgent: cursorTestUserAgent}, nil, config.AgentHarnessCursor},
		"cursor, another version":      {map[string]string{headers.UserAgent: " cursor/2.3"}, nil, config.AgentHarnessCursor},
		"cursor's built-in browser":    {map[string]string{headers.UserAgent: cursorBrowserTestUserAgent}, nil, ""},
		"openclaw's system prompt":     {map[string]string{headers.UserAgent: openClawTestUserAgent}, openClaw, config.AgentHarnessOpenClaw},
		"openclaw as instructions":     {map[string]string{}, instructions, config.AgentHarnessOpenClaw},
		"a prompt that names openclaw": {map[string]string{headers.UserAgent: openClawTestUserAgent}, other, ""},
		"openclaw's compaction":        {map[string]string{headers.UserAgent: openClawTestUserAgent}, summary("3b1d9a4e"), config.AgentHarnessOpenClaw},
		"a summary with no session":    {map[string]string{headers.UserAgent: openClawTestUserAgent}, summary(""), ""},
		"a summary from another agent": {map[string]string{headers.UserAgent: "python-httpx/0.28.1"}, summary("3b1d9a4e"), ""},
		"a summary with tools":         {map[string]string{headers.UserAgent: openClawTestUserAgent}, summaryWithTools, ""},
		"claude code headers lead":     {map[string]string{headers.UserAgent: cursorTestUserAgent, headers.XClaudeCodeSessionID: "s1"}, nil, config.AgentHarnessClaudeCode},
		"no user agent":                {map[string]string{}, nil, ""},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{Headers: test.headers, SemanticRequest: test.request}
			require.Equal(t, test.harness, agentHarness(ctx))
			require.Equal(t, test.harness, requestHarness(ctx))
		})
	}
}

func TestAgentRulesNameCursorAndOpenClaw(t *testing.T) {
	for _, harness := range []string{config.AgentHarnessCursor, config.AgentHarnessOpenClaw} {
		require.NoError(t, config.ValidateAgentRuleContract(config.AgentRule{Name: "client", Harness: harness}))
	}
	require.ErrorContains(t, config.ValidateAgentRuleContract(config.AgentRule{
		Name: "client", Harness: config.AgentHarnessCursor, RequestClasses: []string{config.AgentRequestClassCompaction},
	}), "carry no request class")
}

// Cursor names no session, so Replay files a conversation's calls under one
// id hashed from how the conversation opens, and another conversation, or
// the same opening from another account, under another; no text is kept in
// the id.
func TestReplayFilesACursorConversationUnderOneSession(t *testing.T) {
	router := memoryReplayRouter(t)
	cursor := map[string]string{headers.UserAgent: cursorTestUserAgent}
	first := recordReplayCall(t, router, cursor, llmprotocol.OpenAIChatV1, cursorCall("What does median return?", false))
	followUp := recordReplayCall(t, router, cursor, llmprotocol.OpenAIChatV1, cursorCall("What does median return?", true))
	other := recordReplayCall(t, router, cursor, llmprotocol.OpenAIChatV1, cursorCall("What does spread compute?", false))
	account := recordReplayCall(t, router, cursor, llmprotocol.OpenAIChatV1,
		strings.Replace(cursorCall("What does median return?", false), "cursor-account-1", "cursor-account-2", 1))

	require.Equal(t, store.HarnessCursor, first.Agent.Harness)
	require.True(t, strings.HasPrefix(first.SessionID, "cursor-start-"), first.SessionID)
	require.Equal(t, first.SessionID, followUp.SessionID)
	require.NotEqual(t, first.SessionID, other.SessionID)
	require.NotEqual(t, first.SessionID, account.SessionID)
	require.NotContains(t, first.SessionID, "median")
	require.Equal(t, 0, first.ImageCount)
}

// A Claude Code subagent's call keeps its type and request class, and the
// image a tool returned in its newest turn is counted apart from history.
func TestReplayRecordsAgentFactsAndImages(t *testing.T) {
	router := memoryReplayRouter(t)
	claude := map[string]string{
		headers.XClaudeCodeSessionID: "cc-run", headers.XClaudeCodeAgentID: "a1", headers.XClaudeCodeAgentType: "Explore",
		headers.XClaudeCodeRequestClass: "subagent",
	}
	image := `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + testImageBase64 + `"}}`
	body := `{"model":"vllm-sr/auto","max_tokens":8,"messages":[` +
		`{"role":"user","content":[{"type":"text","text":"Compare these."},` + image + `]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"Read","input":{"path":"shot.png"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[` + image + `]}]}]}`
	record := recordReplayCall(t, router, claude, llmprotocol.AnthropicMessagesV1, body)

	require.Equal(t, store.HarnessClaudeCode, record.Agent.Harness)
	require.Equal(t, "Explore", record.Agent.Type)
	require.Equal(t, "subagent", record.Agent.RequestClass)
	require.Equal(t, 1, record.ImageCount)
	require.Equal(t, 1, record.HistoryImageCount)
}

func TestRequestImageCountsSplitTheNewestTurn(t *testing.T) {
	text := func(role llmprotocol.Role) llmprotocol.Message {
		return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "x"}}}
	}
	picture := llmprotocol.Message{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentImage}}}
	for name, test := range map[string]struct {
		messages      []llmprotocol.Message
		turn, history int
	}{
		"a picture in the newest turn":  {[]llmprotocol.Message{text(llmprotocol.RoleUser), text(llmprotocol.RoleAssistant), picture}, 1, 0},
		"a picture only in the history": {[]llmprotocol.Message{picture, text(llmprotocol.RoleAssistant), text(llmprotocol.RoleUser)}, 0, 1},
		"no assistant yet":              {[]llmprotocol.Message{picture, picture}, 2, 0},
	} {
		t.Run(name, func(t *testing.T) {
			turn, history := requestImageCounts(&llmprotocol.Request{Messages: test.messages})
			require.Equal(t, test.turn, turn)
			require.Equal(t, test.history, history)
		})
	}
}

// A run marks the events every harness shares: compactions, housekeeping
// calls, images, tool loops, a subagent's first call, and a guard decision
// that refused with its fast response.
func TestRunCallsMarkSharedAgentEvents(t *testing.T) {
	agent := func(class string) *store.AgentIdentity {
		return &store.AgentIdentity{Harness: store.HarnessClaudeCode, RequestClass: class}
	}
	for name, test := range map[string]struct {
		record routerreplay.RoutingRecord
		marker string
		absent bool
	}{
		"a compaction":            {record: routerreplay.RoutingRecord{Agent: agent(config.AgentRequestClassCompaction)}, marker: routerreplay.RunMarkerCompaction},
		"an auxiliary call":       {record: routerreplay.RoutingRecord{Agent: agent(config.AgentRequestClassAuxiliary)}, marker: routerreplay.RunMarkerHousekeeping},
		"a session title":         {record: routerreplay.RoutingRecord{RequestKind: store.RequestKindSessionTitle}, marker: routerreplay.RunMarkerHousekeeping},
		"an image":                {record: routerreplay.RoutingRecord{ImageCount: 1}, marker: routerreplay.RunMarkerImage},
		"a guard's fast response": {record: routerreplay.RoutingRecord{JailbreakDetected: true, SelectionMethod: "fast_response", ResponseStatus: 200}, marker: routerreplay.RunMarkerBlock},
		"a plain fast response":   {record: routerreplay.RoutingRecord{SelectionMethod: "fast_response", ResponseStatus: 200}, marker: routerreplay.RunMarkerBlock, absent: true},
		"a guard with no model":   {record: routerreplay.RoutingRecord{JailbreakDetected: true, ResponseStatus: 200}, marker: routerreplay.RunMarkerBlock},
		"a flagged call answered": {record: routerreplay.RoutingRecord{JailbreakDetected: true, SelectedModel: "a", ResponseStatus: 200}, marker: routerreplay.RunMarkerBlock, absent: true},
		"a main call":             {record: routerreplay.RoutingRecord{Agent: agent(config.AgentRequestClassMain)}, marker: routerreplay.RunMarkerCompaction, absent: true},
	} {
		t.Run(name, func(t *testing.T) {
			call := buildReplayRunCall(test.record)
			if test.absent {
				require.NotContains(t, call.Markers, test.marker)
				return
			}
			require.Contains(t, call.Markers, test.marker)
		})
	}

	record := routerreplay.RoutingRecord{Agent: agent(config.AgentRequestClassMain), ImageCount: 2}
	call := buildReplayRunCall(record)
	require.Equal(t, config.AgentRequestClassMain, call.RequestClass)
	require.Equal(t, 2, call.ImageCount)
}

func TestRunMarksASubagentsFirstCallWithItsType(t *testing.T) {
	lanes := newReplayRunLanes()
	main := lanes.place(replayRunEntry{})
	first := lanes.place(replayRunEntry{agent: store.AgentIdentity{ID: "sub", Type: "Explore"}})
	second := lanes.place(replayRunEntry{agent: store.AgentIdentity{ID: "sub", Type: "Explore"}})

	require.NotContains(t, main.Markers, routerreplay.RunMarkerSubagentStart)
	require.Contains(t, first.Markers, routerreplay.RunMarkerSubagentStart)
	require.NotContains(t, second.Markers, routerreplay.RunMarkerSubagentStart)
	agents := lanes.agents(nil, []routerreplay.RunCall{main, first, second})
	require.Equal(t, "Explore", agents[1].Type)
}

// OpenCode 1.19 opens its compaction with other instructions than earlier
// releases did; both read as a compaction.
func TestOpenCodeCompactionInstructionsOfBothReleases(t *testing.T) {
	for _, instruction := range []string{
		"You are an anchored context summarization assistant for coding sessions.",
		"You are a context summarization agent. You are given a conversation between a user and an agent.",
	} {
		request := &llmprotocol.Request{Instructions: []llmprotocol.InstructionBlock{
			{Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: instruction}}},
		}}
		require.Equal(t, config.AgentRequestClassCompaction, openCodeRequestClass(request, false))
	}
}

// OpenClaw names its session in prompt_cache_key when its model entry turns
// cache keys on: the agent turns of one compaction generation and the
// compaction itself share it. Without a key the Router derives one.
func TestOpenClawSessionFromItsPromptCacheKey(t *testing.T) {
	request := func(key string) *llmprotocol.Request {
		return &llmprotocol.Request{PromptCacheKey: key, Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: openClawTestSystem}}},
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Run the tests."}}},
		}}
	}
	session := func(key string) string {
		ctx := &RequestContext{Headers: map[string]string{headers.UserAgent: openClawTestUserAgent}, SemanticRequest: request(key)}
		return harnessRunSession(ctx, config.RouterLearningProtectionConfig{})
	}
	require.Equal(t, "openclaw-3b1d9a4e", session("3b1d9a4e:0"))
	require.Equal(t, "openclaw-3b1d9a4e", session("3b1d9a4e:1"))
	require.Equal(t, "openclaw-3b1d9a4e", session("3b1d9a4e"))
	require.True(t, strings.HasPrefix(session(""), "openclaw-start-"))
	require.True(t, strings.HasPrefix(session(strings.Repeat("k", maxOpenClawSessionBytes+1)), "openclaw-start-"))
}

// OpenClaw's compaction names OpenClaw only through its summary instructions,
// its agent and the session in prompt_cache_key, so Replay files it in the
// session of the turns it summarizes, as a compaction.
func TestReplayFilesOpenClawsCompactionInItsSession(t *testing.T) {
	router := memoryReplayRouter(t)
	openClaw := map[string]string{headers.UserAgent: openClawTestUserAgent}
	turn := recordReplayCall(t, router, openClaw, llmprotocol.OpenAIChatV1,
		`{"model":"vllm-sr/auto","prompt_cache_key":"3b1d9a4e:0","messages":[`+
			`{"role":"system","content":"<!-- openclaw:attempt:STABLE -->\nYou are a personal assistant running inside OpenClaw."},`+
			`{"role":"user","content":"Run the tests."}]}`)
	compaction := recordReplayCall(t, router, openClaw, llmprotocol.OpenAIChatV1,
		`{"model":"vllm-sr/auto","prompt_cache_key":"3b1d9a4e","messages":[`+
			`{"role":"system","content":"`+openClawTestSummary+`"},`+
			`{"role":"user","content":"The conversation."}]}`)

	require.Equal(t, store.HarnessOpenClaw, compaction.Agent.Harness)
	require.Equal(t, config.AgentRequestClassCompaction, compaction.Agent.RequestClass)
	require.Contains(t, compaction.SessionID, "openclaw-3b1d9a4e")
	require.Equal(t, turn.SessionID, compaction.SessionID)
	require.Equal(t, config.AgentRequestClassMain, turn.Agent.RequestClass)
}
