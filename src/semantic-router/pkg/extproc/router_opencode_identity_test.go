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

// The identity OpenCode 1.18.10 was captured sending through its
// OpenAI-compatible provider: its User-Agent and the session in x-session-id
// (and x-session-affinity) on every call. A subagent's calls carry the child
// session it runs in as x-session-id and the session that spawned it as
// x-parent-session-id. Its session-title and compaction calls carry no tools
// and open with its built-in instructions.
const openCodeTestUserAgent = "opencode/1.18.10 ai-sdk/provider-utils/4"

func openCodeHeaders(session, parent string) map[string]string {
	values := map[string]string{headers.UserAgent: openCodeTestUserAgent, headers.XSessionID: session, "x-session-affinity": session}
	if parent != "" {
		values[headers.XParentSessionID] = parent
	}
	return values
}

const (
	openCodeMainCall       = `{"model":"vllm-sr/auto","max_tokens":32000,"messages":[{"role":"system","content":"You are opencode, an interactive CLI tool that helps users with software engineering tasks."},{"role":"user","content":"Read textstats.py"}],"tools":[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}`
	openCodeTitleCall      = `{"model":"vllm-sr/auto","max_tokens":32000,"messages":[{"role":"system","content":"You are a title generator. You output ONLY a thread title. Nothing else."},{"role":"user","content":"Generate a title for this conversation:"},{"role":"user","content":"Read textstats.py"}]}`
	openCodeCompactionCall = `{"model":"vllm-sr/auto","max_tokens":32000,"messages":[{"role":"system","content":"You are an anchored context summarization assistant for coding sessions.\n\nSummarize only the conversation history."},{"role":"user","content":"Summarize the session so far."}]}`
	openCodeTitleWithTools = `{"model":"vllm-sr/auto","max_tokens":32000,"messages":[{"role":"system","content":"You are a title generator. You output ONLY a thread title."},{"role":"user","content":"Read textstats.py"}],"tools":[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object"}}}]}`
)

// With no identity headers configured, protection, Replay and Insights read
// OpenCode's session and key each subagent as an agent of the session that
// spawned it. A pinned session still wins, and another client's
// x-parent-session-id names nothing.
func TestOpenCodeIdentityWithoutConfig(t *testing.T) {
	for _, tc := range []struct {
		name         string
		headers      map[string]string
		session      string
		agent        string
		agentHeader  string
		insightAgent bool
	}{
		{name: "main session", headers: openCodeHeaders("ses_main", ""), session: "ses_main"},
		{
			name: "subagent session", headers: openCodeHeaders("ses_child", "ses_main"),
			session: "ses_main", agent: "ses_child", agentHeader: headers.XSessionID, insightAgent: true,
		},
		{
			name: "a pinned session comes first",
			headers: func() map[string]string {
				values := openCodeHeaders("ses_child", "ses_main")
				values[headers.VSRSessionID] = "pinned"
				return values
			}(),
			session: "pinned", agent: "ses_child", agentHeader: headers.XSessionID, insightAgent: true,
		},
		{
			name:    "another client's parent header",
			headers: map[string]string{headers.XSessionID: "s-child", headers.XParentSessionID: "s-main"},
			session: "s-child",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &config.RouterConfig{}
			cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
			r := &OpenAIRouter{Config: cfg}
			ctx := &RequestContext{Headers: tc.headers}
			identity, ok := r.protectionIdentity(ctx, cfg.RouterLearning.Protection)
			require.True(t, ok)
			require.Equal(t, tc.session, identity.sessionID)
			require.Equal(t, tc.agent, identity.agentID)
			require.Equal(t, tc.agentHeader, identity.agentHeader)
			record := routerreplay.RoutingRecord{SessionID: "derived-session"}
			r.populateReplayIdentity(&record, ctx)
			require.Equal(t, tc.session, record.SessionID)
			record.SessionPolicy = map[string]interface{}{"identity": newRouterLearningIdentityDiagnostics(identity).toPolicyMap()}
			require.Equal(t, tc.insightAgent, agentOf(record) != "")
		})
	}
}

// Configured identity headers replace the defaults for OpenCode too: its
// subagent is no agent of the parent session then.
func TestOpenCodeIdentityConfiguredHeadersReplaceTheDefaults(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
	cfg.RouterLearning.Protection.Identity.Headers.Session = stringPtr("x-tenant-session")
	cfg.RouterLearning.Protection.Identity.Headers.Agent = stringPtr("x-tenant-agent")
	r := &OpenAIRouter{Config: cfg}
	values := openCodeHeaders("ses_child", "ses_main")
	values["x-tenant-session"] = "tenant-session"
	identity, ok := r.protectionIdentity(&RequestContext{Headers: values}, cfg.RouterLearning.Protection)
	require.True(t, ok)
	require.Equal(t, "tenant-session", identity.sessionID)
	require.Empty(t, identity.agentID)
	require.Equal(t, "x-tenant-agent", identity.agentHeader)
}

// The agent signal names OpenCode, its subagents, and the class of each
// call: its session title is auxiliary and its summary compaction, read from
// the built-in instructions of a call without tools.
func TestOpenCodeAgentFacts(t *testing.T) {
	router := &OpenAIRouter{Config: reviewEntrypointConfig()}
	for name, test := range map[string]struct {
		headers map[string]string
		body    string
		harness string
		role    string
		class   string
	}{
		"main call":                 {openCodeHeaders("ses_main", ""), openCodeMainCall, config.AgentHarnessOpenCode, config.AgentRoleMain, config.AgentRequestClassMain},
		"subagent call":             {openCodeHeaders("ses_child", "ses_main"), openCodeMainCall, config.AgentHarnessOpenCode, config.AgentRoleSubagent, config.AgentRequestClassSubagent},
		"session title":             {openCodeHeaders("ses_main", ""), openCodeTitleCall, config.AgentHarnessOpenCode, config.AgentRoleMain, config.AgentRequestClassAuxiliary},
		"compaction":                {openCodeHeaders("ses_main", ""), openCodeCompactionCall, config.AgentHarnessOpenCode, config.AgentRoleMain, config.AgentRequestClassCompaction},
		"subagent compaction":       {openCodeHeaders("ses_child", "ses_main"), openCodeCompactionCall, config.AgentHarnessOpenCode, config.AgentRoleSubagent, config.AgentRequestClassCompaction},
		"title prompt with tools":   {openCodeHeaders("ses_main", ""), openCodeTitleWithTools, config.AgentHarnessOpenCode, config.AgentRoleMain, config.AgentRequestClassMain},
		"same title from elsewhere": {map[string]string{headers.XSessionID: "s"}, openCodeTitleCall, "", config.AgentRoleMain, ""},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, Headers: test.headers, RequestID: "opencode-facts", TraceContext: t.Context()}
			_, immediate := router.prepareProtocolRequest([]byte(test.body), ctx)
			require.Nil(t, immediate)
			facts := agentFacts(ctx, config.RouterLearningProtectionConfig{})
			require.Equal(t, test.harness, facts.Harness)
			require.Equal(t, test.role, facts.Role)
			require.Equal(t, test.class, facts.RequestClass)
			require.Empty(t, facts.Type)
		})
	}
}

// OpenCode's title call is a session-title request at ingress, and only
// OpenCode's: the same call from another client, or with tools, is not.
func TestOpenCodeTitleRequestIsRecognisedAtIngress(t *testing.T) {
	router := &OpenAIRouter{Config: reviewEntrypointConfig()}
	for name, test := range map[string]struct {
		headers map[string]string
		body    string
		isTitle bool
	}{
		"opencode title":          {openCodeHeaders("ses_main", ""), openCodeTitleCall, true},
		"opencode main call":      {openCodeHeaders("ses_main", ""), openCodeMainCall, false},
		"opencode compaction":     {openCodeHeaders("ses_main", ""), openCodeCompactionCall, false},
		"title prompt with tools": {openCodeHeaders("ses_main", ""), openCodeTitleWithTools, false},
		"another client":          {map[string]string{headers.XSessionID: "s"}, openCodeTitleCall, false},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, Headers: test.headers, RequestID: "opencode-title", TraceContext: t.Context()}
			_, immediate := router.prepareProtocolRequest([]byte(test.body), ctx)
			require.Nil(t, immediate)
			require.Equal(t, test.isTitle, ctx.SessionTitleRequest)
		})
	}
}

// OpenCode answers its title request with the title alone, as plain text;
// its first line names the session, buffered or streamed, without any
// reasoning the model printed before it.
func TestOpenCodeTitleAnswerNamesTheSession(t *testing.T) {
	title := func(ctx *RequestContext) { ctx.SessionTitleRequest = true }
	for name, test := range map[string]struct {
		reply func(*testing.T, *OpenAIRouter, *RequestContext)
		want  string
	}{
		"buffered": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(" Fix word counts \n"), "")
		}, "Fix word counts"},
		"reasoning before the title": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer("<think>A short title.</think>Fix word counts"), "")
		}, "Fix word counts"},
		"streamed in fragments": {func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			ctx.ExpectStreamingResponse = true
			chatStream(t, r, ctx, []string{"Fix word", " counts"}, true)
		}, "Fix word counts"},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startSessionReplay(t, config.PromptCaptureRaw, openCodeHeaders("ses_main", ""), title)
			test.reply(t, router, ctx)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, store.RequestKindSessionTitle, record.RequestKind)
			require.Equal(t, test.want, record.SessionName)
		})
	}
}

// Replay files every call of an OpenCode run under the main session, its
// subagent's under that session as an agent the main agent spawned.
func TestReplayFilesOpenCodeCallsUnderTheirRun(t *testing.T) {
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}},
		replayTestPlugin(map[string]interface{}{"enabled": true}))
	router := replayRuntimeRouter(t, cfg)
	main := recordRunCall(t, router, openCodeHeaders("ses_main", ""), llmprotocol.OpenAIChatV1)
	subagent := recordRunCall(t, router, openCodeHeaders("ses_child", "ses_main"), llmprotocol.OpenAIChatV1)
	require.Equal(t, []string{"ses_main", "ses_main"}, []string{main.SessionID, subagent.SessionID})
	require.Equal(t, &store.AgentIdentity{Harness: store.HarnessOpenCode, RequestClass: config.AgentRequestClassMain}, main.Agent)
	require.Equal(t, &store.AgentIdentity{
		Harness: store.HarnessOpenCode, ID: shortLearningIdentityHash("ses_child"), RequestClass: config.AgentRequestClassSubagent,
	}, subagent.Agent)
}
