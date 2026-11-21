package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// The User-Agent DeepSeek Harness 0.2.1-rc.5 was captured sending through
// its OpenAI-compatible provider, with no session or agent header.
const deepSeekHarnessTestUserAgent = "deepseek-harness/0.2.1-rc.5 (+https://github.com/deepseek-ai/deepseek-harness)"

func TestDeepSeekHarnessIsNamedByItsUserAgent(t *testing.T) {
	for name, test := range map[string]struct {
		headers map[string]string
		harness string
	}{
		"its user agent":           {map[string]string{headers.UserAgent: deepSeekHarnessTestUserAgent}, config.AgentHarnessDeepSeekHarness},
		"case and spacing":         {map[string]string{headers.UserAgent: "  DeepSeek-Harness/0.3.0"}, config.AgentHarnessDeepSeekHarness},
		"another deepseek client":  {map[string]string{headers.UserAgent: "deepseek-chat/1.0"}, ""},
		"opencode stays opencode":  {map[string]string{headers.UserAgent: openCodeTestUserAgent}, config.AgentHarnessOpenCode},
		"claude code headers lead": {map[string]string{headers.UserAgent: deepSeekHarnessTestUserAgent, headers.XClaudeCodeSessionID: "s1"}, config.AgentHarnessClaudeCode},
		"no user agent":            {map[string]string{}, ""},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := &RequestContext{Headers: test.headers}
			require.Equal(t, test.harness, agentHarness(ctx))
			require.Equal(t, test.harness, requestHarness(ctx))
		})
	}
}

// A DeepSeek Harness call is filed under its harness in Replay and named by
// the agent signal; with no session header it keeps the derived session and
// takes no main or subagent role.
func TestDeepSeekHarnessReplayAndAgentFacts(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.RouterLearning.Protection.Scope = config.RouterLearningScopeSession
	r := &OpenAIRouter{Config: cfg}
	ctx := &RequestContext{Headers: map[string]string{headers.UserAgent: deepSeekHarnessTestUserAgent}}

	record := routerreplay.RoutingRecord{SessionID: "derived-session"}
	r.populateReplayIdentity(&record, ctx)
	require.Equal(t, "derived-session", record.SessionID)
	require.Equal(t, &store.AgentIdentity{Harness: store.HarnessDeepSeekHarness}, record.Agent)

	facts := agentFacts(ctx, cfg.RouterLearning.Protection)
	require.Equal(t, config.AgentHarnessDeepSeekHarness, facts.Harness)
	require.Empty(t, facts.Role)
	require.Empty(t, facts.RequestClass)
}

func TestAgentRuleAcceptsTheDeepSeekHarness(t *testing.T) {
	require.NoError(t, config.ValidateAgentRuleContract(config.AgentRule{Name: "dsh", Harness: config.AgentHarnessDeepSeekHarness}))
	require.Contains(t, config.SupportedAgentHarnesses(), "deepseek_harness")
}
