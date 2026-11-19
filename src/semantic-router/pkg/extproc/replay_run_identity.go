package extproc

import (
	"cmp"
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// maxAgentKindLength bounds the subagent kind a harness names.
const maxAgentKindLength = 64

// replayRunParent is the client request a Looper hop is made for: its run
// and its request id. A hop carries none of the client's headers, so its
// record takes them from the request instead of reading a session of its own.
type replayRunParent struct {
	sessionID string
	agent     *store.AgentIdentity
	requestID string
}

type replayRunParentKey struct{}

// withReplayRunParent marks the context Looper hops run under with the run
// of the client request reqCtx.
func (r *OpenAIRouter) withReplayRunParent(ctx context.Context, reqCtx *RequestContext) context.Context {
	// The hop's record is filed where the request's own record is.
	record := store.Record{SessionID: reqCtx.SessionID}
	r.populateReplayIdentity(&record, reqCtx)
	return context.WithValue(ctx, replayRunParentKey{}, replayRunParent{
		sessionID: record.SessionID, agent: record.Agent, requestID: reqCtx.RequestID,
	})
}

func replayRunParentFrom(ctx context.Context) *replayRunParent {
	if ctx == nil {
		return nil
	}
	if parent, ok := ctx.Value(replayRunParentKey{}).(replayRunParent); ok {
		return &parent
	}
	return nil
}

// applyReplayRunParent files a Looper hop's record in its client request's
// run, under the agent that made the request.
func applyReplayRunParent(record *store.Record, parent *replayRunParent) {
	if parent == nil {
		return
	}
	record.SessionID, record.Agent, record.ParentRequestID = parent.sessionID, parent.agent, parent.requestID
}

// harnessRunSession is the session the harness names for a request's agent
// run: the one protection reads, else the one a supported harness sends,
// pinned or Codex's session-id, else for Cursor and OpenClaw, which send
// no session header, the session OpenClaw names in prompt_cache_key or the
// conversation id the Router derives. A run keeps it whichever
// identity headers protection is configured with and whether protection can
// key the call.
func harnessRunSession(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) string {
	if sessionID, _ := protectionSessionID(ctx, cfg); sessionID != "" {
		return sessionID
	}
	if sessionID, _ := protectionSessionID(ctx, config.RouterLearningProtectionConfig{}); sessionID != "" {
		return sessionID
	}
	switch harness := requestHarness(ctx); harness {
	case store.HarnessOpenClaw:
		if session := openClawSession(ctx); session != "" {
			return session
		}
		return derivedConversationID(ctx, harness)
	case store.HarnessCursor:
		return derivedConversationID(ctx, harness)
	}
	return ""
}

// harnessRunAgent is the agent of its run a request comes from, as the
// harness names it: protection's agent, else the one a supported harness
// sends, with the subagent type and request class the agent signal reads. Codex names the thread that spawned a thread; a thread its root
// spawned, and every Claude Code or OpenCode subagent, hangs off the main
// agent.
func harnessRunAgent(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) *store.AgentIdentity {
	agentID, _ := protectionAgentID(ctx, cfg)
	if agentID == "" {
		// The facts follow the identity the record keeps, so a subagent's
		// call reads as one whichever headers protection is set to read.
		cfg = config.RouterLearningProtectionConfig{}
		agentID, _ = protectionAgentID(ctx, cfg)
	}
	facts := agentFacts(ctx, cfg)
	identity := store.AgentIdentity{
		Harness:      requestHarness(ctx),
		ID:           shortLearningIdentityHash(agentID),
		ParentID:     shortLearningIdentityHash(codexParentThread(ctx)),
		RequestClass: facts.RequestClass,
	}
	if identity.Harness != store.HarnessCodex {
		identity.Type = facts.Type
	}
	if identity.ID != "" {
		// Codex names a child's kind, such as collab_spawn or review, and
		// Claude Code a subagent's type, such as Explore.
		kind := cmp.Or(headerValueCI(ctx, headers.XOpenAISubagent), headerValueCI(ctx, headers.XClaudeCodeAgentType))
		identity.Kind = boundedIdentityValue(kind, maxAgentKindLength)
	}
	if identity == (store.AgentIdentity{}) {
		return nil
	}
	return &identity
}

// requestTraceRun is the run a request's spans join when its harness sent
// no trace context: the session and agent its record is filed under, read
// from the headers alone.
func (r *OpenAIRouter) requestTraceRun(ctx *RequestContext) tracing.Run {
	cfg := config.RouterLearningProtectionConfig{}
	if r.Config != nil {
		cfg = r.Config.RouterLearning.Protection
	}
	run := tracing.Run{SessionID: harnessRunSession(ctx, cfg)}
	if agent := harnessRunAgent(ctx, cfg); agent != nil {
		run.Harness, run.AgentID, run.ParentAgentID, run.AgentKind = agent.Harness, agent.ID, agent.ParentID, agent.Kind
	}
	return run
}

// codexParentThread is the thread that spawned a Codex thread, empty when
// the root thread, whose id is the session's, spawned it.
func codexParentThread(ctx *RequestContext) string {
	parent := strings.TrimSpace(headerValueCI(ctx, headers.XCodexParentThreadID))
	session := strings.TrimSpace(headerValueCI(ctx, headers.CodexSessionID))
	if parent == "" || parent == session || parent == ctx.AgentTurn.SessionID {
		return ""
	}
	return parent
}

// requestHarness names the supported harness whose identity headers a
// request carries.
func requestHarness(ctx *RequestContext) string {
	switch {
	case headerValueCI(ctx, headers.XClaudeCodeSessionID) != "" || headerValueCI(ctx, headers.XClaudeCodeAgentID) != "":
		return store.HarnessClaudeCode
	case headerValueCI(ctx, headers.CodexSessionID) != "" || ctx.AgentTurn.SessionID != "" || ctx.AgentTurn.ThreadID != "":
		return store.HarnessCodex
	case isOpenCode(ctx):
		return store.HarnessOpenCode
	case isDeepSeekHarness(ctx):
		return store.HarnessDeepSeekHarness
	case isCursor(ctx):
		return store.HarnessCursor
	case isOpenClaw(ctx):
		return store.HarnessOpenClaw
	default:
		return ""
	}
}

func boundedIdentityValue(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) > limit {
		return ""
	}
	return value
}
