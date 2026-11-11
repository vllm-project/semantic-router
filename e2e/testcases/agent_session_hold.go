package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The agent-session recipe in the masking profile has two decisions that
// allow the same two models in opposite order, and session protection keyed
// on agentSessionHeader, scoped per agent by agentIDHeader, with a 3 second
// idle timeout.
const (
	agentSessionHeader     = "x-e2e-agent-session"
	agentIDHeader          = "x-e2e-agent-id"
	agentSessionEntrypoint = "vllm-sr/agent-session"
	agentSessionFirstModel = "openai/mock-model"
	agentSessionEditModel  = "openai/mock-model-b"
	agentSessionIdle       = 3 * time.Second
)

func init() {
	pkgtestcases.Register("agent-session-holds-model", pkgtestcases.TestCase{
		Description: "A session keeps the model its first turn got when a later turn selects a decision that allows it, a subagent in it is held on its own model without disturbing the parent, and a request without the session, a new session and a session past its idle timeout get the decision's own model",
		Tags:        []string{"session", "learning", "agent", "functional"},
		Fn:          testAgentSessionHoldsModel,
	})
}

type agentSessionTurn struct {
	name          string
	session       string
	agent         string
	keyword       string
	wantDecision  string
	wantModel     string
	waitIdleFirst bool
}

func testAgentSessionHoldsModel(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	held := fmt.Sprintf("agent-session-%d", time.Now().UnixNano())
	turns := []agentSessionTurn{
		// Without a session, the edit decision's own first model: the hold
		// below is only evidence if this differs from it.
		{name: "no session", keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionEditModel},
		{name: "first turn", session: held, keyword: "e2e-session-plan", wantDecision: "session_plan", wantModel: agentSessionFirstModel},
		{name: "held turn", session: held, keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionFirstModel},
		{name: "new session", session: held + "-other", keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionEditModel},
		// A subagent is held on its own key: its first turn does not inherit
		// the parent's model, its next turn keeps its own, and the parent's
		// hold is unchanged by it.
		{name: "subagent first turn", session: held, agent: "sub-1", keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionEditModel},
		{name: "subagent held turn", session: held, agent: "sub-1", keyword: "e2e-session-plan", wantDecision: "session_plan", wantModel: agentSessionEditModel},
		{name: "parent after subagent", session: held, keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionFirstModel},
		{name: "after idle timeout", session: held, keyword: "e2e-session-edit", wantDecision: "session_edit", wantModel: agentSessionEditModel, waitIdleFirst: true},
	}
	seen := make([]string, 0, len(turns))
	for _, turn := range turns {
		if turn.waitIdleFirst {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(agentSessionIdle + time.Second):
			}
		}
		var headers http.Header
		if turn.session != "" {
			headers = http.Header{agentSessionHeader: {turn.session}}
		}
		if turn.agent != "" {
			headers[agentIDHeader] = []string{turn.agent}
		}
		resp, err := runMaskingRawRequest(ctx, session, "/v1/chat/completions", map[string]interface{}{
			"model":    agentSessionEntrypoint,
			"messages": []map[string]interface{}{{"role": "user", "content": turn.keyword + " the parser change"}},
		}, headers)
		if err != nil {
			return fmt.Errorf("%s: %w", turn.name, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d: %s", turn.name, resp.StatusCode, truncateString(string(resp.Body), 300))
		}
		decision := resp.Headers.Get("x-vsr-selected-decision")
		model := resp.Headers.Get("x-vsr-selected-model")
		if decision != turn.wantDecision || model != turn.wantModel {
			return fmt.Errorf("%s: decision %q model %q, want %q %q (trace: %s)",
				turn.name, decision, model, turn.wantDecision, turn.wantModel,
				truncateString(resp.Headers.Get("x-vsr-trace"), 500))
		}
		seen = append(seen, turn.name+"="+model)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"turns": seen})
	}
	return nil
}
