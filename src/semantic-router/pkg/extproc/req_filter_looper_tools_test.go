package extproc

import (
	"encoding/json"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func looperChatTool(name string) string {
	return fmt.Sprintf(`{"type":"function","function":{"name":%q,"description":"Tool %s","parameters":{"type":"object","properties":{}}}}`, name, name)
}

// looperCallToolNames are the names of the tools one backend call carried.
func looperCallToolNames(t *testing.T, body json.RawMessage) []string {
	t.Helper()
	var wire struct {
		Tools []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("decode a backend call: %v", err)
	}
	names := []string{}
	for _, tool := range wire.Tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

// Every model call of a Looper decision carries only the tools its tools
// policy allows, as a single-model decision's call does.
func TestLooperCallsCarryOnlyTheToolsThePolicyAllows(t *testing.T) {
	request := `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"What is the capital of France?"}],"tools":[` +
		looperChatTool("admin_delete") + `,` + looperChatTool("lookup") + `]}`
	const plugin = `
      plugins:
        - type: tools
          configuration:
            enabled: true
            policy: {mode: filtered, block_tools: [admin_delete]}`
	for _, fixture := range looperFixtures {
		if !slices.Contains([]string{"escalate", "fusion-one-call", "workflows-static"}, fixture.name) {
			continue
		}
		t.Run(fixture.name, func(t *testing.T) {
			fixture.request, fixture.decision = request, fixture.decision+plugin
			harness := newLooperHarness(t, fixture)
			for _, adapter := range []string{"extproc", "sessions"} {
				record := harness.serve(t, looperAdapters[adapter](harness.router), fixture.request)
				carried := 0
				for _, call := range record.Calls {
					names := looperCallToolNames(t, call.Body)
					if len(names) > 0 && !slices.Equal(names, []string{"lookup"}) {
						t.Errorf("%s: the call to %s carried tools %v, want [lookup]", adapter, call.Model, names)
					}
					if len(names) > 0 {
						carried++
					}
				}
				if carried == 0 {
					t.Errorf("%s: none of %d calls carried the allowed tool", adapter, len(record.Calls))
				}
			}
		})
	}
}

// A workflow paused on a tool call, and resumed once the client ran it, sends
// every model only the tools the plugin allows: the policy removes
// admin_delete before the selection, which the judge would have kept, and
// the selection, once per request, drops notes.
func TestLooperWorkflowResumeCarriesOnlyTheToolsThePluginAllows(t *testing.T) {
	server, tracker := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"admin_delete": 0.9, "lookup": 0.9}}
	router.decisionDecider = judge
	router.decisionCards = map[string]modelservice.ModelCard{config.DefaultDecisionDeployment: {ID: "judge", Surfaces: []string{"decisions"}}}
	router.Config.ModelDeployments = map[string]config.ModelDeployment{
		config.DefaultDecisionDeployment: {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
	}
	plugin := config.ToolsPluginConfig{
		Enabled: true,
		Policy:  &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeFiltered, BlockTools: []string{"admin_delete"}},
		Select:  &config.ToolsSelectConfig{Mode: config.ToolsSelectModeFilter},
	}
	opts := workflowRouteOptions{plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginTools, Configuration: config.MustStructuredPayload(plugin)},
	}}
	withMoreTools := func(body []byte) []byte {
		var request map[string]any
		if err := json.Unmarshal(body, &request); err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"admin_delete", "notes"} {
			var tool any
			_ = json.Unmarshal([]byte(looperChatTool(name)), &tool)
			request["tools"] = append(request["tools"].([]any), tool)
		}
		out, _ := json.Marshal(request)
		return out
	}

	pause := routeWorkflowRequestWith(t, router, withMoreTools(workflowPauseChatBody(t)), opts)
	if got := immediateStatus(pause); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pause))
	}
	resume := routeWorkflowRequestWith(t, router, withMoreTools(workflowResumeChatBody(t, immediateBody(pause))), opts)
	if got := immediateStatus(resume); got != 200 {
		t.Fatalf("resume status = %d, body %s", got, immediateBody(resume))
	}
	if !tracker.sawToolResult() || !tracker.sawFinal() {
		t.Fatal("the resume did not finish the workflow")
	}
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	// The paused worker call, the resumed one and the final call.
	if len(tracker.tools) != 3 {
		t.Fatalf("backend calls carried tools %v, want 3 calls", tracker.tools)
	}
	for index, names := range tracker.tools {
		if !slices.Equal(names, []string{"lookup"}) {
			t.Errorf("backend call %d carried tools %v, want [lookup]", index, names)
		}
	}
	selections := 0
	for _, asked := range judge.asked() {
		if slices.ContainsFunc(asked.Questions, func(q modelservice.Question) bool { return q.ID == toolNeedQuestionID }) {
			selections++
		}
	}
	if selections != 2 {
		t.Errorf("the selection ran %d times for 2 requests, want 2", selections)
	}
}

// A workflow paused before its decision's policy blocked a tool resumes
// under the policy in force: the resumed call and the final call, which the
// workflow builds from the requests it stored at the pause, carry only the
// tools the policy allows now.
func TestLooperWorkflowResumeAppliesTheCurrentPolicy(t *testing.T) {
	server, tracker := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
	withBlockedTool := func(body []byte) []byte {
		var request map[string]any
		if err := json.Unmarshal(body, &request); err != nil {
			t.Fatal(err)
		}
		var tool any
		_ = json.Unmarshal([]byte(looperChatTool("admin_delete")), &tool)
		request["tools"] = append(request["tools"].([]any), tool)
		out, _ := json.Marshal(request)
		return out
	}
	pause := routeWorkflowRequestWith(t, router, withBlockedTool(workflowPauseChatBody(t)), workflowRouteOptions{})
	if got := immediateStatus(pause); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pause))
	}
	tightened := config.ToolsPluginConfig{Enabled: true, Policy: &config.ToolsPolicyConfig{
		Mode: config.ToolsPolicyModeFiltered, BlockTools: []string{"admin_delete"},
	}}
	opts := workflowRouteOptions{plugins: []config.DecisionPlugin{
		{Type: config.DecisionPluginTools, Configuration: config.MustStructuredPayload(tightened)},
	}}
	resume := routeWorkflowRequestWith(t, router, withBlockedTool(workflowResumeChatBody(t, immediateBody(pause))), opts)
	if got := immediateStatus(resume); got != 200 {
		t.Fatalf("resume status = %d, body %s", got, immediateBody(resume))
	}
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	if len(tracker.tools) != 3 || !tracker.workerSawToolResult || !tracker.finalCalled {
		t.Fatalf("backend calls carried tools %v, want the paused, the resumed and the final call", tracker.tools)
	}
	for index, names := range tracker.tools[1:] {
		if !slices.Equal(names, []string{"lookup"}) {
			t.Errorf("call %d after the resume carried tools %v, want [lookup]", index+1, names)
		}
	}
}

// Under trusted facts a Looper decision's selection adds no catalog tool: a
// call the gate narrows keeps only the request's own allowed tools.
func TestLooperTrustedFactsSelectionAddsNoCatalogTool(t *testing.T) {
	server, tracker := newWorkflowPauseResumeServer(t)
	router := newWorkflowRouter(t, newWorkflowLooperConfig(t, server.URL, config.WorkflowStateBackendMemory))
	judge := &toolsJudge{need: 0.9, choose: map[string]float64{"catalog_tool": 0.9}}
	router.decisionDecider = judge
	router.decisionCards = map[string]modelservice.ModelCard{config.DefaultDecisionDeployment: {ID: "judge", Surfaces: []string{"decisions"}}}
	router.Config.ModelDeployments = map[string]config.ModelDeployment{
		config.DefaultDecisionDeployment: {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
	}
	// A catalog read longer ago than freshness_seconds is unavailable, which
	// narrows each authorized call.
	router.toolCatalogs = map[string]*toolCatalog{trustedFactsCatalogPath: {
		tools:    []config.CatalogTool{{Tool: llmprotocol.Tool{Name: "catalog_tool", InputSchema: json.RawMessage(`{"type":"object"}`)}, Description: "Catalog tool"}},
		loadedAt: time.Now().Add(-time.Hour),
	}}
	plugin := trustedFactsTestConfig(t, &config.TrustedFactsConfig{
		Enabled:          true,
		Enforcement:      config.TrustedEnforcementAuthoritative,
		TrustSources:     []string{config.TrustedSourceOperatorPolicy},
		StageRoles:       []string{config.TrustedStageCandidate, config.TrustedStageFinal},
		FreshnessSeconds: 1,
	})
	opts := workflowRouteOptions{plugins: []config.DecisionPlugin{mustToolsDecisionPlugin(t, plugin)}}

	pause := routeWorkflowRequestWith(t, router, workflowPauseChatBody(t), opts)
	if got := immediateStatus(pause); got != 200 {
		t.Fatalf("pause status = %d, body %s", got, immediateBody(pause))
	}
	tracker.mu.Lock()
	defer tracker.mu.Unlock()
	for index, names := range tracker.tools {
		if !slices.Equal(names, []string{"lookup"}) {
			t.Errorf("backend call %d carried tools %v, want [lookup]", index, names)
		}
	}
	if asked := len(judge.asked()); asked != 0 {
		t.Errorf("the selection asked the judge %d times", asked)
	}
}
