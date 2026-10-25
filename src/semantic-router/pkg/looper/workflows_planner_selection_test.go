package looper

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func dynamicPlannerCandidateRequest() *Request {
	req := candidateStageRequest()
	req.ModelRefs = []config.ModelRef{{Model: "worker"}, {Model: "second"}}
	req.PermittedModels = append(req.PermittedModels, "second")
	req.ModelParams["second"] = config.ModelParams{Capabilities: []string{"text", "structured_json"}, ContextWindowSize: 4096, MaxOutputTokens: 512}
	req.Algorithm = &config.AlgorithmConfig{Type: config.DecisionAlgorithmWorkflows, Workflows: &config.WorkflowsAlgorithmConfig{
		Mode: config.WorkflowModeDynamic, MaxSteps: 1, MaxParallel: 2, MinSuccessfulResponses: 2,
		MaxCompletionTokens: 16, Planner: config.WorkflowPlannerConfig{MaxCompletionTokens: 32},
	}}
	return req
}

func TestDynamicWorkflowPlannerScansActualStageWithoutCalls(t *testing.T) {
	for _, reason := range []string{"json capability", "planner context", "planner output"} {
		t.Run(reason, func(t *testing.T) {
			client, calls, _ := candidateStageClient(t)
			l := newWorkflowsLooper(&config.LooperConfig{}, borrowClient(client))
			req := dynamicPlannerCandidateRequest()
			p := req.ModelParams["worker"]
			if reason != "json capability" {
				p.Capabilities = []string{"text", "structured_json"}
			}
			if reason == "planner context" {
				p.ContextWindowSize = 64
			}
			if reason == "planner output" {
				p.MaxOutputTokens = 24
			}
			req.ModelParams["worker"] = p
			cfg := resolveWorkflowsExecutionConfig(req)
			resolved, err := l.resolveDynamicWorkflowPlanner(req, cfg, extractOriginalContent(req.OriginalRequest), modelRefsToNames(req.ModelRefs))
			if err != nil || resolved.PlannerModel != "second" || calls.Load() != 0 {
				t.Fatalf("resolved=%q err=%v calls=%d", resolved.PlannerModel, err, calls.Load())
			}
			if req.Algorithm.Workflows.Planner.Model != "" || cfg.PlannerModel != "" {
				t.Fatal("resolver mutated recipe configuration")
			}
		})
	}
}

func TestDynamicWorkflowPlannerPreservesOrderExplicitTargetAndMinimum(t *testing.T) {
	client, calls, _ := candidateStageClient(t)
	l := newWorkflowsLooper(&config.LooperConfig{}, borrowClient(client))
	req := dynamicPlannerCandidateRequest()
	p := req.ModelParams["worker"]
	p.Capabilities = []string{"text", "structured_json"}
	req.ModelParams["worker"] = p
	cfg := resolveWorkflowsExecutionConfig(req)
	resolved, err := l.resolveDynamicWorkflowPlanner(req, cfg, "Agenda", modelRefsToNames(req.ModelRefs))
	if err != nil || resolved.PlannerModel != "worker" {
		t.Fatalf("assignment order lost: %q %v", resolved.PlannerModel, err)
	}
	cfg.PlannerModel = "second"
	resolved, err = l.resolveDynamicWorkflowPlanner(req, cfg, "Agenda", modelRefsToNames(req.ModelRefs))
	if err != nil || resolved.PlannerModel != "second" {
		t.Fatalf("explicit planner changed: %q %v", resolved.PlannerModel, err)
	}
	p = req.ModelParams["second"]
	p.Capabilities = []string{"text"}
	req.ModelParams["second"] = p
	if _, err := l.resolveDynamicWorkflowPlanner(req, cfg, "Agenda", modelRefsToNames(req.ModelRefs)); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("explicit ineligible planner was rerouted: %v", err)
	}
	cfg.PlannerModel = ""
	if _, err := l.resolveDynamicWorkflowPlanner(req, cfg, "Agenda", []string{"worker", "worker"}); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("distinct worker minimum weakened: %v", err)
	}
	p = req.ModelParams["worker"]
	p.Capabilities = []string{"text"}
	req.ModelParams["worker"] = p
	if _, err := l.resolveDynamicWorkflowPlanner(req, cfg, "Agenda", modelRefsToNames(req.ModelRefs)); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("no eligible planner accepted: %v", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("admission scan made %d model calls", calls.Load())
	}
}

func TestDynamicWorkflowDefaultPlannerStillCallsPlannerTwoWorkersAndFinal(t *testing.T) {
	type call struct {
		Model   string
		Max     int
		Planner bool
	}
	var mu sync.Mutex
	var observed []call
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Model  string `json:"model"`
			Max    int    `json:"max_completion_tokens"`
			Format *struct {
				Type string `json:"type"`
			} `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		planner := body.Format != nil && body.Format.Type == "json_object"
		mu.Lock()
		observed = append(observed, call{body.Model, body.Max, planner})
		mu.Unlock()
		content := "Agenda summary."
		if planner {
			content = `{"steps":[{"id":"draft","role":"worker","models":["worker","second"],"prompt":"Summarize the agenda."}],"final":{"prompt":"Combine the summaries."}}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id": "workflow", "object": "chat.completion", "created": 1, "model": body.Model,
			"choices": []interface{}{map[string]interface{}{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		})
	}))
	t.Cleanup(server.Close)
	l := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL})
	t.Cleanup(func() { _ = l.Close() })
	resp, err := l.Execute(context.Background(), dynamicPlannerCandidateRequest())
	if err != nil {
		t.Fatalf("dynamic execution failed: %v", err)
	}
	if resp.Model != "second" {
		t.Fatalf("final coordinator=%q", resp.Model)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(observed) != 4 || observed[0] != (call{"second", 32, true}) || observed[3] != (call{"second", 16, false}) {
		t.Fatalf("planner/final stages=%v", observed)
	}
	workers := map[string]bool{}
	for _, c := range observed[1:3] {
		if c.Planner || c.Max != 16 {
			t.Fatalf("planner budget leaked into worker stage: %+v", c)
		}
		workers[c.Model] = true
	}
	if !workers["worker"] || !workers["second"] {
		t.Fatalf("two distinct workers missing: %v", observed)
	}
}

func TestDynamicWorkflowResumeKeepsAdmittedPlanner(t *testing.T) {
	cfg := workflowsExecutionConfig{Mode: config.WorkflowModeDynamic}
	state := &workflowPendingToolState{PlannerResp: &ModelResponse{Model: "second"}}
	resolved, err := resolvedWorkflowPlannerForResume(cfg, state, []string{"worker", "second"})
	if err != nil || resolved.PlannerModel != "second" {
		t.Fatalf("resumed coordinator changed: %q %v", resolved.PlannerModel, err)
	}
	if _, err := resolvedWorkflowPlannerForResume(cfg, state, []string{"worker"}); !errors.Is(err, selection.ErrNoEligibleCandidates) {
		t.Fatalf("removed coordinator accepted: %v", err)
	}
}
