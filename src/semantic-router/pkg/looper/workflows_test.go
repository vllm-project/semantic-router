package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowsDynamicExecutesPlannerBoundedWorkersAndSynthesis(t *testing.T) {
	server := newWorkflowTestServer(t, map[string]string{
		"qwen-coordinator": `{"steps":[{"id":"solve","role":"worker","models":["worker-a","worker-b"],"prompt":"solve independently"}],"final":{"prompt":"merge the best answer"}}`,
		"worker-a":         "worker-a answer",
		"worker-b":         "worker-b answer",
	})
	defer server.Close()

	req := workflowTestRequest()
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: req,
		ModelRefs: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				MaxSteps:    3,
				MaxParallel: 2,
			},
		},
		DecisionName: "flow-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if resp.AlgorithmType != "workflows" {
		t.Fatalf("algorithm = %q, want workflows", resp.AlgorithmType)
	}
	if resp.Model != "qwen-coordinator" {
		t.Fatalf("final model = %q, want qwen-coordinator", resp.Model)
	}
	for _, want := range []string{"qwen-coordinator", "worker-a", "worker-b"} {
		if !workflowContainsString(resp.ModelsUsed, want) {
			t.Fatalf("models used %v missing %q", resp.ModelsUsed, want)
		}
	}

	var body map[string]interface{}
	if err := json.Unmarshal(resp.Body, &body); err != nil {
		t.Fatalf("response body is not JSON: %v", err)
	}
	if _, ok := body["flow"]; !ok {
		t.Fatalf("response body missing flow trace: %s", string(resp.Body))
	}
}

func TestWorkflowsDynamicPlannerStripsToolAndFunctionSchemas(t *testing.T) {
	server := newWorkflowToolSchemaServer(t)
	defer server.Close()

	_, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowMixedToolSchemasTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker-a"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:        config.WorkflowModeDynamic,
				Planner:     config.WorkflowPlannerConfig{Model: "qwen-coordinator"},
				MaxSteps:    2,
				MaxParallel: 1,
			},
		},
		DecisionName: "flow-tool-schema-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
}

func TestWorkflowsDynamicUsesPlannerTokenBudgetSeparately(t *testing.T) {
	plannerMaxTokens := []int64{}
	workerMaxTokens := []int64{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		if payload.MaxCompletionTokens == nil {
			t.Fatalf("request for model %q missing max_completion_tokens", payload.Model)
		}
		switch payload.Model {
		case "qwen-coordinator":
			plannerMaxTokens = append(plannerMaxTokens, *payload.MaxCompletionTokens)
			_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"solve","role":"worker","models":["worker-a"],"prompt":"solve"}],"final":{"model":"worker-a","prompt":"merge"}}`))
		case "worker-a":
			workerMaxTokens = append(workerMaxTokens, *payload.MaxCompletionTokens)
			_, _ = w.Write(workflowChatCompletion("worker-a", "worker or final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	_, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker-a"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model:               "qwen-coordinator",
					MaxCompletionTokens: 1024,
				},
				MaxSteps:            2,
				MaxParallel:         1,
				MaxCompletionTokens: 32768,
			},
		},
		DecisionName: "flow-planner-budget-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if len(plannerMaxTokens) != 1 || plannerMaxTokens[0] != 1024 {
		t.Fatalf("planner max tokens = %v, want [1024]", plannerMaxTokens)
	}
	if len(workerMaxTokens) != 2 {
		t.Fatalf("worker/final calls = %v, want two calls", workerMaxTokens)
	}
	for _, got := range workerMaxTokens {
		if got != 32768 {
			t.Fatalf("worker/final max tokens = %v, want all 32768", workerMaxTokens)
		}
	}
}

func TestWorkflowsDynamicUsesConfiguredFinalModel(t *testing.T) {
	server := newWorkflowTestServer(t, map[string]string{
		"qwen-coordinator": `{"steps":[{"id":"solve","role":"worker","models":["worker-a"],"prompt":"solve independently"}],"final":{"prompt":"merge the worker answer"}}`,
		"worker-a":         "worker answer",
		"gpt-final":        "final answer from configured model",
	})
	defer server.Close()

	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "gpt-final"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode:    config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{Model: "qwen-coordinator"},
				Final: config.WorkflowFinalConfig{
					Model:  "gpt-final",
					Prompt: "Use the best worker evidence and preserve strict output formats.",
				},
				MaxSteps:    3,
				MaxParallel: 1,
			},
		},
		DecisionName: "flow-final-model-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if resp.Model != "gpt-final" {
		t.Fatalf("final model = %q, want gpt-final", resp.Model)
	}
	if !strings.Contains(string(resp.Body), "final answer from configured model") {
		t.Fatalf("response body missing configured final model output: %s", string(resp.Body))
	}
	trace, ok := resp.IntermediateResponses.(*workflowTrace)
	if !ok || trace.Plan == nil || trace.Plan.Final == nil {
		t.Fatalf("missing workflow trace final: %#v", resp.IntermediateResponses)
	}
	if trace.Plan.Final.Model != "gpt-final" {
		t.Fatalf("trace final model = %q, want gpt-final", trace.Plan.Final.Model)
	}
}

func TestWorkflowsParallelStepReturnsAfterQuorum(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "qwen-coordinator":
			_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"solve","role":"worker","models":["worker-a","worker-b"],"prompt":"solve"}],"final":{"model":"worker-a","prompt":"merge"}}`))
		case "worker-a":
			_, _ = w.Write(workflowChatCompletion("worker-a", "fast answer"))
		case "worker-b":
			time.Sleep(2 * time.Second)
			_, _ = w.Write(workflowChatCompletion("worker-b", "slow answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	start := time.Now()
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker-a"}, {Model: "worker-b"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				Final:                  config.WorkflowFinalConfig{Model: "worker-a"},
				MaxSteps:               2,
				MaxParallel:            2,
				RoundTimeoutSeconds:    1,
				MinSuccessfulResponses: 1,
				OnError:                config.WorkflowOnErrorSkip,
			},
		},
		DecisionName: "flow-quorum-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 1800*time.Millisecond {
		t.Fatalf("workflow waited for slow worker; elapsed=%v", elapsed)
	}
	if !strings.Contains(string(resp.Body), "fast answer") {
		t.Fatalf("response missing fast worker answer: %s", string(resp.Body))
	}
}

func TestWorkflowsFinalTimeoutFallsBackToWorkerResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "qwen-coordinator":
			_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"solve","role":"worker","models":["worker-a"],"prompt":"solve"}],"final":{"model":"gpt-final","prompt":"merge"}}`))
		case "worker-a":
			_, _ = w.Write(workflowChatCompletion("worker-a", "B"))
		case "gpt-final":
			time.Sleep(2 * time.Second)
			_, _ = w.Write(workflowChatCompletion("gpt-final", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := workflowTestRequest()
	req.Messages = []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Answer the multiple choice question with exactly one letter: A, B, C, or D."),
	}
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest:    req,
		OutputContract:     singleChoiceOutputContractForTest(),
		OutputContractSpec: singleChoiceOutputContractSpecForTest(),
		ModelRefs:          []config.ModelRef{{Model: "worker-a"}, {Model: "gpt-final"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				MaxSteps:            2,
				MaxParallel:         1,
				RoundTimeoutSeconds: 1,
				OnError:             config.WorkflowOnErrorSkip,
			},
		},
		DecisionName: "flow-final-timeout-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if got := workflowResponseContent(t, resp.Body); got != "B" {
		t.Fatalf("fallback final content = %q, want B", got)
	}
}

func TestWorkflowsFinalTimeoutUsesSingleChoiceMajorityFallback(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := decodeWorkflowRequestPayload(t, r)
		w.Header().Set("Content-Type", "application/json")
		switch payload.Model {
		case "qwen-coordinator":
			_, _ = w.Write(workflowChatCompletion("qwen-coordinator", `{"steps":[{"id":"solve","role":"worker","models":["worker-a","worker-b","worker-c"],"prompt":"solve independently"}],"final":{"model":"gpt-final","prompt":"merge"}}`))
		case "worker-a":
			_, _ = w.Write(workflowChatCompletion("worker-a", "B"))
		case "worker-b":
			_, _ = w.Write(workflowChatCompletion("worker-b", "B"))
		case "worker-c":
			_, _ = w.Write(workflowChatCompletion("worker-c", "C"))
		case "gpt-final":
			time.Sleep(2 * time.Second)
			_, _ = w.Write(workflowChatCompletion("gpt-final", "final answer"))
		default:
			t.Fatalf("unexpected model call: %s", payload.Model)
		}
	}))
	defer server.Close()

	req := workflowTestRequest()
	req.Messages = []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage("Answer the multiple choice question with exactly one letter: A, B, C, or D."),
	}
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest:    req,
		OutputContract:     singleChoiceOutputContractForTest(),
		OutputContractSpec: singleChoiceOutputContractSpecForTest(),
		ModelRefs: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
			{Model: "worker-c"},
			{Model: "gpt-final"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				MaxSteps:               2,
				MaxParallel:            3,
				RoundTimeoutSeconds:    1,
				MinSuccessfulResponses: 3,
				OnError:                config.WorkflowOnErrorSkip,
			},
		},
		DecisionName: "flow-final-timeout-majority-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if got := workflowResponseContent(t, resp.Body); got != "B" {
		t.Fatalf("fallback final content = %q, want B", got)
	}
	if resp.Model != "worker-a" {
		t.Fatalf("fallback model = %q, want first majority response model worker-a", resp.Model)
	}
}

func TestWorkflowsStaticExecutesConfiguredRoles(t *testing.T) {
	server := newWorkflowTestServer(t, map[string]string{
		"thinker-model":  "thinker output",
		"worker-model":   "worker output",
		"verifier-model": "verifier output",
	})
	defer server.Close()

	includeTrace := true
	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs: []config.ModelRef{
			{Model: "thinker-model"},
			{Model: "worker-model"},
			{Model: "verifier-model"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeStatic,
				Roles: []config.WorkflowRoleConfig{
					{Name: "thinker", Models: []string{"thinker-model"}, Prompt: "plan"},
					{Name: "worker", Models: []string{"worker-model"}, Prompt: "solve"},
					{Name: "verifier", Models: []string{"verifier-model"}, Prompt: "check"},
				},
				Final:                        config.WorkflowFinalConfig{Model: "verifier-model"},
				MaxSteps:                     3,
				MaxParallel:                  1,
				IncludeIntermediateResponses: &includeTrace,
			},
		},
		DecisionName: "static-flow-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if resp.Model != "verifier-model" {
		t.Fatalf("final model = %q, want verifier-model", resp.Model)
	}
	trace, ok := resp.IntermediateResponses.(*workflowTrace)
	if !ok {
		t.Fatalf("trace type = %T, want *workflowTrace", resp.IntermediateResponses)
	}
	if len(trace.Steps) != 3 {
		t.Fatalf("trace steps = %d, want 3", len(trace.Steps))
	}
	for idx, wantRole := range []string{"thinker", "worker", "verifier"} {
		if trace.Steps[idx].Role != wantRole {
			t.Fatalf("step %d role = %q, want %q", idx, trace.Steps[idx].Role, wantRole)
		}
	}
	if len(trace.Steps[1].Responses) == 0 || trace.Steps[1].Responses[0].AgentID != "worker:0:worker-model" {
		t.Fatalf("worker response trace missing agent_id: %#v", trace.Steps[1].Responses)
	}
}

func TestWorkflowsDynamicRejectsPlannerModelOutsideModelRefs(t *testing.T) {
	server := newWorkflowTestServer(t, map[string]string{
		"qwen-coordinator": `{"steps":[{"id":"escape","role":"worker","models":["not-allowed"],"prompt":"use forbidden model"}]}`,
	})
	defer server.Close()

	_, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs:       []config.ModelRef{{Model: "worker-a"}},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				MaxSteps:    3,
				MaxParallel: 2,
			},
		},
		DecisionName: "flow-test",
	})
	if err == nil || !strings.Contains(err.Error(), "outside decision modelRefs") {
		t.Fatalf("expected modelRefs boundary error, got %v", err)
	}
}

func TestWorkflowsDynamicFallsBackWhenPlannerJSONInvalidAndOnErrorSkip(t *testing.T) {
	server := newWorkflowTestServer(t, map[string]string{
		"qwen-coordinator": "not json",
		"worker-a":         "A",
		"worker-b":         "B",
	})
	defer server.Close()

	resp, err := NewWorkflowsLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), &Request{
		OriginalRequest: workflowTestRequest(),
		ModelRefs: []config.ModelRef{
			{Model: "worker-a"},
			{Model: "worker-b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "workflows",
			Workflows: &config.WorkflowsAlgorithmConfig{
				Mode: config.WorkflowModeDynamic,
				Planner: config.WorkflowPlannerConfig{
					Model: "qwen-coordinator",
				},
				MaxSteps:    3,
				MaxParallel: 2,
				OnError:     config.WorkflowOnErrorSkip,
			},
		},
		DecisionName: "flow-test",
	})
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	var body map[string]interface{}
	if unmarshalErr := json.Unmarshal(resp.Body, &body); unmarshalErr != nil {
		t.Fatalf("response body is not JSON: %v", unmarshalErr)
	}
	flow, ok := body["flow"].(map[string]interface{})
	if !ok {
		t.Fatalf("response body missing flow trace: %s", string(resp.Body))
	}
	plan := flow["plan"].(map[string]interface{})
	steps := plan["steps"].([]interface{})
	firstStep := steps[0].(map[string]interface{})
	if firstStep["id"] != "fallback_solve" {
		t.Fatalf("fallback step id = %v", firstStep["id"])
	}
}

func TestParseWorkflowPlanFromReasoningContent(t *testing.T) {
	raw := []byte(`{
		"choices": [{
			"message": {
				"role": "assistant",
				"content": "",
				"reasoning_content": "Plan internally.\n{\"steps\":[{\"id\":\"solve\",\"role\":\"worker\",\"models\":[\"worker-a\"],\"prompt\":\"solve\"}]}"
			}
		}]
	}`)
	plan, err := parseWorkflowPlanFromResponse(&ModelResponse{Raw: raw})
	if err != nil {
		t.Fatalf("parseWorkflowPlanFromResponse failed: %v", err)
	}
	if len(plan.Steps) != 1 || plan.Steps[0].Models[0] != "worker-a" {
		t.Fatalf("unexpected plan: %#v", plan)
	}
}

func TestParseWorkflowPlanRepairsPlannerJSONDrift(t *testing.T) {
	content := "```json\n{`steps`:[{`id`:\"solve\",`role`:\"worker\",`models`:[\"worker-a\"],`prompt`:\"solve\",}],`final`:{`prompt`:\"merge\",},}\n```"
	raw, err := json.Marshal(map[string]interface{}{
		"choices": []map[string]interface{}{{
			"message": map[string]interface{}{
				"role":    "assistant",
				"content": content,
			},
		}},
	})
	if err != nil {
		t.Fatalf("marshal raw response: %v", err)
	}
	plan, err := parseWorkflowPlanFromResponse(&ModelResponse{Raw: raw})
	if err != nil {
		t.Fatalf("parseWorkflowPlanFromResponse failed: %v", err)
	}
	if len(plan.Steps) != 1 || plan.Steps[0].ID != "solve" || plan.Final.Prompt != "merge" {
		t.Fatalf("unexpected plan: %#v", plan)
	}
}

func TestParseWorkflowPlanRepairsInvalidStringEscapes(t *testing.T) {
	plan, err := parseWorkflowPlan(`{"steps":[{"id":"solve","role":"worker","models":["worker-a"],"prompt":"handle \(escaped\) text"}]}`)
	if err != nil {
		t.Fatalf("parseWorkflowPlan failed: %v", err)
	}
	if len(plan.Steps) != 1 || plan.Steps[0].Prompt != `handle \(escaped\) text` {
		t.Fatalf("unexpected plan: %#v", plan)
	}
}

func TestConfigureWorkflowPlannerRequestDisablesQwenThinking(t *testing.T) {
	req := workflowTestRequest()
	configureWorkflowPlannerRequest(req, "qwen/qwen3.6-rocm")
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal planner request: %v", err)
	}
	if !strings.Contains(string(body), `"response_format":{"type":"json_object"}`) {
		t.Fatalf("planner request missing JSON response format: %s", string(body))
	}
	if !strings.Contains(string(body), `"chat_template_kwargs":{"enable_thinking":false}`) {
		t.Fatalf("planner request missing Qwen thinking control: %s", string(body))
	}
}

func TestStripFusionToolUsePreservesPlannerExtraFields(t *testing.T) {
	req := workflowTestRequest()
	configureWorkflowPlannerRequest(req, "qwen/qwen3.6-rocm")
	stripped := stripFusionToolUse(req)
	body, err := json.Marshal(stripped)
	if err != nil {
		t.Fatalf("marshal stripped request: %v", err)
	}
	if !strings.Contains(string(body), `"chat_template_kwargs":{"enable_thinking":false}`) {
		t.Fatalf("planner extra fields were not preserved: %s", string(body))
	}
}

func TestApplyWorkflowModelReasoningControlUsesModelParamsFamily(t *testing.T) {
	useReasoning := false
	req := workflowTestRequest()
	applyWorkflowModelReasoningControl(req, "worker-alias", &Request{
		ModelRefs: []config.ModelRef{{
			Model: "worker-alias",
			ModelReasoningControl: config.ModelReasoningControl{
				UseReasoning: &useReasoning,
			},
		}},
		ModelParams: map[string]config.ModelParams{
			"worker-alias": {ReasoningFamily: "qwen3"},
		},
	})
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("marshal request: %v", err)
	}
	if !strings.Contains(string(body), `"chat_template_kwargs":{"enable_thinking":false}`) {
		t.Fatalf("model reasoning control was not applied: %s", string(body))
	}
}

func TestBuildWorkflowFinalPromptPreservesConstrainedOutputFormat(t *testing.T) {
	prompt := buildWorkflowFinalPrompt(&workflowPlan{}, "Answer with only A, B, C, or D.", "", []workflowStepResult{
		{
			step: workflowPlanStep{ID: "solve", Role: "worker"},
			responses: []*ModelResponse{{
				Model:   "worker-a",
				Content: "The answer is C.",
			}},
		},
	})
	for _, want := range []string{
		"Preserve any constrained output format exactly",
		"only that format",
		"Answer with only A, B, C, or D.",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("final prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestExtractRequestOutputContractFromExplicitMarker(t *testing.T) {
	var req openai.ChatCompletionNewParams
	raw := []byte(`{
		"model":"vllm-sr/flow",
		"messages":[
			{"role":"system","content":"You are answering an exam.\n\nRequired output contract:\nExplanation: <brief explanation>\nExact Answer: <answer>\nConfidence: <0-100>"},
			{"role":"user","content":"Compute the invariant."}
		]
	}`)
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("parse request: %v", err)
	}

	contract := extractRequestOutputContract(&req)
	want := "Explanation: <brief explanation>\nExact Answer: <answer>\nConfidence: <0-100>"
	if contract != want {
		t.Fatalf("output contract = %q, want %q", contract, want)
	}
}

func TestExtractRequestOutputContractIgnoresUnmarkedFormatText(t *testing.T) {
	var req openai.ChatCompletionNewParams
	raw := []byte(`{
		"model":"vllm-sr/flow",
		"messages":[
			{"role":"system","content":"Your response should be in the following format:\nExplanation: <brief explanation>\nExact Answer: <answer>\nConfidence: <0-100>"},
			{"role":"user","content":"Compute the invariant."}
		]
	}`)
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("parse request: %v", err)
	}

	if contract := extractRequestOutputContract(&req); contract != "" {
		t.Fatalf("unmarked text must not be inferred as output contract:\n%s", contract)
	}
}

func TestRequestOutputContractMergesDecisionContract(t *testing.T) {
	var req openai.ChatCompletionNewParams
	raw := []byte(`{
		"model":"vllm-sr/flow",
		"messages":[
			{"role":"user","content":"Solve the task.\n\nRequired output contract:\nReturn only a JSON object with answer and confidence."}
		]
	}`)
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatalf("parse request: %v", err)
	}

	contract := requestOutputContract(&req, "Preserve benchmark scoring format exactly.")
	for _, want := range []string{
		"Return only a JSON object",
		"Preserve benchmark scoring format exactly.",
	} {
		if !strings.Contains(contract, want) {
			t.Fatalf("merged output contract missing %q:\n%s", want, contract)
		}
	}
}

func TestBuildWorkflowFinalPromptIncludesSystemOutputContract(t *testing.T) {
	outputContract := "Your response should be in the following format:\nExplanation: <brief explanation>\nExact Answer: <answer>\nConfidence: <0-100>"
	prompt := buildWorkflowFinalPrompt(&workflowPlan{}, "Compute the invariant.", outputContract, []workflowStepResult{
		{
			step: workflowPlanStep{ID: "solve", Role: "worker"},
			responses: []*ModelResponse{{
				Model:   "worker-a",
				Content: "Exact Answer: Z+Z",
			}},
		},
	})
	for _, want := range []string{
		"Required output contract",
		"Exact Answer:",
		"Confidence:",
		"Do not reveal hidden reasoning",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("final prompt missing %q:\n%s", want, prompt)
		}
	}
}

func TestWorkflowStepPromptRespectsAccessList(t *testing.T) {
	prompt := buildWorkflowStepPrompt(workflowTestRequest(), workflowPlanStep{
		ID:         "verify",
		Role:       "verifier",
		Prompt:     "Check only visible outputs.",
		AccessList: []string{"public"},
	}, []workflowStepResult{
		{
			step: workflowPlanStep{ID: "private", Role: "worker"},
			responses: []*ModelResponse{{
				Model:   "worker-private",
				Content: "secret private output",
			}},
		},
		{
			step: workflowPlanStep{ID: "public", Role: "worker"},
			responses: []*ModelResponse{{
				Model:   "worker-public",
				Content: "allowed public output",
			}},
		},
	})
	if strings.Contains(prompt, "secret private output") {
		t.Fatalf("prompt leaked private prior output:\n%s", prompt)
	}
	if !strings.Contains(prompt, "allowed public output") {
		t.Fatalf("prompt missing allowed prior output:\n%s", prompt)
	}
}

func TestWorkflowStepPromptRespectsAgentAccessList(t *testing.T) {
	prompt := buildWorkflowStepPrompt(workflowTestRequest(), workflowPlanStep{
		ID:         "verify",
		Role:       "verifier",
		Prompt:     "Check only one agent output.",
		AccessList: []string{"solve:1:worker-b"},
	}, []workflowStepResult{
		{
			step: workflowPlanStep{
				ID:     "solve",
				Role:   "worker",
				Models: []string{"worker-a", "worker-b"},
			},
			responses: []*ModelResponse{
				{
					Model:   "worker-a",
					Content: "private worker-a output",
				},
				{
					Model:   "worker-b",
					Content: "allowed worker-b output",
				},
			},
		},
	})
	if strings.Contains(prompt, "private worker-a output") {
		t.Fatalf("prompt leaked non-allowed agent output:\n%s", prompt)
	}
	if !strings.Contains(prompt, "solve:1:worker-b") || !strings.Contains(prompt, "allowed worker-b output") {
		t.Fatalf("prompt missing allowed agent output:\n%s", prompt)
	}
}

func TestWorkflowFinalOutputContractExtractsSingleChoiceAnswer(t *testing.T) {
	resp := &ModelResponse{
		Model:   "qwen-coordinator",
		Content: "B",
	}
	applyFinalOutputContract(singleChoiceOutputContractSpecForTest(), resp)
	if resp.Content != "B" {
		t.Fatalf("final content = %q, want B", resp.Content)
	}
}

func TestWorkflowFinalOutputContractDoesNotInferLabeledChoice(t *testing.T) {
	resp := &ModelResponse{
		Model:   "gpt-final",
		Content: "The final answer follows from comparing the choices.\n\nANSWER: C",
	}
	applyFinalOutputContract(singleChoiceOutputContractSpecForTest(), resp)
	if resp.Content != "The final answer follows from comparing the choices.\n\nANSWER: C" {
		t.Fatalf("final content changed by heuristic extraction: %q", resp.Content)
	}
}

func TestWorkflowFinalOutputContractUsesReasoningOnlyWhenContentEmpty(t *testing.T) {
	resp := &ModelResponse{
		Model:            "gpt-final",
		Content:          "",
		ReasoningContent: "A",
	}
	spec := singleChoiceOutputContractSpecForTest()
	spec.Extract = &config.OutputContractExtractSpec{
		Mode:    config.OutputContractExtractModeExact,
		Sources: []string{config.OutputContractExtractSourceReasoningContent},
	}
	applyFinalOutputContract(spec, resp)
	if resp.Content != "A" {
		t.Fatalf("final content = %q, want A", resp.Content)
	}
}

func TestJSONActionOutputContractKeepsFinalValidAction(t *testing.T) {
	resp := &ModelResponse{
		Model:   "gpt-final",
		Content: "Here is the next action:\n```json\n{\"analysis\":\"inspect\",\"plan\":\"list files\",\"commands\":[{\"keystrokes\":\"ls -la /app\\n\",\"duration\":0.2}],\"task_complete\":false}\n```",
	}

	applyJSONActionOutputContract(terminalActionOutputContractSpecForTest(), resp, nil)

	var action map[string]interface{}
	if err := json.Unmarshal([]byte(resp.Content), &action); err != nil {
		t.Fatalf("final content is not JSON: %v\n%s", err, resp.Content)
	}
	if action["analysis"] != "inspect" || action["task_complete"] != false {
		t.Fatalf("unexpected action: %#v", action)
	}
	commands := action["commands"].([]interface{})
	command := commands[0].(map[string]interface{})
	if got := command["duration"]; got != jsonActionMinCommandDuration {
		t.Fatalf("duration = %#v, want minimum %.1f", got, jsonActionMinCommandDuration)
	}
	assertJSONActionFieldOrder(t, resp.Content)
}

func TestJSONActionOutputContractFallsBackToCandidateAction(t *testing.T) {
	resp := &ModelResponse{
		Model:        "judge",
		Content:      "ANSWER: Reference 1",
		HasToolCalls: true,
		Raw:          []byte(`{"choices":[{"message":{"tool_calls":[{"id":"call_1"}]}}]}`),
	}
	candidates := []*ModelResponse{{
		Model:   "worker-a",
		Content: `{"analysis":"read source","plan":"inspect inputs","commands":[{"keystrokes":"cat /app/decomp.c\n","duration":0.5}],"task_complete":false}`,
	}}

	applyJSONActionOutputContract(terminalActionOutputContractSpecForTest(), resp, candidates)

	var action map[string]interface{}
	if err := json.Unmarshal([]byte(resp.Content), &action); err != nil {
		t.Fatalf("fallback content is not JSON: %v\n%s", err, resp.Content)
	}
	if resp.HasToolCalls {
		t.Fatal("JSON action fallback should clear tool-call formatting")
	}
	if resp.Raw != nil {
		t.Fatal("JSON action fallback should clear stale raw tool-call payload")
	}
	commands, ok := action["commands"].([]interface{})
	if !ok || len(commands) != 1 {
		t.Fatalf("unexpected commands: %#v", action["commands"])
	}
	assertJSONActionFieldOrder(t, resp.Content)
}

func TestJSONActionOutputContractSkipsWithoutTypedSpec(t *testing.T) {
	original := `{"analysis":"inspect","plan":"read file","commands":[{"keystrokes":"cat /app/decomp.c\n","duration":0.1}],"task_complete":false}`
	resp := &ModelResponse{
		Model:   "worker",
		Content: original,
	}

	applyJSONActionOutputContract(nil, resp, nil)

	if resp.Content != original {
		t.Fatalf("content changed without explicit directive: %s", resp.Content)
	}
}

func TestHandoffPromptDoesNotInjectEvalSpecificActionContract(t *testing.T) {
	prompt := "Here are the answers the other agent provided.\n\nANSWER: Reference 2\n\nContinue working on this task from where the previous agent left off. You can no longer ask questions. Please follow the spec to interact with the terminal."

	withContract := textWithOutputContract(prompt, "")

	if withContract != prompt {
		t.Fatalf("handoff prompt should not receive implicit eval contract:\n%s", withContract)
	}
}

func TestDecisionOutputContractInjectsActionContract(t *testing.T) {
	prompt := "Continue working on this task from where the previous agent left off."

	withContract := textWithOutputContract(prompt, terminalActionOutputContractForTest())

	if !strings.Contains(withContract, `"analysis"`) ||
		!strings.Contains(withContract, `"plan"`) ||
		!strings.Contains(withContract, `"commands"`) ||
		!strings.Contains(withContract, `"task_complete"`) {
		t.Fatalf("configured action contract was not injected:\n%s", withContract)
	}
	if !requestsJSONAction(terminalActionOutputContractSpecForTest()) {
		t.Fatal("configured action spec should request JSON action output")
	}
}

func TestReferenceSelectionOutputContractDereferencesExplicitPostprocess(t *testing.T) {
	resp := &ModelResponse{
		Model:   "judge",
		Content: "2",
	}
	candidates := []*ModelResponse{
		{Model: "worker-a", Content: "Partial summary from worker A."},
		{Model: "worker-b", Content: "Major actions completed: inspected /app/decomp.c and checked /app/data.txt size."},
	}

	applyReferenceSelectionOutputContract(referenceDereferenceOutputContractSpecForTest(), resp, candidates)

	if resp.Content != candidates[1].Content {
		t.Fatalf("content = %q, want selected candidate %q", resp.Content, candidates[1].Content)
	}
}

func TestReferenceSelectionOutputContractDoesNotDereferenceWithoutSpec(t *testing.T) {
	resp := &ModelResponse{
		Model:   "judge",
		Content: "ANSWER: Reference 1",
	}
	candidates := []*ModelResponse{{Model: "worker-a", Content: "A"}}

	applyReferenceSelectionOutputContract(nil, resp, candidates)

	if resp.Content != "ANSWER: Reference 1" {
		t.Fatalf("unconfigured prompt should not dereference references, got %q", resp.Content)
	}
}

func TestReMoMSynthesisPromptKeepsConfiguredActionContractLast(t *testing.T) {
	l := NewReMoMLooper(&config.LooperConfig{})
	cfg := &config.ReMoMAlgorithmConfig{
		SynthesisTemplate: `Pick the best reference.

Original Problem:
{{.OriginalContent}}

Reference Responses:
{{range $i, $resp := .ReferenceResponses}}
Reference {{add $i 1}}:
{{$resp.Content}}
{{end}}

Return only ANSWER: Reference N.`,
	}
	original := textWithOutputContract(
		"Here are the answers the other agent provided.\n\nANSWER: Reference 2\n\nContinue working on this task from where the previous agent left off. You can no longer ask questions. Please follow the spec to interact with the terminal.",
		terminalActionOutputContractForTest(),
	)

	prompt, err := l.buildSynthesisPrompt(cfg, original, []*ModelResponse{{
		Model:   "worker-a",
		Content: "ANSWER: Reference 1",
	}})
	if err != nil {
		t.Fatalf("buildSynthesisPrompt returned error: %v", err)
	}

	answerInstruction := strings.LastIndex(prompt, "Return only ANSWER")
	actionInstruction := strings.LastIndex(prompt, "Respond with only a valid JSON object")
	if answerInstruction < 0 || actionInstruction < 0 {
		t.Fatalf("missing expected instructions in prompt:\n%s", prompt)
	}
	if actionInstruction < answerInstruction {
		t.Fatalf("configured action contract should be appended after conflicting synthesis instruction:\n%s", prompt)
	}
}

func singleChoiceOutputContractForTest() string {
	return `Return exactly one final multiple-choice answer letter: A, B, C, or D. Do not include prose, labels, or punctuation.`
}

func singleChoiceOutputContractSpecForTest() *config.OutputContractSpec {
	return &config.OutputContractSpec{
		Type: config.OutputContractTypeChoice,
		ChoiceSet: &config.OutputContractChoiceSetSpec{
			Values: []string{"A", "B", "C", "D"},
		},
		Render: &config.OutputContractRenderSpec{
			Mode: config.OutputContractRenderModeValue,
		},
	}
}

func terminalActionOutputContractForTest() string {
	return `Respond with only a valid JSON object using exactly this field order:
{
  "analysis": "Analyze the current terminal state and what remains to be done.",
  "plan": "Describe the next terminal actions.",
  "commands": [
    {
      "keystrokes": "the exact keystrokes to send to the terminal, ending commands with \n",
      "duration": 1.0
    }
  ],
  "task_complete": false
}
Do not include markdown fences, prose, or text outside the JSON object.`
}

func terminalActionOutputContractSpecForTest() *config.OutputContractSpec {
	return &config.OutputContractSpec{
		Type: config.OutputContractTypeStructuredJSON,
		JSONSchema: &config.OutputContractJSONSchemaSpec{
			SchemaRef: config.OutputContractJSONTerminalActionV1,
		},
		Extract: &config.OutputContractExtractSpec{
			Mode: config.OutputContractExtractModeJSONObject,
			Sources: []string{
				config.OutputContractExtractSourceContent,
				config.OutputContractExtractSourceCandidateResponses,
			},
		},
	}
}

func referenceDereferenceOutputContractSpecForTest() *config.OutputContractSpec {
	return &config.OutputContractSpec{
		Type: config.OutputContractTypeReferenceSelect,
		Reference: &config.OutputContractReferenceSpec{
			Source:   config.OutputContractExtractSourceCandidateResponses,
			IDFormat: config.OutputContractReferenceIDFormatIndex,
		},
		Extract: &config.OutputContractExtractSpec{
			Mode:    config.OutputContractExtractModeExact,
			Sources: []string{config.OutputContractExtractSourceContent},
		},
		Postprocess: []config.OutputContractPostprocess{{
			Type: config.OutputContractPostprocessDereferenceSelectedReference,
		}},
	}
}

func assertJSONActionFieldOrder(t *testing.T, content string) {
	t.Helper()
	analysis := strings.Index(content, `"analysis"`)
	plan := strings.Index(content, `"plan"`)
	commands := strings.Index(content, `"commands"`)
	taskComplete := strings.Index(content, `"task_complete"`)
	if analysis < 0 || plan < 0 || commands < 0 || taskComplete < 0 {
		t.Fatalf("missing required field in JSON action: %s", content)
	}
	if analysis >= plan || plan >= commands || commands >= taskComplete {
		t.Fatalf("JSON action fields are not in expected order: %s", content)
	}
}

func TestWorkflowSingleChoiceFallbackUsesWorkerMajorityWhenFinalHasNoAnswer(t *testing.T) {
	resp := &ModelResponse{Model: "gpt-final", Content: ""}
	stepResults := []workflowStepResult{{
		responses: []*ModelResponse{
			{Model: "model-a", Content: "B"},
			{Model: "model-b", Content: "B"},
			{Model: "model-c", Content: "C"},
		},
	}}

	applyWorkflowSingleChoiceFallback(
		singleChoiceOutputContractSpecForTest(),
		resp,
		stepResults,
	)

	if resp.Content != "B" {
		t.Fatalf("fallback content = %q, want B", resp.Content)
	}
}

func TestWorkflowSingleChoiceFallbackDoesNotOverrideFinalAnswer(t *testing.T) {
	resp := &ModelResponse{Model: "gpt-final", Content: "A"}
	stepResults := []workflowStepResult{{
		responses: []*ModelResponse{{Model: "model-a", Content: "B"}},
	}}

	applyWorkflowSingleChoiceFallback(
		singleChoiceOutputContractSpecForTest(),
		resp,
		stepResults,
	)

	if resp.Content != "A" {
		t.Fatalf("fallback overrode final content: %q", resp.Content)
	}
}

func TestValidateWorkflowPlanRejectsFutureAccessList(t *testing.T) {
	plan := &workflowPlan{Steps: []workflowPlanStep{
		{
			ID:         "first",
			Models:     []string{"worker-a"},
			Prompt:     "solve",
			AccessList: []string{"second"},
		},
		{
			ID:     "second",
			Models: []string{"worker-a"},
			Prompt: "review",
		},
	}}
	err := validateWorkflowPlan(plan, []string{"worker-a"}, workflowsExecutionConfig{MaxSteps: 3, MaxParallel: 1})
	if err == nil || !strings.Contains(err.Error(), "unknown or future step") {
		t.Fatalf("expected future access_list error, got %v", err)
	}
}

func TestValidateWorkflowPlanAllowsPreviousAgentAccessList(t *testing.T) {
	plan := &workflowPlan{Steps: []workflowPlanStep{
		{
			ID:     "solve",
			Models: []string{"worker-a", "worker-b"},
			Prompt: "solve",
		},
		{
			ID:         "review",
			Models:     []string{"worker-a"},
			Prompt:     "review one worker",
			AccessList: []string{"solve:1:worker-b"},
		},
	}}
	err := validateWorkflowPlan(plan, []string{"worker-a", "worker-b"}, workflowsExecutionConfig{MaxSteps: 3, MaxParallel: 2})
	if err != nil {
		t.Fatalf("validateWorkflowPlan rejected previous agent access_list: %v", err)
	}
}

func TestWorkflowsFinalToolCallsArePreserved(t *testing.T) {
	raw := []byte(`{
		"id":"chatcmpl-tool",
		"object":"chat.completion",
		"created":0,
		"model":"qwen-coordinator",
		"choices":[{
			"index":0,
			"message":{
				"role":"assistant",
				"content":null,
				"tool_calls":[{
					"id":"call_1",
					"type":"function",
					"function":{"name":"lookup","arguments":"{\"query\":\"flow\"}"}
				}]
			},
			"finish_reason":"tool_calls"
		}]
	}`)
	finalResp := &ModelResponse{
		Model:        "qwen-coordinator",
		Raw:          raw,
		HasToolCalls: true,
	}
	trace := &workflowTrace{Mode: config.WorkflowModeDynamic}
	cfg := workflowsExecutionConfig{IncludeIntermediateResponses: true}

	resp, err := formatWorkflowJSONResponse(finalResp, []string{"qwen-coordinator"}, 1, trace, TokenUsage{}, cfg)
	if err != nil {
		t.Fatalf("formatWorkflowJSONResponse failed: %v", err)
	}
	var body map[string]interface{}
	if unmarshalErr := json.Unmarshal(resp.Body, &body); unmarshalErr != nil {
		t.Fatalf("response body is not JSON: %v", unmarshalErr)
	}
	choices := body["choices"].([]interface{})
	choice := choices[0].(map[string]interface{})
	message := choice["message"].(map[string]interface{})
	if _, ok := message["tool_calls"].([]interface{}); !ok {
		t.Fatalf("tool_calls missing from response: %s", string(resp.Body))
	}
	if choice["finish_reason"] != "tool_calls" {
		t.Fatalf("finish_reason = %v", choice["finish_reason"])
	}
	if _, ok := body["flow"]; !ok {
		t.Fatalf("flow trace missing from response: %s", string(resp.Body))
	}

	streaming, err := formatWorkflowStreamingResponse(finalResp, []string{"qwen-coordinator"}, 1, trace, TokenUsage{}, cfg)
	if err != nil {
		t.Fatalf("formatWorkflowStreamingResponse failed: %v", err)
	}
	if !strings.Contains(string(streaming.Body), `"tool_calls"`) {
		t.Fatalf("streaming response missing tool_calls: %s", string(streaming.Body))
	}
	if !strings.Contains(string(streaming.Body), `"finish_reason":"tool_calls"`) {
		t.Fatalf("streaming response missing tool_calls finish: %s", string(streaming.Body))
	}
}

func TestWorkflowPlannerPromptIncludesMultipleChoicePlanningRule(t *testing.T) {
	prompt := buildWorkflowPlannerPrompt(
		"Answer the following multiple choice question.",
		[]string{"model-a", "model-b"},
		workflowsExecutionConfig{MaxSteps: 3, MaxParallel: 2},
		singleChoiceOutputContractSpecForTest(),
	)

	for _, want := range []string{
		"multiple-choice benchmark",
		"parallel independent-solver step",
	} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("planner prompt missing %q:\n%s", want, prompt)
		}
	}
}

func workflowResponseContent(t *testing.T, body []byte) string {
	t.Helper()
	var payload struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("response body is not JSON: %v", err)
	}
	if len(payload.Choices) == 0 {
		t.Fatalf("response body has no choices: %s", string(body))
	}
	return payload.Choices[0].Message.Content
}
