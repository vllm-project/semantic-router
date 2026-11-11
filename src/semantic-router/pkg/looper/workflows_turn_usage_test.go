package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// withUsage adds a usage block to a chat completion body.
func withUsage(t *testing.T, body []byte, prompt, completion int64) []byte {
	t.Helper()
	var parsed map[string]interface{}
	require.NoError(t, json.Unmarshal(body, &parsed))
	parsed["usage"] = map[string]int64{"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
	out, err := json.Marshal(parsed)
	require.NoError(t, err)
	return out
}

// requireWorkflowUsage checks the usage a Flow turn reports, to the client and
// to the Router.
func requireWorkflowUsage(t *testing.T, resp *Response, prompt, completion int64) {
	t.Helper()
	var body struct {
		Usage TokenUsage `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	want := TokenUsage{PromptTokens: prompt, CompletionTokens: completion, TotalTokens: prompt + completion}
	require.Equal(t, want, body.Usage, "client usage")
	require.Equal(t, want, resp.Usage, "Router usage")
}

func workflowModelServer(t *testing.T, answer func(model string, afterTool bool) []byte) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model    string                   `json:"model"`
			Messages []map[string]interface{} `json:"messages"`
		}
		require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
		body := answer(payload.Model, payloadHasToolMessage(payload.Messages))
		if body == nil {
			http.Error(w, `{"error":"unavailable"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write(body)
	}))
}

// Each turn of a Flow that stops for a tool call reports the calls it made:
// the resumed turn does not bill the first turn's calls again.
func TestWorkflowToolTurnsReportOnlyTheirOwnCalls(t *testing.T) {
	server := workflowModelServer(t, func(model string, afterTool bool) []byte {
		switch {
		case model == "thinker-model":
			return withUsage(t, workflowChatCompletion(model, "evidence"), 100, 10)
		case model == "worker-model" && afterTool:
			return withUsage(t, workflowChatCompletion(model, "worker done"), 30, 3)
		case model == "worker-model":
			return withUsage(t, workflowToolCallCompletion(model, "call_1"), 20, 2)
		default:
			return withUsage(t, workflowChatCompletion(model, "final"), 40, 4)
		}
	})
	defer server.Close()
	cfg := workflowToolLooperConfig(server.URL, t.TempDir())

	first, err := NewWorkflowsLooper(cfg).Execute(context.Background(), workflowTwoStepToolLooperRequest(workflowToolTestRequest()))
	require.NoError(t, err)
	requireWorkflowUsage(t, first, 120, 12)
	assistant, toolCallID := assistantToolMessageFromResponse(t, first.Body)
	second, err := NewWorkflowsLooper(cfg).Execute(context.Background(), workflowTwoStepToolLooperRequest(workflowToolResumeRequest(t, assistant, toolCallID)))
	require.NoError(t, err)
	requireWorkflowUsage(t, second, 70, 7)
}

// With on_error: skip, a failed final synthesis serves a worker's answer,
// whose call is counted once.
func TestWorkflowSkipFallbackCountsTheWorkerCallOnce(t *testing.T) {
	server := workflowModelServer(t, func(model string, _ bool) []byte {
		if model == "verifier-model" {
			return nil
		}
		return withUsage(t, workflowChatCompletion(model, "worker answer"), 100, 10)
	})
	defer server.Close()
	req := workflowToolLooperRequest(workflowTestRequest())
	req.Algorithm.OnError = config.WorkflowOnErrorSkip

	resp, err := NewWorkflowsLooper(workflowToolLooperConfig(server.URL, t.TempDir())).Execute(context.Background(), req)
	require.NoError(t, err)
	requireWorkflowUsage(t, resp, 100, 10)
}

// output_contract_spec shapes the final answer of a turn resumed after a tool
// call as it shapes a first turn's.
func TestWorkflowResumedAnswerFollowsTheOutputContract(t *testing.T) {
	final := "Here is my action:\n{\"analysis\":\"done\",\"plan\":\"\",\"commands\":[],\"task_complete\":true}"
	spec := &config.OutputContractSpec{
		Type:       config.OutputContractTypeStructuredJSON,
		JSONSchema: &config.OutputContractJSONSchemaSpec{SchemaRef: config.OutputContractJSONTerminalActionV1},
		Extract:    &config.OutputContractExtractSpec{Mode: config.OutputContractExtractModeJSONObject},
	}
	finalContent := func(callsTool bool) string {
		server := workflowModelServer(t, func(model string, afterTool bool) []byte {
			switch {
			case model == "worker-model" && callsTool && !afterTool:
				return workflowToolCallCompletion(model, "call_1")
			case model == "worker-model":
				return workflowChatCompletion(model, "worker")
			default:
				return workflowChatCompletion(model, final)
			}
		})
		defer server.Close()
		cfg := workflowToolLooperConfig(server.URL, t.TempDir())
		req := workflowToolLooperRequest(workflowToolTestRequest())
		req.OutputContractSpec = spec
		resp, err := NewWorkflowsLooper(cfg).Execute(context.Background(), req)
		require.NoError(t, err)
		if callsTool {
			assistant, toolCallID := assistantToolMessageFromResponse(t, resp.Body)
			req = workflowToolLooperRequest(workflowToolResumeRequest(t, assistant, toolCallID))
			req.OutputContractSpec = spec
			resp, err = NewWorkflowsLooper(cfg).Execute(context.Background(), req)
			require.NoError(t, err)
		}
		var body struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		require.NoError(t, json.Unmarshal(resp.Body, &body))
		return body.Choices[0].Message.Content
	}

	firstTurn := finalContent(false)
	require.NotEqual(t, final, firstTurn, "the first turn's answer is normalized")
	require.Equal(t, firstTurn, finalContent(true))
}

// With on_error: skip, a turn resumed after a tool call serves a worker's
// answer when the final synthesis fails, as a first turn does.
func TestWorkflowResumedTurnFallsBackOnSkip(t *testing.T) {
	server := workflowModelServer(t, func(model string, afterTool bool) []byte {
		switch {
		case model == "verifier-model":
			return nil
		case afterTool:
			return withUsage(t, workflowChatCompletion(model, "worker answer"), 30, 3)
		default:
			return withUsage(t, workflowToolCallCompletion(model, "call_1"), 20, 2)
		}
	})
	defer server.Close()
	cfg := workflowToolLooperConfig(server.URL, t.TempDir())
	req := workflowToolLooperRequest(workflowToolTestRequest())
	req.Algorithm.OnError = config.WorkflowOnErrorSkip
	first, err := NewWorkflowsLooper(cfg).Execute(context.Background(), req)
	require.NoError(t, err)
	assistant, toolCallID := assistantToolMessageFromResponse(t, first.Body)
	req = workflowToolLooperRequest(workflowToolResumeRequest(t, assistant, toolCallID))
	req.Algorithm.OnError = config.WorkflowOnErrorSkip

	second, err := NewWorkflowsLooper(cfg).Execute(context.Background(), req)
	require.NoError(t, err)
	require.Contains(t, string(second.Body), `"content":"worker answer"`)
	requireWorkflowUsage(t, second, 30, 3)
}

// With on_error: skip, a final synthesizer that called a tool and then fails
// on the tool result leaves a worker's answer, as a first turn does.
func TestWorkflowResumedFinalAgentFallsBackOnSkip(t *testing.T) {
	server := workflowModelServer(t, func(model string, afterTool bool) []byte {
		switch {
		case model == "verifier-model" && afterTool:
			return nil
		case model == "verifier-model":
			return withUsage(t, workflowToolCallCompletion(model, "call_final"), 20, 2)
		default:
			return withUsage(t, workflowChatCompletion(model, "worker answer"), 30, 3)
		}
	})
	defer server.Close()
	cfg := workflowToolLooperConfig(server.URL, t.TempDir())
	req := workflowToolLooperRequest(workflowToolTestRequest())
	req.Algorithm.OnError = config.WorkflowOnErrorSkip
	first, err := NewWorkflowsLooper(cfg).Execute(context.Background(), req)
	require.NoError(t, err)
	assistant, toolCallID := assistantToolMessageFromResponse(t, first.Body)
	req = workflowToolLooperRequest(workflowToolResumeRequest(t, assistant, toolCallID))
	req.Algorithm.OnError = config.WorkflowOnErrorSkip

	second, err := NewWorkflowsLooper(cfg).Execute(context.Background(), req)
	require.NoError(t, err)
	require.Contains(t, string(second.Body), `"content":"worker answer"`)
}
