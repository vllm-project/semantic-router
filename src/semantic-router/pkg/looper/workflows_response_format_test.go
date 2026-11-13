package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const workflowJSONSchemaFormat = `{"type":"json_schema","json_schema":{"name":"answer","strict":true,` +
	`"schema":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"],"additionalProperties":false}}}`

// withJSONSchemaFormat sets the client's response_format as request parsing
// leaves it: the json_schema variant with its schema.
func withJSONSchemaFormat(t *testing.T, req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	t.Helper()
	var format shared.ResponseFormatJSONSchemaParam
	if err := json.Unmarshal([]byte(workflowJSONSchemaFormat), &format); err != nil {
		t.Fatalf("parse the response format: %v", err)
	}
	req.ResponseFormat = openai.ChatCompletionNewParamsResponseFormatUnion{OfJSONSchema: &format}
	return req
}

// TestWorkflowsToolResumeKeepsClientJSONSchema holds every call of a Flow that
// pauses for a client tool to the client's json_schema response_format: the
// worker's tool call, the worker's turn after the tool result, which starts
// from the request the paused state stored, and the final.
func TestWorkflowsToolResumeKeepsClientJSONSchema(t *testing.T) {
	var (
		mu      sync.Mutex
		formats = map[string][]string{}
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload struct {
			Model          string                   `json:"model"`
			Messages       []map[string]interface{} `json:"messages"`
			ResponseFormat json.RawMessage          `json:"response_format"`
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Errorf("decode request: %v", err)
			return
		}
		mu.Lock()
		formats[payload.Model] = append(formats[payload.Model], string(payload.ResponseFormat))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch {
		case payload.Model == "worker-model" && !payloadHasToolMessage(payload.Messages):
			_, _ = w.Write(workflowToolCallCompletion("worker-model", "call_lookup"))
		default:
			_, _ = w.Write(workflowChatCompletion(payload.Model, `{"value":"42"}`))
		}
	}))
	defer server.Close()

	looperCfg := workflowToolLooperConfig(server.URL, t.TempDir())
	firstResp, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(),
		workflowToolLooperRequest(withJSONSchemaFormat(t, workflowToolTestRequest())))
	if err != nil {
		t.Fatalf("first Execute failed: %v", err)
	}
	assistantMessage, toolCallID := assistantToolMessageFromResponse(t, firstResp.Body)
	resumeReq := withJSONSchemaFormat(t, workflowToolResumeRequest(t, assistantMessage, toolCallID))
	if _, err := NewWorkflowsLooper(looperCfg).Execute(context.Background(), workflowToolLooperRequest(resumeReq)); err != nil {
		t.Fatalf("resume Execute failed: %v", err)
	}

	require.Len(t, formats["worker-model"], 2, "the worker's tool call and its turn after the tool result")
	require.Len(t, formats["verifier-model"], 1, "the final")
	for model, calls := range formats {
		for i, format := range calls {
			assert.JSONEq(t, workflowJSONSchemaFormat, format, "%s call %d", model, i+1)
		}
	}
}
