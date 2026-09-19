package extproc

import (
	"encoding/json"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var looperResponseTraceFields = [...]string{"flow", "fusion", "reasoning_mom_responses"}

func isolateLooperResponseTraces(body []byte) ([]byte, map[string]json.RawMessage) {
	traces := make(map[string]json.RawMessage)
	for _, field := range looperResponseTraceFields {
		var value json.RawMessage
		if isLooperSSEBody(body) {
			body, value = isolateLooperFieldFromSSE(body, field)
		} else {
			body, value = isolateLooperFieldFromJSON(body, field)
		}
		if len(value) > 0 {
			traces[field] = value
		}
	}
	return body, traces
}

func restoreLooperResponseTraces(body []byte, traces map[string]json.RawMessage, ctx *RequestContext) []byte {
	// Workflow traces retain their existing visibility rules. Fusion and ReMoM
	// already apply their public trace controls when they construct the response.
	body = restoreLooperWorkflowTrace(body, traces["flow"], ctx)
	if ctx.SourceFormat != "" && ctx.SourceFormat != llmprotocol.OpenAIChatV1 {
		return body
	}
	for _, field := range looperResponseTraceFields[1:] {
		value := traces[field]
		if len(value) == 0 {
			continue
		}
		if isLooperSSEBody(body) {
			body = restoreLooperFieldSSE(body, value, field)
		} else {
			body = restoreLooperFieldJSON(body, value, field)
		}
	}
	return body
}
