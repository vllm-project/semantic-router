package routerreplay

import (
	"encoding/json"
	"fmt"
	"maps"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
)

// Tool-trace step sources. A step's source names the body capture switch that
// covers its text: capture_request_body for request steps,
// capture_response_body for response and stream steps.
const (
	ToolTraceSourceRequest  = "request"
	ToolTraceSourceResponse = "response"
	ToolTraceSourceStream   = "stream"
)

// fallbackExecutionRecordKey is the outcome metadata entry that holds a
// fallback execution record, whose failed attempts carry provider replies.
const fallbackExecutionRecordKey = "execution_record"

// omitUncapturedText applies the body capture switches to every other field
// that holds request or response text. With capture_request_body off, the
// prompt, the tool definitions and each request step's text, tool arguments
// and tool results are dropped. With capture_response_body off, the same goes
// for response steps and the hallucination span excerpts. Step types, roles,
// tool names, call IDs and statuses stay. Left-out text is simply absent and
// not flagged content_redacted, which keeps marking text a reader's role may
// not see and tool results kept out by capture_tool_results.
func omitUncapturedText(rec *RoutingRecord, policy recorderPolicy) {
	if !policy.captureRequestBody {
		rec.Prompt, rec.PromptTruncated = "", false
		rec.ToolDefinitions, rec.ToolDefinitionsTruncated = "", false
	}
	rec.HallucinationSpans, rec.HallucinationSpanDetails = omitUncapturedSpans(
		rec.HallucinationSpans, rec.HallucinationSpanDetails, policy)
	omitUncapturedStepText(rec.ToolTrace, policy)
}

// capturesStep reports whether a trace step's text may be stored. A step from
// any other source than the known ones is kept only when both sides are.
func (p recorderPolicy) capturesStep(source string) bool {
	switch source {
	case ToolTraceSourceRequest:
		return p.captureRequestBody
	case ToolTraceSourceResponse, ToolTraceSourceStream:
		return p.captureResponseBody
	default:
		return p.captureRequestBody && p.captureResponseBody
	}
}

func omitUncapturedStepText(trace *ToolTrace, policy recorderPolicy) {
	if trace == nil {
		return
	}
	for i := range trace.Steps {
		step := &trace.Steps[i]
		if !policy.capturesStep(step.Source) {
			step.Text, step.Arguments, step.RawArguments, step.Output, step.RawOutput = "", "", "", "", ""
			step.Truncated, step.RawTruncated = false, false
		}
	}
}

// omitUncapturedSpans drops the detector's excerpts of the answer when the
// response is not captured. Span offsets and severities stay.
func omitUncapturedSpans(spans []string, details []HallucinationSpan, policy recorderPolicy) ([]string, []HallucinationSpan) {
	if policy.captureResponseBody {
		return spans, details
	}
	details = append([]HallucinationSpan(nil), details...)
	for i := range details {
		details[i].Text, details[i].Explanation = "", ""
	}
	return nil, details
}

// omitUncapturedReplies keeps provider replies out of a fallback execution
// record when the response is not captured: a failed attempt keeps its status
// with a fixed reason in place of the reply. A record that cannot be read is
// dropped rather than stored as it came.
func omitUncapturedReplies(outcome Outcome, policy recorderPolicy) Outcome {
	raw, ok := outcome.Metadata[fallbackExecutionRecordKey]
	if policy.captureResponseBody || !ok {
		return outcome
	}
	outcome.Metadata = maps.Clone(outcome.Metadata)
	delete(outcome.Metadata, fallbackExecutionRecordKey)
	var execution fallback.ExecutionRecord
	if json.Unmarshal([]byte(raw), &execution) != nil {
		return outcome
	}
	for i := range execution.Attempts {
		if execution.Attempts[i].ErrorMessage != "" {
			execution.Attempts[i].ErrorMessage = fmt.Sprintf("upstream returned status %d", execution.Attempts[i].StatusCode)
		}
	}
	if encoded, err := json.Marshal(execution); err == nil {
		outcome.Metadata[fallbackExecutionRecordKey] = string(encoded)
	}
	return outcome
}
