package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// finalizeLooperResponseExtensions is the final gate for every Looper response,
// including native multi-choice streams. Required answer bytes are never
// truncated; optional evidence is omitted atomically if it cannot be represented
// in the target protocol or would exceed the final body/frame limit.
func finalizeLooperResponseExtensions(engine *protocolcodec.Engine, body []byte, extensions []looper.ResponseExtension, ctx *RequestContext, streaming bool) ([]byte, error) {
	if err := engine.ValidateEncodedResponse(body, streaming); err != nil {
		return nil, err
	}
	target := ctx.SourceFormat
	if target == "" {
		target = llmprotocol.OpenAIChatV1
	}
	for _, extension := range extensions {
		field, value := extension.Name(), extension.JSON()
		if field == "flow" && !looperShouldRestoreWorkflowTrace(ctx, value) {
			continue
		}
		reason := ""
		if target != llmprotocol.OpenAIChatV1 {
			reason = "router_extension_unsupported_protocol"
		} else {
			candidate := restoreLooperFieldJSON(body, value, field)
			if streaming {
				candidate = restoreLooperFieldSSE(body, value, field)
			}
			if err := engine.ValidateEncodedResponse(candidate, streaming); err != nil {
				reason = "router_extension_size_limit"
			} else {
				body = candidate
			}
		}
		if reason != "" {
			// Keep optional-evidence disposition visible within the header budget.
			ctx.ProtocolDiagnostics = append(llmprotocol.Diagnostics{{
				Source: llmprotocol.OpenAIChatV1, Target: target, Field: field,
				Action: llmprotocol.DiagnosticDropped, Reason: reason,
			}}, ctx.ProtocolDiagnostics...)
		}
	}
	return body, nil
}
