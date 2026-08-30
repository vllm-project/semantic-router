package extproc

import (
	"bytes"
	"encoding/json"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// toolObservability is captured during pre-dispatch selection and copied onto
// the final ExtProc response after Anthropic / specified / auto dispatch.
type toolObservability struct {
	Strategy   string
	Confidence string
	LatencyMs  string
}

// applyToolSelectionBeforeDispatch runs semantic tool selection once on the
// protocol-neutral IR before provider translation or backend dispatch.
func (r *OpenAIRouter) applyToolSelectionBeforeDispatch(
	openAIRequest *openai.ChatCompletionNewParams,
	response *ext_proc.ProcessingResponse,
	ctx *RequestContext,
) {
	var anthropicWire []byte
	if ctx != nil && ctx.ClientProtocol == config.ClientProtocolAnthropic {
		anthropicWire = append([]byte(nil), ctx.workingRequestBody()...)
	}
	before := snapshotToolSelection(openAIRequest)
	r.handleToolSelectionForRequest(openAIRequest, response, ctx)
	if len(anthropicWire) > 0 {
		ctx.setWorkingRequestBody(anthropicWire)
		return
	}
	if !toolSelectionChanged(before, openAIRequest) {
		return
	}
	persistSelectedToolsToWorkingBody(openAIRequest, ctx)
}

// persistSelectedToolsToWorkingBody copies the IR's selected tools onto the
// OpenAI-shaped working body so specified/auto dispatch can emit them. Do not
// derive this from the temporary ExtProc stub: handleToolSelectionForRequest
// may filter the IR without writing a body mutation.
func persistSelectedToolsToWorkingBody(
	openAIRequest *openai.ChatCompletionNewParams,
	ctx *RequestContext,
) {
	if ctx == nil || openAIRequest == nil {
		return
	}
	serializedRequest, err := serializeOpenAIRequestWithStream(openAIRequest, ctx.ExpectStreamingResponse)
	if err != nil {
		logging.Errorf("Error serializing selected tools before dispatch: %v", err)
		return
	}
	base := ctx.workingRequestBody()
	if len(base) == 0 {
		ctx.setWorkingRequestBody(serializedRequest)
		return
	}
	modifiedBody, err := mergeSerializedToolFields(base, serializedRequest, toolFieldsForUpdate(ctx))
	if err != nil {
		logging.Errorf("Error merging selected tools before dispatch: %v", err)
		return
	}
	ctx.setWorkingRequestBody(modifiedBody)
}

type toolSelectionSnapshot struct {
	names      []string
	toolChoice []byte
}

func snapshotToolSelection(openAIRequest *openai.ChatCompletionNewParams) toolSelectionSnapshot {
	return toolSelectionSnapshot{
		names:      openaiToolNames(openAIRequest),
		toolChoice: marshalToolChoice(openAIRequest),
	}
}

func toolSelectionChanged(before toolSelectionSnapshot, openAIRequest *openai.ChatCompletionNewParams) bool {
	if openAIRequest == nil {
		return false
	}
	after := snapshotToolSelection(openAIRequest)
	if len(before.names) != len(after.names) {
		return true
	}
	for i := range before.names {
		if before.names[i] != after.names[i] {
			return true
		}
	}
	return !bytes.Equal(before.toolChoice, after.toolChoice)
}

func openaiToolNames(openAIRequest *openai.ChatCompletionNewParams) []string {
	if openAIRequest == nil {
		return nil
	}
	names := make([]string, 0, len(openAIRequest.Tools))
	for _, tool := range openAIRequest.Tools {
		names = append(names, tool.Function.Name)
	}
	return names
}

func marshalToolChoice(openAIRequest *openai.ChatCompletionNewParams) []byte {
	if openAIRequest == nil {
		return nil
	}
	raw, err := json.Marshal(openAIRequest.ToolChoice)
	if err != nil {
		return nil
	}
	return raw
}

func appendToolObservabilityHeaders(setHeaders *[]*core.HeaderValueOption, ctx *RequestContext) {
	if setHeaders == nil || ctx == nil || ctx.ToolObservability == nil {
		return
	}
	obs := ctx.ToolObservability
	appendRawHeader(setHeaders, headers.VSRToolsStrategy, obs.Strategy)
	appendRawHeader(setHeaders, headers.VSRToolsConfidence, obs.Confidence)
	appendRawHeader(setHeaders, headers.VSRToolsLatencyMs, obs.LatencyMs)
}

func appendRawHeader(setHeaders *[]*core.HeaderValueOption, key, value string) {
	*setHeaders = append(*setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			Value:    value,
			RawValue: []byte(value),
		},
	})
}
