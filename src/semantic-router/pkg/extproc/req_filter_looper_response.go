package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// createLooperResponse creates an ImmediateResponse from looper output.
func (r *OpenAIRouter) createLooperResponse(
	resp *looper.Response,
	reqCtx *RequestContext,
) *ext_proc.ProcessingResponse {
	response, _, _, err := r.prepareLooperResponse(resp, reqCtx)
	if err != nil {
		return r.createErrorResponse(502, "Looper returned an invalid response")
	}
	return response
}

func (r *OpenAIRouter) prepareLooperResponse(
	resp *looper.Response,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, *llmprotocol.Response, []byte, error) {
	if resp == nil || reqCtx == nil {
		return nil, nil, nil, fmt.Errorf("looper response context is unavailable")
	}
	engine, err := r.protocolEngine()
	if err != nil {
		return nil, nil, nil, err
	}
	return r.prepareLooperResponseWithEngine(resp, reqCtx, engine)
}

//nolint:gocognit,cyclop,nestif // Buffered/native/translated streams converge at one final transport gate.
func (r *OpenAIRouter) prepareLooperResponseWithEngine(resp *looper.Response, reqCtx *RequestContext, engine *protocolcodec.Engine) (*ext_proc.ProcessingResponse, *llmprotocol.Response, []byte, error) {
	var err error
	target := reqCtx.SourceFormat
	if target == "" {
		target = llmprotocol.OpenAIChatV1
	}
	var semantic *llmprotocol.Response
	var body []byte
	contentType := "application/json"
	// Only a router emitter can supply the separate extension channel. No field
	// is removed from provider data based on its JSON name.
	codecBody := resp.ProtocolBody()
	streaming := strings.Contains(strings.ToLower(resp.ContentType), "text/event-stream")
	if streaming && target == llmprotocol.OpenAIChatV1 && len(resp.BufferedBody) > 0 {
		semantic, body, err = prepareNativeLooperStream(engine, resp, reqCtx)
		if err != nil {
			return nil, nil, nil, err
		}
		contentType = "text/event-stream"
	} else if streaming {
		streamContext := llmprotocol.StreamContext{
			Context: reqCtx.TraceContext, Options: clientStreamOptions(reqCtx), PublicModel: resp.Model,
			ResponseID: responseObjectPublicID(reqCtx), PreviousResponseID: responseObjectPreviousID(reqCtx),
		}
		var mutation protocolcodec.StreamEventMutation
		namespaces := requestToolNamespaces(reqCtx)
		if streamContext.ResponseID != "" || len(namespaces) > 0 {
			mutation = func(event *llmprotocol.Event) error {
				if streamContext.ResponseID != "" {
					event.ResponseID = streamContext.ResponseID
				}
				event.ToolCall = namespaces.RestoreCall(event.ToolCall)
				return nil
			}
		}
		stream, streamErr := engine.NewStreamWithMutation(
			llmprotocol.OpenAIChatV1,
			target,
			streamContext, mutation,
		)
		if streamErr != nil {
			return nil, nil, nil, streamErr
		}
		state := &semanticResponseStreamState{
			usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
			items: make(map[int]*semanticStreamItem),
		}
		frames, events, diagnostics, streamErr := stream.Push(codecBody)
		reqCtx.ProtocolDiagnostics = append(reqCtx.ProtocolDiagnostics, diagnostics...)
		state.observe(events)
		for _, frame := range frames {
			body = append(body, frame...)
		}
		finalFrames, finalEvents, finalDiagnostics, finalErr := stream.Finalize(streamErr)
		reqCtx.ProtocolDiagnostics = append(reqCtx.ProtocolDiagnostics, finalDiagnostics...)
		state.observe(finalEvents)
		for _, frame := range finalFrames {
			body = append(body, frame...)
		}
		if streamErr != nil {
			return nil, nil, nil, streamErr
		}
		if finalErr != nil {
			return nil, nil, nil, finalErr
		}
		semantic, err = state.response()
		if err != nil {
			return nil, nil, nil, err
		}
		if semantic.Model != resp.Model {
			semantic.Model = resp.Model
			semantic.Generation++
		}
		contentType = "text/event-stream"
	} else {
		semantic, body, err = prepareBufferedLooperResponse(engine, resp, reqCtx, target)
		if err != nil {
			return nil, nil, nil, err
		}
		streaming = reqCtx.ExpectStreamingResponse
		if streaming {
			contentType = "text/event-stream"
		}
	}
	if headerValueCI(reqCtx, headers.VSRBenchExpectedConfigHash) != "" {
		reqCtx.BenchmarkModelUsage = r.benchmarkLooperUsage(resp, reqCtx)
	}
	body, err = finalizeLooperResponseExtensions(engine, body, resp.RouterExtensions(), reqCtx, streaming)
	if err != nil {
		return nil, nil, nil, err
	}
	reqCtx.SemanticResponse = semantic
	reqCtx.ImmediateResponseEncoded = true
	reqCtx.LooperTrace = newLooperTrace(resp)
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_OK},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: buildLooperResponseHeaders(contentType),
				},
				Body: body,
			},
		},
	}, semantic, body, nil
}

// buildLooperResponseHeaders writes the content-type of the immediate body
// and the keystone headers of the Looper path; the reply boundary adds the
// rest of the public set and the trace.
func buildLooperResponseHeaders(contentType string) []*core.HeaderValueOption {
	if contentType == "" {
		contentType = "application/json"
	}
	setHeaders := []*core.HeaderValueOption{newHeaderValueOption("content-type", contentType)}
	return append(setHeaders, httputil.KeystoneHeaderOptions(headers.ResponsePathLooper)...)
}

func newHeaderValueOption(key string, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
}

func looperShouldRestoreWorkflowTrace(reqCtx *RequestContext, flow json.RawMessage) bool {
	if looperIncludeIntermediateResponses(reqCtx) {
		return true
	}
	return looperIsWorkflowDecision(reqCtx) && looperWorkflowTraceHasFailedModels(flow)
}

func looperIsWorkflowDecision(reqCtx *RequestContext) bool {
	if reqCtx == nil || reqCtx.VSRSelectedDecision == nil {
		return false
	}
	alg := reqCtx.VSRSelectedDecision.Algorithm
	return alg != nil && alg.Type == config.DecisionAlgorithmWorkflows
}

func looperWorkflowTraceHasFailedModels(flow json.RawMessage) bool {
	var trace struct {
		FailedModels []json.RawMessage `json:"failed_models"`
	}
	if err := json.Unmarshal(flow, &trace); err != nil {
		return false
	}
	return len(trace.FailedModels) > 0
}

func looperIncludeIntermediateResponses(reqCtx *RequestContext) bool {
	if reqCtx == nil || reqCtx.VSRSelectedDecision == nil {
		return false
	}
	alg := reqCtx.VSRSelectedDecision.Algorithm
	if alg == nil || alg.Type != config.DecisionAlgorithmWorkflows {
		return false
	}
	if alg.Workflows == nil || alg.Workflows.IncludeIntermediateResponses == nil {
		return true
	}
	return *alg.Workflows.IncludeIntermediateResponses
}

func isLooperSSEBody(body []byte) bool {
	trimmed := bytes.TrimSpace(body)
	return bytes.HasPrefix(trimmed, []byte("data:")) || bytes.Contains(body, []byte("\ndata:"))
}

func restoreLooperFieldSSE(body []byte, flow json.RawMessage, field string) []byte {
	if len(flow) == 0 {
		return body
	}
	lines := bytes.Split(body, []byte("\n"))
	out := make([]byte, 0, len(body)+len(flow)+8)
	restored := false
	for i, line := range lines {
		if rest, ok := bytes.CutPrefix(line, []byte("data:")); ok && !restored {
			payload := bytes.TrimSpace(rest)
			if len(payload) > 0 && payload[0] == '{' && !bytes.Equal(payload, []byte("[DONE]")) {
				payload = restoreLooperFieldJSON(payload, flow, field)
				line = append([]byte("data: "), payload...)
				restored = true
			}
		}
		out = append(out, line...)
		if i < len(lines)-1 {
			out = append(out, '\n')
		}
	}
	return out
}

func restoreLooperFieldJSON(body []byte, flow json.RawMessage, field string) []byte {
	if len(flow) == 0 {
		return body
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return body
	}
	obj[field] = flow
	restored, err := json.Marshal(obj)
	if err != nil {
		return body
	}
	return restored
}
