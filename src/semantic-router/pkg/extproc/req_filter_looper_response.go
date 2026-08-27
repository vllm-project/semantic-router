package extproc

import (
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
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

//nolint:gocognit,cyclop,nestif // Looper output normalizes buffered and streaming terminal variants at one boundary.
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
	target := reqCtx.SourceFormat
	if target == "" {
		target = llmprotocol.OpenAIChatV1
	}
	var semantic *llmprotocol.Response
	var body []byte
	contentType := "application/json"
	if strings.Contains(strings.ToLower(resp.ContentType), "text/event-stream") {
		stream, streamErr := engine.NewStream(
			llmprotocol.OpenAIChatV1,
			target,
			llmprotocol.StreamContext{
				Context: reqCtx.TraceContext, PublicModel: resp.Model,
			},
		)
		if streamErr != nil {
			return nil, nil, nil, streamErr
		}
		state := &semanticResponseStreamState{
			usage: llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
			items: make(map[int]*semanticStreamItem),
		}
		frames, events, diagnostics, streamErr := stream.Push(resp.Body)
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
		translated, translateErr := engine.TranslateResponse(
			llmprotocol.OpenAIChatV1,
			target,
			resp.Body,
			func(response *llmprotocol.Response) error {
				if response.Model != resp.Model {
					response.Model = resp.Model
				}
				return nil
			},
		)
		if translateErr != nil {
			return nil, nil, nil, translateErr
		}
		semantic = &translated.Response
		body = translated.Body
		reqCtx.ResponseEnvelope = translated.Envelope
		reqCtx.ProtocolDiagnostics = append(reqCtx.ProtocolDiagnostics, translated.Diagnostics...)
	}
	reqCtx.SemanticResponse = semantic
	reqCtx.ImmediateResponseEncoded = true
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{Code: typev3.StatusCode_OK},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: buildLooperResponseHeaders(resp, reqCtx, contentType),
				},
				Body: body,
			},
		},
	}, semantic, body, nil
}

func buildLooperResponseHeaders(
	resp *looper.Response,
	reqCtx *RequestContext,
	contentType ...string,
) []*core.HeaderValueOption {
	// content-type is a real response header for the immediate body and always
	// rides; the v0.4 keystone headers (#2203) and the final routing facts ride
	// on the default surface.
	wireContentType := "application/json"
	if len(contentType) > 0 && contentType[0] != "" {
		wireContentType = contentType[0]
	}
	setHeaders := []*core.HeaderValueOption{
		newHeaderValueOption("content-type", wireContentType),
	}
	setHeaders = append(setHeaders, httputil.KeystoneHeaderOptions(headers.ResponsePathLooper)...)
	appendLooperRoutingFacts(&setHeaders, resp, reqCtx)
	// The looper execution trace, intermediate decision details and matched
	// signals are demoted to the x-vsr-debug surface (#2205).
	if debugHeadersRequested(reqCtx) {
		appendLooperTraceHeaders(&setHeaders, resp)
		appendLooperDecisionDetailHeaders(&setHeaders, reqCtx)
		appendLooperSignalHeaders(&setHeaders, reqCtx)
	}
	return setHeaders
}

// appendLooperTraceHeaders adds the looper execution trace (selected model,
// models used, iteration count, algorithm, aggregate latency and token
// usage). Demoted to the x-vsr-debug surface (#2205); the trace stays
// recoverable from the replay record.
func appendLooperTraceHeaders(setHeaders *[]*core.HeaderValueOption, resp *looper.Response) {
	if resp == nil {
		return
	}
	*setHeaders = append(*setHeaders,
		newHeaderValueOption(headers.VSRLooperModel, resp.Model),
		newHeaderValueOption(headers.VSRLooperModelsUsed, strings.Join(resp.ModelsUsed, ",")),
		newHeaderValueOption(headers.VSRLooperIterations, fmt.Sprintf("%d", resp.Iterations)),
		newHeaderValueOption(headers.VSRLooperAlgorithm, resp.AlgorithmType),
	)
	// A zero value here means "not measured" (e.g. a caller that bypasses
	// looper.ExecuteWithLatency or an algorithm that doesn't aggregate usage)
	// rather than a genuine zero-cost execution, so omit rather than emit a
	// misleading "0".
	appendPositiveIntHeader(setHeaders, headers.VSRLooperLatencyMs, resp.LatencyMs)
	appendPositiveIntHeader(setHeaders, headers.VSRLooperPromptTokens, resp.Usage.PromptTokens)
	appendPositiveIntHeader(setHeaders, headers.VSRLooperCompletionTokens, resp.Usage.CompletionTokens)
	appendPositiveIntHeader(setHeaders, headers.VSRLooperTotalTokens, resp.Usage.TotalTokens)
}

func appendPositiveIntHeader(setHeaders *[]*core.HeaderValueOption, key string, value int64) {
	if value <= 0 {
		return
	}
	*setHeaders = append(*setHeaders, newHeaderValueOption(key, fmt.Sprintf("%d", value)))
}

func appendLooperSignalHeaders(
	setHeaders *[]*core.HeaderValueOption,
	reqCtx *RequestContext,
) {
	if reqCtx == nil {
		return
	}
	appendJoinedHeader(setHeaders, headers.VSRMatchedKeywords, reqCtx.VSRMatchedKeywords)
	appendJoinedHeader(setHeaders, headers.VSRMatchedEmbeddings, reqCtx.VSRMatchedEmbeddings)
	appendJoinedHeader(setHeaders, headers.VSRMatchedDomains, reqCtx.VSRMatchedDomains)
	appendJoinedHeader(setHeaders, headers.VSRMatchedFactCheck, reqCtx.VSRMatchedFactCheck)
	appendJoinedHeader(setHeaders, headers.VSRMatchedUserFeedback, reqCtx.VSRMatchedUserFeedback)
	appendJoinedHeader(setHeaders, headers.VSRMatchedReask, reqCtx.VSRMatchedReask)
	appendJoinedHeader(setHeaders, headers.VSRMatchedPreference, reqCtx.VSRMatchedPreference)
	appendJoinedHeader(setHeaders, headers.VSRMatchedLanguage, reqCtx.VSRMatchedLanguage)
	appendJoinedHeader(setHeaders, headers.VSRMatchedContext, reqCtx.VSRMatchedContext)
	appendJoinedHeader(setHeaders, headers.VSRMatchedStructure, reqCtx.VSRMatchedStructure)
	appendJoinedHeader(setHeaders, headers.VSRMatchedComplexity, reqCtx.VSRMatchedComplexity)
	appendJoinedHeader(setHeaders, headers.VSRMatchedModality, reqCtx.VSRMatchedModality)
	appendJoinedHeader(setHeaders, headers.VSRMatchedAuthz, reqCtx.VSRMatchedAuthz)
	appendJoinedHeader(setHeaders, headers.VSRMatchedJailbreak, reqCtx.VSRMatchedJailbreak)
	appendJoinedHeader(setHeaders, headers.VSRMatchedPII, reqCtx.VSRMatchedPII)
	appendJoinedHeader(setHeaders, headers.VSRMatchedKB, reqCtx.VSRMatchedKB)
	appendJoinedHeader(setHeaders, headers.VSRMatchedConversation, reqCtx.VSRMatchedConversation)
	appendJoinedHeader(setHeaders, headers.VSRMatchedEvent, reqCtx.VSRMatchedEvent)
	appendJoinedHeader(setHeaders, headers.VSRMatchedProjection, reqCtx.VSRMatchedProjection)

	if reqCtx.VSRContextTokenCount > 0 {
		*setHeaders = append(
			*setHeaders,
			newHeaderValueOption(
				headers.VSRContextTokenCount,
				fmt.Sprintf("%d", reqCtx.VSRContextTokenCount),
			),
		)
	}
}

// appendLooperRoutingFacts adds the final routing facts that ride on the default
// surface: the selected model, decision, confidence, and the replay-id entry
// point. The looper resolves the model itself, so it falls back to resp.Model
// when the context did not record an override.
func appendLooperRoutingFacts(
	setHeaders *[]*core.HeaderValueOption,
	resp *looper.Response,
	reqCtx *RequestContext,
) {
	selectedModel := ""
	if resp != nil {
		selectedModel = resp.Model
	}
	if reqCtx == nil {
		appendOptionalHeader(setHeaders, headers.VSRSelectedModel, selectedModel)
		return
	}
	if reqCtx.VSRSelectedModel != "" {
		selectedModel = reqCtx.VSRSelectedModel
	}
	appendOptionalHeader(setHeaders, headers.VSRSelectedModel, selectedModel)
	appendOptionalHeader(setHeaders, headers.VSRSelectedRecipe, string(reqCtx.Routing.RecipeName()))
	appendOptionalHeader(setHeaders, headers.VSRSelectedDecision, reqCtx.VSRSelectedDecisionName)
	if reqCtx.VSRSelectedDecisionName != "" && reqCtx.VSRSelectedDecisionConfidence >= 0 {
		appendOptionalHeader(
			setHeaders,
			headers.VSRSelectedConfidence,
			fmt.Sprintf("%.4f", reqCtx.VSRSelectedDecisionConfidence),
		)
	}
	appendOptionalHeader(setHeaders, headers.RouterReplayID, reqCtx.RouterReplayID)
}

// appendLooperDecisionDetailHeaders adds the intermediate decision details
// (selected category, session phase). Demoted to the x-vsr-debug surface
// (#2205); both remain recoverable from the replay record.
func appendLooperDecisionDetailHeaders(
	setHeaders *[]*core.HeaderValueOption,
	reqCtx *RequestContext,
) {
	if reqCtx == nil {
		return
	}
	appendOptionalHeader(setHeaders, headers.VSRSelectedCategory, reqCtx.VSRSelectedCategory)
	appendOptionalHeader(setHeaders, headers.VSRSessionPhase, sessionPolicyPhase(reqCtx))
}

func appendJoinedHeader(
	setHeaders *[]*core.HeaderValueOption,
	key string,
	values []string,
) {
	if len(values) == 0 {
		return
	}
	*setHeaders = append(*setHeaders, newHeaderValueOption(key, strings.Join(values, ",")))
}

func appendOptionalHeader(
	setHeaders *[]*core.HeaderValueOption,
	key string,
	value string,
) {
	if value == "" {
		return
	}
	*setHeaders = append(*setHeaders, newHeaderValueOption(key, value))
}

func newHeaderValueOption(key string, value string) *core.HeaderValueOption {
	return &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	}
}
