/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package extproc

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// isLooperRequest checks if the incoming request is from looper (internal request)
// If so, extproc should skip plugin processing to avoid recursion
func (r *OpenAIRouter) isLooperRequest(ctx *RequestContext) bool {
	return ctx.LooperRequest
}

// shouldUseLooper checks if the decision requires looper execution
// Returns true if:
// - Decision has an Algorithm configured AND
// - Decision has at least one ModelRef (ReMoM supports single model) AND
// - Looper endpoint is configured in router config
func (r *OpenAIRouter) shouldUseLooper(decision *config.Decision) bool {
	if decision == nil || r.Config == nil {
		return false
	}
	if decision.Algorithm == nil {
		return false
	}
	if !config.IsExecutorAlgorithmType(decision.Algorithm.Type) || decision.Algorithm.IsNative() {
		return false
	}

	return hasLooperModelInputs(decision)
}

func hasLooperModelInputs(decision *config.Decision) bool {
	switch decision.Algorithm.Type {
	case config.DecisionAlgorithmReMoM:
		return len(decision.ModelRefs) >= 1
	case config.DecisionAlgorithmFusion:
		return len(decision.ModelRefs) >= 1 || hasFusionAnalysisModels(decision)
	case config.DecisionAlgorithmWorkflows:
		return len(decision.ModelRefs) >= 1
	default:
		return len(decision.ModelRefs) > 1
	}
}

func hasFusionAnalysisModels(decision *config.Decision) bool {
	return decision.Algorithm.Fusion != nil &&
		len(decision.Algorithm.Fusion.AnalysisModels) > 0
}

// looperConstructionFailed answers a decision whose Looper algorithm cannot
// be built.
func (r *OpenAIRouter) looperConstructionFailed(
	decision *config.Decision,
	reqCtx *RequestContext,
	err error,
) *ext_proc.ProcessingResponse {
	r.logLooperConstructionFailure(decision, reqCtx, err)
	return r.createErrorResponse(500, "Looper construction failed.")
}

func (r *OpenAIRouter) logLooperConstructionFailure(decision *config.Decision, reqCtx *RequestContext, err error) {
	logging.ComponentErrorEvent("extproc", "looper_construction_failed", map[string]interface{}{
		"request_id": reqCtx.RequestID,
		"decision":   decision.Name,
		"algorithm":  decision.Algorithm.Type,
		"error":      err.Error(),
	})
}

// looperModelClient is the client for the Router's own model calls outside a
// request graph, such as a context recovery's: hops served in process. It is
// nil when the router has nothing to send hops with.
func (r *OpenAIRouter) looperModelClient() *looper.Client {
	hops := r.looperHops()
	if hops == nil {
		return nil
	}
	return looper.NewHopClient(&r.Config.Looper, hops)
}

// handleLooperExecution executes the looper for multi-model decisions
func (r *OpenAIRouter) handleLooperExecution(
	ctx context.Context,
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	if r.WorkflowStateService != nil && decision != nil && decision.Algorithm != nil &&
		decision.Algorithm.Type == config.DecisionAlgorithmWorkflows {
		if !r.WorkflowStateService.Acquire() {
			return r.createErrorResponse(503, "Router is shutting down"), nil
		}
		defer r.WorkflowStateService.Release()
	}
	// Looper builds every model call from the request the decision's tools
	// policy and selection have already narrowed (see buildLooperRequest).
	// The trusted-facts gate runs inside Looper, per model attempt, at each
	// attempt's own stage.
	looperReq, errorResponse := r.buildLooperRequest(request, decision, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}

	ownership := looper.ResolveFinalStageOwnership(looperReq)
	if r.shouldStreamLooperFinalStage(ownership, reqCtx, request, decision) {
		return r.handleLooperStreamingDispatch(ctx, looperReq, decision, request.Model, reqCtx)
	}
	r.logLooperStreamingFallbackIfRequested(ownership, reqCtx, request, decision)

	resp, errorResponse := r.executeLooper(ctx, looperReq, decision, request.Model, reqCtx)
	if errorResponse != nil {
		return errorResponse, nil
	}

	return r.finalizeLooperBufferedResponse(resp, decision, request.Model, reqCtx)
}

func (r *OpenAIRouter) shouldStreamLooperFinalStage(
	ownership looper.StageOwnership,
	reqCtx *RequestContext,
	request *llmprotocol.Request,
	decision *config.Decision,
) bool {
	if r == nil || r.Config == nil || !r.Config.Looper.IsStreamingEnabled() {
		return false
	}
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.Type != config.DecisionAlgorithmFusion {
		return false
	}
	if reqCtx == nil || !reqCtx.ExpectStreamingResponse {
		return false
	}
	if request == nil || !request.Stream {
		return false
	}
	if reqCtx.SourceFormat != llmprotocol.OpenAIChatV1 {
		return false
	}
	return ownership.Eligibility == looper.StreamingEligible
}

func (r *OpenAIRouter) logLooperStreamingFallbackIfRequested(
	ownership looper.StageOwnership,
	reqCtx *RequestContext,
	request *llmprotocol.Request,
	decision *config.Decision,
) {
	if r == nil || r.Config == nil || !r.Config.Looper.IsStreamingEnabled() {
		return
	}
	if reqCtx == nil || !reqCtx.ExpectStreamingResponse || request == nil || !request.Stream {
		return
	}
	decisionName := ""
	if decision != nil {
		decisionName = decision.Name
	}
	logging.ComponentEvent("extproc", "looper_streaming_fallback_to_buffered", map[string]interface{}{
		"request_id":         reqCtx.RequestID,
		"decision":           decisionName,
		"algorithm":          ownership.AlgorithmType,
		"stage_name":         ownership.StageName,
		"stage_role":         string(ownership.StageRole),
		"target_model":       ownership.TargetModel,
		"final_user_visible": ownership.IsFinalUserVisible,
		"eligibility":        string(ownership.Eligibility),
		"source_format":      string(reqCtx.SourceFormat),
		"buffering_reason":   looperStreamingBufferingReason(ownership, reqCtx, decision),
	})
}

func looperStreamingBufferingReason(
	ownership looper.StageOwnership,
	reqCtx *RequestContext,
	decision *config.Decision,
) string {
	if ownership.BufferingReason != looper.BufferingReasonNone {
		return string(ownership.BufferingReason)
	}
	if decision == nil || decision.Algorithm == nil || decision.Algorithm.Type != config.DecisionAlgorithmFusion {
		return string(looper.BufferingReasonUnsupportedAlgorithm)
	}
	if reqCtx != nil && reqCtx.SourceFormat != llmprotocol.OpenAIChatV1 {
		return "unsupported_source_format"
	}
	return "streaming_ineligible"
}

func (r *OpenAIRouter) handleLooperStreamingDispatch(
	ctx context.Context,
	looperReq *looper.Request,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	runtimeConfig := r.Config.Looper
	for _, entrypoint := range r.Config.EffectiveEntrypoints(config.ChatAPI) {
		runtimeConfig.EntrypointModels = append(runtimeConfig.EntrypointModels, entrypoint.ModelNames...)
	}
	hops := r.looperHops()
	if hops == nil {
		return r.looperConstructionFailed(decision, reqCtx, errNoLooperHops), nil
	}
	client := looper.NewHopClient(&runtimeConfig, hops)
	looperInstance, err := looper.FactoryWithClientAndWorkflowState(&runtimeConfig, decision.Algorithm.Type, client, r.WorkflowStateService)
	if err != nil {
		return r.looperConstructionFailed(decision, reqCtx, err), nil
	}

	streamingLooper, ok := looperInstance.(looper.StreamingLooper)
	if !ok {
		return r.looperConstructionFailed(
			decision,
			reqCtx,
			fmt.Errorf("looper algorithm %q does not implement StreamingLooper", decision.Algorithm.Type),
		), nil
	}

	plan, err := streamingLooper.PrepareStreamingSynthesis(ctx, looperReq)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_streaming_prepare_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"decision":   decision.Name,
			"algorithm":  decision.Algorithm.Type,
			"error":      err.Error(),
		})
		evidence, _ := looperFailureEvidenceFromError(err)
		if evidence.algorithm == "" && decision != nil && decision.Algorithm != nil {
			evidence.algorithm = decision.Algorithm.Type
		}
		var quorumOutcome *looper.FusionQuorumOutcome
		if outcome, ok := looper.FusionQuorumOutcomeFromError(err); ok {
			quorumOutcome = outcome
			evidence.fusionQuorum = quorumOutcome
		}
		finalizeLooperQuorumOutcome(reqCtx, quorumOutcome, decision, false)

		statusCode := 502
		lifecycleState := routerreplay.LifecycleFailed
		reason := "looper_streaming_prepare_failed"

		if errors.Is(err, context.Canceled) || (quorumOutcome != nil && quorumOutcome.Disposition == looper.FusionQuorumCancelled) {
			statusCode = 499
			lifecycleState = routerreplay.LifecycleAborted
			reason = "client_cancelled_during_looper_prepare"
		} else if errors.Is(err, context.DeadlineExceeded) {
			statusCode = 504
			reason = "looper_streaming_deadline_exceeded"
		}

		return r.recordLooperFailureWithLifecycle(
			reqCtx, originalModel, decision, statusCode,
			"Looper intermediate execution failed", lifecycleState, reason,
			evidence,
		), nil
	}

	rawSynthesisBody, err := json.Marshal(plan.SynthesisParams)
	if err != nil {
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		return r.recordLooperFailure(
			reqCtx, originalModel, decision, 502,
			"Looper streaming synthesis params marshal failed", "looper_stream_marshal_failed",
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		), nil
	}

	targetModel := strings.TrimSpace(plan.TargetModel)
	if targetModel == "" && plan.SynthesisParams != nil {
		targetModel = strings.TrimSpace(plan.SynthesisParams.Model)
	}

	dispatchResponse, synthesisTextBytes, err := r.buildLooperStreamingDispatchResponse(
		rawSynthesisBody, targetModel, decision, reqCtx,
	)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_streaming_dispatch_encode_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"decision":   decision.Name,
			"target":     targetModel,
			"error":      err.Error(),
		})
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		statusCode := 502
		lifecycleState := routerreplay.LifecycleFailed
		reason := "looper_stream_dispatch_encode_failed"
		msg := "Looper streaming dispatch encoding failed"

		var budgetErr *selection.RequestBudgetError
		if errors.As(err, &budgetErr) || (reqCtx != nil && reqCtx.ImmediateProtocolError != nil) {
			statusCode = 400
			reason = "looper_stream_dispatch_bad_request"
			msg = "Looper streaming dispatch rejected: " + err.Error()
		} else if errors.Is(err, context.Canceled) {
			statusCode = 499
			lifecycleState = routerreplay.LifecycleAborted
			reason = "client_cancelled_during_looper_dispatch"
			msg = "Client cancelled during looper dispatch"
		}

		return r.recordLooperFailureWithLifecycle(
			reqCtx, originalModel, decision, statusCode,
			msg, lifecycleState, reason,
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		), nil
	}
	if dispatchResponse.GetImmediateResponse() != nil {
		finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, false)
		r.recordLooperImmediateDispatchFailure(
			reqCtx, originalModel, decision, dispatchResponse,
			looperFailureEvidence{
				algorithm:    decision.Algorithm.Type,
				modelsUsed:   plan.ModelsUsed,
				iterations:   plan.Iterations,
				usage:        plan.IntermediateUsage,
				fusionQuorum: plan.QuorumOutcome,
			},
		)
		return dispatchResponse, nil
	}

	dispatchedModelsUsed := appendUniqueModel(plan.ModelsUsed, targetModel)
	dispatchedIterations := plan.Iterations + 1

	// Update RequestContext state for subsequent response stream handling and audit
	reqCtx.LooperStreamingActive = true
	reqCtx.LooperTargetModel = targetModel
	reqCtx.LooperSynthesisTextBytes = synthesisTextBytes
	reqCtx.LooperIntermediateUsage = plan.IntermediateUsage
	reqCtx.LooperExecutionTrace = plan.Trace
	reqCtx.LooperTrace = &looperTrace{
		algorithm:  decision.Algorithm.Type,
		model:      targetModel,
		modelsUsed: dispatchedModelsUsed,
		iterations: dispatchedIterations,
		usage:      plan.IntermediateUsage,
	}
	reqCtx.LooperModelsUsed = dispatchedModelsUsed
	reqCtx.LooperIterations = dispatchedIterations
	reqCtx.ResponsePath = headers.ResponsePathLooper
	reqCtx.VSRSelectedModel = targetModel
	reqCtx.RequestModel = originalModel // Preserve client's original requested model for rewrite & accounting
	reqCtx.VSRSelectionMethod = decision.Algorithm.Type

	finalizeLooperQuorumOutcome(reqCtx, plan.QuorumOutcome, decision, true)

	// Start replay so downstream stream completion or cancellation can record usage and lifecycle
	r.startRouterReplay(reqCtx, originalModel, targetModel, decision.Name)

	mutatedBody := dispatchResponse.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	logging.ComponentDebugEvent("extproc", "looper_streaming_dispatch_prepared", map[string]interface{}{
		"request_id":     reqCtx.RequestID,
		"original_model": originalModel,
		"target_model":   targetModel,
		"decision":       decision.Name,
		"body_bytes":     len(mutatedBody),
	})

	return dispatchResponse, nil
}

func appendUniqueModel(models []string, target string) []string {
	out := slices.Clone(models)
	if target == "" || slices.Contains(out, target) {
		return out
	}
	return append(out, target)
}

func ensureStreamingRequestJSON(raw []byte) []byte {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return raw
	}
	fields["stream"] = json.RawMessage("true")
	patched, err := json.Marshal(fields)
	if err != nil {
		return raw
	}
	return patched
}

func (r *OpenAIRouter) buildLooperStreamingDispatchResponse(
	rawSynthesisBody []byte,
	targetModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, int, error) {
	if reqCtx != nil {
		clientSemanticReq := reqCtx.SemanticRequest
		clientEnvelope := reqCtx.ProtocolEnvelope
		defer func() {
			reqCtx.SemanticRequest = clientSemanticReq
			reqCtx.ProtocolEnvelope = clientEnvelope
		}()
	}

	engine, err := r.protocolEngine()
	if err != nil {
		return nil, 0, err
	}
	clientOptions := clientStreamOptions(reqCtx)
	// openai.ChatCompletionNewParams has no Stream struct field (streaming is
	// normally added by NewStreaming), yet preserves StreamOptions; mark the
	// marshaled synthesis body as stream=true so DecodeRequest validates it as
	// a streaming request rather than rejecting stream_options_without_stream.
	rawSynthesisBody = ensureStreamingRequestJSON(rawSynthesisBody)
	synthesisReq, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, rawSynthesisBody)
	if err != nil {
		return nil, 0, err
	}
	synthesisReq.Stream = true
	synthesisReq.StreamOptions = clientOptions
	synthesisReq.Generation++
	synthesisTextBytes := extractSemanticRequestSignals(&synthesisReq).ContextTextBytes

	// Retire the client's raw byte-replay envelope because the outgoing payload
	// is now the synthesized/fallback stage request.
	reqCtx.ProtocolEnvelope = llmprotocol.Envelope{}

	useReasoning, reasoningEffort := r.getReasoningInfoFromDecision(decision, targetModel)
	applyLooperReasoningContext(reqCtx, targetModel, useReasoning, reasoningEffort)

	backendModel := reqCtx.backendModelForCandidate(targetModel)
	_, _, hasConfiguredBackend, _ := r.Config.ResolvePrimaryBackendForModel(backendModel)
	if hasConfiguredBackend || r.Config.Looper.Endpoint == "" {
		dispatch, prepErr := r.prepareProviderDispatch(&synthesisReq, targetModel, "", useReasoning, reqCtx)
		if prepErr != nil {
			return nil, 0, prepErr
		}
		// Keep decision stage controls for final provider adaptation without
		// re-inserting the semantic system prompt onto the synthesis request.
		dispatch.decisionName = decision.Name
		reqCtx.SemanticRequest.Stream = true
		response := r.buildProviderDispatchResponse(dispatch, reqCtx)
		if response.GetImmediateResponse() != nil {
			return response, synthesisTextBytes, nil
		}
		finalized, finErr := r.finalizeProviderDispatchResponse(dispatch, response, reqCtx)
		return finalized, synthesisTextBytes, finErr
	}

	// Legacy Looper.Endpoint-only test fixtures have no VLLMEndpoints; still
	// route through encodeDispatchRequest and routeHeaderState so include_usage,
	// external model mapping, credentials, and headers remain uniform.
	targetFormat, fmtErr := wireFormatForModel(r.Config.GetModelAPIFormat(backendModel))
	if fmtErr != nil {
		targetFormat = llmprotocol.OpenAIChatV1
	}
	dispatch := &providerDispatch{
		logicalModel:  targetModel,
		backendModel:  backendModel,
		upstreamModel: r.Config.ResolveExternalModelID(targetModel, ""),
		targetFormat:  targetFormat,
		decisionName:  decision.Name,
		useReasoning:  useReasoning,
	}
	synthesisReq.Model = dispatch.upstreamModel
	synthesisReq.Stream = true
	synthesisReq.Generation++
	reqCtx.TargetFormat = dispatch.targetFormat
	reqCtx.SemanticRequest = &synthesisReq
	reqCtx.RequestModel = dispatch.logicalModel
	reqCtx.VSRSelectedModel = dispatch.logicalModel

	state := &routeHeaderState{
		setHeaders:    r.startUpstreamSpanAndInjectHeaders(dispatch, reqCtx),
		removeHeaders: []string{"content-length"},
	}
	if r.CredentialResolver != nil {
		if credErr := r.appendProviderCredential(state, dispatch.effectiveBackendModel(), "", reqCtx); credErr != nil {
			return credErr, synthesisTextBytes, nil
		}
	}
	appendRoutingHeaders(&state.setHeaders, dispatch.effectiveBackendModel())
	setProviderRequestPath(&state.setHeaders, nil, dispatch.targetFormat)
	r.applyDecisionHeaderMutations(state, reqCtx)

	if maskErr := r.applyMaskingBeforeDispatch(reqCtx); maskErr != nil {
		return nil, 0, maskErr
	}
	prepared, err := r.materializePreparedProviderRequest(dispatch, reqCtx)
	if err != nil {
		return nil, 0, err
	}
	appendContentLengthHeader(&state.setHeaders, prepared.byteLength())
	response := buildRequestBodyContinueResponse(state, nil, r.shouldClearRouteCache())
	bindPreparedDispatchArtifact(response.GetRequestBody().GetResponse(), reqCtx, prepared)
	return response, synthesisTextBytes, nil
}

func (r *OpenAIRouter) finalizeLooperBufferedResponse(
	resp *looper.Response,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*ext_proc.ProcessingResponse, error) {
	// The looper's final model must land on the request context before the
	// response headers are built. Runtime model selection has already put a
	// preliminary panel candidate in VSRSelectedModel, which the public headers
	// report, so building headers first would advertise a panel model as the
	// responder even when a fallback target produced the answer.
	reqCtx.RequestModel = resp.Model
	reqCtx.VSRSelectedModel = resp.Model
	reqCtx.VSRSelectionMethod = resp.AlgorithmType

	response, semanticResponse, clientBody, err := r.prepareLooperResponse(resp, reqCtx)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "looper_response_encode_failed", map[string]interface{}{
			"request_id": reqCtx.RequestID,
			"format":     reqCtx.SourceFormat,
			"error":      err.Error(),
		})
		// Encoding failed, so an error response is returned instead of the
		// recovered fallback.
		// Promote the terminal disposition first, then route through the shared
		// failure recorder. That helper is what actually creates the Replay
		// record: the usage, status, body, and quorum writers all no-op while
		// RouterReplayID is empty, so recording evidence without starting Replay
		// would persist nothing and return no Replay ID header.
		finalizeLooperQuorumOutcome(reqCtx, resp.QuorumOutcome, decision, false)
		return r.recordLooperFailure(
			reqCtx, originalModel, decision, 502,
			"Looper returned an invalid response", "looper_response_encode_failed",
			looperFailureEvidence{
				algorithm:    resp.AlgorithmType,
				modelsUsed:   resp.ModelsUsed,
				iterations:   resp.Iterations,
				usage:        resp.Usage,
				fusionQuorum: resp.QuorumOutcome,
			},
		), nil
	}
	// The response phase never sees a Looper answer, so the decision's
	// response policy reads it here. Nothing has left yet: even a stream is
	// rendered from the finished answer.
	plan := r.prepareResponsePolicy(reqCtx, semanticResponse, clientBody)
	// Replay keeps the reply the client receives, with its Looper fields and
	// in stream form for a stream, not the noted answer the policy encoded.
	reply := r.releaseLooperAnswer(resp, response, clientBody, plan, reqCtx)
	plan.finalBody = reply.GetImmediateResponse().GetBody()
	// Replay starts inside the recorder, so its ID only exists now; the reply
	// boundary adds it to the already-built response.
	r.recordSuccessfulLooperExecution(resp, originalModel, decision, reqCtx, plan)
	if plan.blocked == nil {
		addLooperCostHeaders(reply, reqCtx, resp)
	}
	return reply, nil
}

// addLooperCostHeaders prices a Looper answer as its record prices it: the
// sum of its calls, when every one has a price.
func addLooperCostHeaders(reply *ext_proc.ProcessingResponse, reqCtx *RequestContext, resp *looper.Response) {
	calls := looperRouterCalls(reqCtx.VSRLooperDiagnostics, resp.Usage, resp.Iterations)
	if calls.Cost == nil || calls.UnpricedCalls > 0 {
		return
	}
	appendImmediateResponseHeader(reply, headers.VSRCost, strconv.FormatFloat(*calls.Cost, 'f', -1, 64))
	appendImmediateResponseHeader(reply, headers.VSRCostCurrency, calls.Currency)
}

// releaseLooperAnswer is the reply the response policy leaves of a Looper
// answer: the policy's refusal, in the client's protocol and in stream form
// for a streaming client, or the answer with the guard's notes.
func (r *OpenAIRouter) releaseLooperAnswer(
	resp *looper.Response,
	response *ext_proc.ProcessingResponse,
	clientBody []byte,
	plan *responsePolicyPlan,
	reqCtx *RequestContext,
) *ext_proc.ProcessingResponse {
	streaming := reqCtx.ExpectStreamingResponse
	if plan.blocked != nil {
		// A refused answer is never kept, though a stream refuses it with a 200.
		if state := reqCtx.ResponseObjectState; state != nil {
			state.PersistenceAttempted = true
		}
		if streaming {
			response.GetImmediateResponse().Body = r.encodeStreamRefusal(reqCtx, plan.refusal)
			return response
		}
		reqCtx.ImmediateProtocolError, reqCtx.ImmediateResponseEncoded = plan.refusal, false
		return plan.blocked
	}
	noted := plan.finalBody
	if bytes.Equal(noted, clientBody) {
		return response
	}
	if streaming {
		// The stream encoder writes one choice; Ratings streams one per model.
		if len(reqCtx.SemanticResponse.Alternatives) > 0 {
			noted = prependChoiceStreamText(clientBody, reqCtx.ResponseGuard.notePrefix())
		} else {
			noted = r.encodeNotedStream(reqCtx)
		}
		if noted == nil {
			return response
		}
	}
	// The first render already reported each extension it dropped.
	diagnostics := reqCtx.ProtocolDiagnostics
	engine, err := r.protocolEngine()
	if err == nil {
		noted, err = finalizeLooperResponseExtensions(engine, noted, resp.RouterExtensions(), reqCtx, streaming)
	}
	reqCtx.ProtocolDiagnostics = diagnostics
	if err == nil {
		response.GetImmediateResponse().Body = noted
	}
	return response
}

func (r *OpenAIRouter) executeLooper(
	ctx context.Context,
	looperReq *looper.Request,
	decision *config.Decision,
	originalModel string,
	reqCtx *RequestContext,
) (*looper.Response, *ext_proc.ProcessingResponse) {
	runtimeConfig := r.Config.Looper
	for _, entrypoint := range r.Config.EffectiveEntrypoints(config.ChatAPI) {
		runtimeConfig.EntrypointModels = append(runtimeConfig.EntrypointModels, entrypoint.ModelNames...)
	}
	program, err := looper.Template(&runtimeConfig, decision.Algorithm.Type, decision.Algorithm.Budget, r.WorkflowStateService)
	if err != nil {
		return nil, r.looperConstructionFailed(decision, reqCtx, err)
	}
	hops := r.looperHops()
	if hops == nil {
		return nil, r.looperConstructionFailed(decision, reqCtx, errNoLooperHops)
	}
	return r.executeLooperGraph(ctx, program, hops, looperReq, originalModel, decision, reqCtx)
}

func (r *OpenAIRouter) buildLooperRequest(
	request *llmprotocol.Request,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*looper.Request, *ext_proc.ProcessingResponse) {
	modelRefs := decision.ModelRefs
	if reqCtx.VSREligibleModelRefs != nil {
		modelRefs = reqCtx.VSREligibleModelRefs
	}
	// Build looper request.
	// Responses uses buffered aggregation so its codec can enforce the existing
	// target capability and projection gates on the complete neutral result.
	// This controls internal execution only: the final transport gate renders
	// the client's requested JSON or SSE representation independently.
	streaming := reqCtx.ExpectStreamingResponse
	if isResponseAPIRequest(reqCtx) {
		streaming = false
	}
	logging.ComponentEvent("extproc", "looper_execution_started", map[string]interface{}{
		"request_id":       reqCtx.RequestID,
		"decision":         decision.Name,
		"algorithm":        decision.Algorithm.Type,
		"candidate_models": len(modelRefs),
		"streaming":        streaming,
		"response_api":     isResponseAPIRequest(reqCtx),
	})
	if _, err := r.applyDispatchRequestParams(request, reqCtx); err != nil {
		return nil, r.createErrorResponse(400, "Invalid request parameter policy")
	}
	if refused := r.applyLooperToolsPlugin(request, reqCtx); refused != nil {
		return nil, refused
	}
	if carriesReasoningContinuity(request, nil) {
		return nil, r.recordLooperFailure(
			reqCtx, request.Model, decision, 400,
			"Looper cannot represent the request semantics", "looper_reasoning_continuity_unsupported",
		)
	}
	engine, err := r.protocolEngine()
	if err == nil {
		var encoded protocolcodec.RequestResult
		encoded, err = engine.EncodeRequest(llmprotocol.OpenAIChatV1, *request, llmprotocol.Envelope{})
		if err == nil {
			// Chat drops a Responses blob the client replayed; the reply says so.
			reqCtx.ProtocolDiagnostics = append(reqCtx.ProtocolDiagnostics, encoded.Diagnostics...)
			var openAIRequest *openai.ChatCompletionNewParams
			openAIRequest, err = parseOpenAIRequest(encoded.Body)
			if err == nil {
				looperReq := &looper.Request{
					OriginalRequest:       openAIRequest,
					CandidateRequirements: r.candidateRequirements(reqCtx).Clone(),
					PermittedModels:       looperPermittedModels(modelRefs, decision.Algorithm),
					Grounding:             r.groundingForRecipe(reqCtx.Routing.RecipeName()),
					BaseContextTokens:     contextTokensWithoutReserves(reqCtx, reqCtx.VSRContextTokenCount),
					ModelRefs:             modelRefs,
					ModelParams:           r.getModelParams(),
					Algorithm:             decision.Algorithm,
					IsStreaming:           streaming,
					DecisionName:          decision.Name,
					RecipeName:            reqCtx.Routing.RecipeName(),
					OutputContract:        decision.OutputContract,
					OutputContractSpec:    decision.OutputContractSpec,
					ToolStageGate:         r.looperToolStageGate(request, decision, reqCtx),
					AnswerChecks:          r.escalateAnswerChecker(decision.Algorithm),
				}
				if params := decision.GetRequestParamsConfig(); params != nil && params.MaxTokensLimit != nil {
					limit := *params.MaxTokensLimit
					looperReq.MaxTokensLimit = &limit
				}
				return looperReq, nil
			}
		}
	}
	if err != nil {
		return nil, r.recordLooperFailure(
			reqCtx, request.Model, decision, 400,
			"Looper cannot represent the request semantics", "looper_request_unsupported",
		)
	}
	return nil, r.recordLooperFailure(
		reqCtx, request.Model, decision, 400,
		"Looper cannot represent the request semantics", "looper_request_unsupported",
	)
}

// carriesReasoningContinuity reports whether a request history or a reply holds
// typed reasoning continuity blocks. Internal Looper calls are built from
// openai-go parameters, which drop the blocks, so a tool turn would reach the
// provider without the state it needs; those paths refuse instead. Responses
// blobs are excluded because Chat request encoding deliberately drops them.
func carriesReasoningContinuity(request *llmprotocol.Request, response *llmprotocol.Response) bool {
	if request != nil && llmprotocol.RequiredCapabilities(*request).Supports(llmprotocol.CapabilityReasoningContinuity) {
		return true
	}
	return response != nil &&
		llmprotocol.RequiredResponseCapabilities(*response).Supports(llmprotocol.CapabilityReasoningContinuity)
}

func (r *OpenAIRouter) recordSuccessfulLooperExecution(
	resp *looper.Response,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
	plan *responsePolicyPlan,
) {
	// Protocol encoding succeeded and the immediate response is about to be
	// returned to Envoy, so promote a ready fallback to its terminal disposition
	// and emit the single quorum metric.
	finalizeLooperQuorumOutcome(reqCtx, resp.QuorumOutcome, decision, true)

	// Capture router replay information if enabled. Detailed attempts remain in
	// Replay; the public response surface keeps only aggregate Looper headers.
	reqCtx.VSRLooperDiagnostics = looperReplayDiagnostics(resp.ExecutionTrace)
	r.startRouterReplay(reqCtx, originalModel, resp.Model, decision.Name)
	r.updateLooperReplayUsage(reqCtx, resp.Usage, resp.Iterations)
	// Looper usage is an aggregate across potentially differently priced
	// calls. Preserve continuity, cache buckets, and completeness without
	// charging the whole cascade at the final response model's price.
	recordSessionTurn(reqCtx, looperResponseUsageMetrics(resp.Usage), sessiontelemetry.TurnPricing{})

	// Update router replay with success status (looper returns immediate response with 200)
	r.updateRouterReplayStatus(reqCtx, 200, false)
	// The hops have settled: the answer reports the budgets after them.
	r.finishRequestSpend(reqCtx)

	// The response body pipeline is bypassed by ImmediateResponse, so the
	// response policy commits here, once Replay is open: Replay, memory and
	// the Responses object keep only an answer the guard released.
	plan.commit()
	if plan != nil && plan.blocked != nil && reqCtx.ExpectStreamingResponse {
		// A stream carries the refusal under its 200, which no later phase
		// ends, so the record ends with it as a held stream's refusal does.
		r.failStreamedRouterReplay(reqCtx, plan.finalBody, "request_rejected")
	}
}

func (r *OpenAIRouter) updateLooperReplayUsage(ctx *RequestContext, usage looper.TokenUsage, calls int) {
	if usage.Unreported {
		return
	}
	promptTokens := int(usage.PromptTokens)
	completionTokens := int(usage.CompletionTokens)
	totalTokens := int(usage.TotalTokens)
	if promptTokens == 0 && completionTokens == 0 && totalTokens == 0 &&
		usage.CachedInputTokens == 0 && usage.CacheWriteTokens == 0 {
		return
	}
	if !usage.Complete() {
		return
	}

	// A looper result can aggregate differently priced model calls. Persist its
	// exact reported token counts, including cache buckets, but leave the model
	// cost unset rather than attributing the whole cascade to the selected
	// model's price: each attempt's own price is the request's router-call
	// cost. Unreported and otherwise incomplete aggregates are omitted so
	// partial usage cannot look complete in replay.
	r.updateRouterReplayUsageCost(ctx, routerreplay.UsageCost{
		PromptTokens:       replayIntPtr(promptTokens),
		CachedPromptTokens: replayIntPtr(int(usage.CachedInputTokens)),
		CacheWriteTokens:   replayIntPtr(int(usage.CacheWriteTokens)),
		CompletionTokens:   replayIntPtr(completionTokens),
		TotalTokens:        replayIntPtr(totalTokens),
		CostAccounting: &store.CostAccounting{
			RouterCalls: []store.RouterCallCost{looperRouterCalls(ctx.VSRLooperDiagnostics, usage, calls)},
		},
	})
}

func looperResponseUsageMetrics(usage looper.TokenUsage) responseUsageMetrics {
	if usage.Unreported {
		return responseUsageMetrics{invalid: true, invalidReason: "looper_usage_unreported"}
	}
	// The aggregate cannot tell a reported zero from an omitted cache field, so
	// only a positive count is backend-reported; otherwise the router estimates.
	metrics := responseUsageMetrics{
		promptTokens:               int(usage.PromptTokens),
		promptTokensReported:       true,
		cachedPromptTokens:         int(usage.CachedInputTokens),
		cachedPromptTokensReported: usage.CachedInputTokens > 0,
		cacheWriteTokens:           int(usage.CacheWriteTokens),
		cacheWriteTokensReported:   usage.CacheWriteTokens > 0,
		completionTokens:           int(usage.CompletionTokens),
		completionTokensReported:   true,
		totalTokens:                int(usage.TotalTokens),
		totalTokensReported:        true,
	}
	if usage.TotalTokens == 0 && usage.PromptTokens == 0 && usage.CompletionTokens == 0 &&
		usage.CachedInputTokens == 0 && usage.CacheWriteTokens == 0 {
		return metrics
	}
	if !usage.Complete() {
		metrics.invalid = true
		metrics.invalidReason = "looper_usage_incomplete"
	}
	return metrics
}

func (r *OpenAIRouter) groundingForRecipe(recipe config.RecipeName) *looper.GroundingBackends {
	if r.RecipeClassifiers != nil {
		classifier, ok := r.RecipeClassifiers.ForRecipe(recipe)
		if !ok {
			return nil
		}
		return classifier.GroundingBackends()
	}
	if recipe == "" || r.Config.IsDefaultRecipe(recipe) {
		return r.Classifier.GroundingBackends()
	}
	return nil
}

func looperPermittedModels(refs []config.ModelRef, algorithm *config.AlgorithmConfig) []string {
	var models []string
	for _, ref := range refs {
		models = append(models, ref.Model)
		if ref.LoRAName != "" {
			models = append(models, ref.LoRAName)
		}
	}
	return append(models, explicitAlgorithmModels(algorithm)...)
}
