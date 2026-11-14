package extproc

import (
	"errors"
	"fmt"
	"slices"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// imagesGenerationsPath is the Images API entry: its requests are routed on
// their prompt and answered with the image an image decision's model generates.
const imagesGenerationsPath = "/v1/images/generations"

// applyDecisionOutput adds the operation the selected decision's output
// contract produces to the request before its candidates are qualified, so a
// model that cannot carry it is never selected. A request that forbids the
// operation, or whose API cannot return its output, is refused before any
// model is called, and so is an Images API request whose decision answers
// without an image.
func (r *OpenAIRouter) applyDecisionOutput(ctx *RequestContext) error {
	if ctx == nil || ctx.VSRSelectedDecision == nil || ctx.SemanticRequest == nil {
		return nil
	}
	decision, request := ctx.VSRSelectedDecision, ctx.SemanticRequest
	if ctx.SourceFormat == llmprotocol.OpenAIImagesV1 {
		return r.applyImagesEntryDecision(ctx, decision, request)
	}
	if ctx.SourceFormat == llmprotocol.OpenAIVideosV1 {
		return r.applyVideosEntryDecision(ctx, decision, request)
	}
	if decision.Output.GeneratesVideos() {
		return r.refuseUncarriableVideoDecision(ctx, decision)
	}
	if !decision.Output.GeneratesImages() {
		return nil
	}
	if request.ImageGeneration == nil && request.ToolChoice.Mode == llmprotocol.ToolChoiceNone {
		return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorInvalidRequest, "image_generation_forbidden",
			"the matched route answers with a generated image, and the request's tool_choice none forbids the image_generation operation")
	}
	client := ctx.SourceFormat
	if client == "" {
		client = llmprotocol.OpenAIChatV1
	}
	if supported, ok := r.codecCapabilitiesForFormat(client); !ok || !supported.Supports(llmprotocol.CapabilityImageGeneration) {
		return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
			fmt.Sprintf("the matched route answers with a generated image, which the %s API cannot return; call the Responses API or the Images API", clientAPIName(client)))
	}
	if selection.ApplyDecisionOutput(request, decision) {
		request.Generation++
	}
	return nil
}

// refuseUncarriableVideoDecision refuses a video decision reached by a client
// whose API cannot return a video job, so the answer never becomes a 502 at the
// response stage.
func (r *OpenAIRouter) refuseUncarriableVideoDecision(ctx *RequestContext, decision *config.Decision) error {
	client := ctx.SourceFormat
	if client == "" {
		client = llmprotocol.OpenAIChatV1
	}
	if supported, ok := r.codecCapabilitiesForFormat(client); ok && supported.Supports(llmprotocol.CapabilityVideoGeneration) {
		return nil
	}
	return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
		fmt.Sprintf("the matched route answers with a video job, which the %s API cannot return; call the Videos API", clientAPIName(client)))
}

// applyVideosEntryDecision admits a Videos API create only to a decision that
// produces a video job. The job is the only answer the create call returns, so
// the video_generation operation is forced rather than inferred by a model.
func (r *OpenAIRouter) applyVideosEntryDecision(ctx *RequestContext, decision *config.Decision, request *llmprotocol.Request) error {
	if !decision.Output.GeneratesVideos() {
		return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
			"the Videos API needs a decision that answers with a generated video")
	}
	client := ctx.SourceFormat
	if client == "" {
		client = llmprotocol.OpenAIVideosV1
	}
	if supported, ok := r.codecCapabilitiesForFormat(client); !ok || !supported.Supports(llmprotocol.CapabilityVideoGeneration) {
		return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
			fmt.Sprintf("the Videos API cannot return a video job on the %s protocol", clientAPIName(client)))
	}
	if request.VideoGeneration == nil {
		request.VideoGeneration = &llmprotocol.VideoGenerationOptions{}
		request.Generation++
	}
	return nil
}

// applyImagesEntryDecision admits an Images API request only to a decision
// that generates images. The image is the only answer the Images API returns,
// so the request forces the image_generation operation on any model. One
// Responses call generates one image, so n above 1 needs one of the decision's
// models on a wire that takes n, which only the Images API does.
func (r *OpenAIRouter) applyImagesEntryDecision(ctx *RequestContext, decision *config.Decision, request *llmprotocol.Request) error {
	if !decision.Output.GeneratesImages() {
		failure := routingFailureImageRouteRequired
		return refuseDecisionOutput(ctx, decision, failure.category, failure.code, failure.message)
	}
	severalImages := llmprotocol.Capabilities(llmprotocol.CapabilityImageGeneration, llmprotocol.CapabilityMultipleCandidates)
	generatesSeveral := func(ref config.ModelRef) bool { return r.modelCanServeCapabilities(ref.Model, severalImages) }
	if request.CandidateCount != nil && *request.CandidateCount > 1 && len(decision.ModelRefs) > 0 && !slices.ContainsFunc(decision.ModelRefs, generatesSeveral) {
		// The decision's request_params can cap or drop n before any model sees it.
		effective, err := selection.EffectiveCandidateRequest(request, decision)
		if err != nil {
			return err
		}
		if n := effective.CandidateCount; n != nil && *n > 1 {
			return refuseDecisionOutput(ctx, decision, llmprotocol.ErrorUnsupportedFeature, "unsupported_capability",
				"n above 1 needs a model on the Images API; a Responses API model generates one image per call")
		}
	}
	if request.ToolChoice.Name == "" && (request.ToolChoice.Mode == "" || request.ToolChoice.Mode == llmprotocol.ToolChoiceAuto) {
		request.ToolChoice = llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration}
		request.Generation++
	}
	return nil
}

// imagesEntryRejection refuses an Images API request that reached no decision
// generating images: it matched no decision, or it names a model directly
// instead of a recipe's entrypoint.
func (r *OpenAIRouter) imagesEntryRejection(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.SourceFormat != llmprotocol.OpenAIImagesV1 {
		return nil
	}
	decision := ctx.VSRSelectedDecision
	if decision != nil && decision.Output.GeneratesImages() {
		return nil
	}
	reason := "the request matched no decision"
	switch {
	case decision != nil:
		reason = fmt.Sprintf("decision %q answers in %s", decision.Name, decision.Output)
	case ctx.Routing.SelectedRecipe() == nil:
		reason = fmt.Sprintf("model %q is not a recipe entrypoint", ctx.RequestModel)
	}
	logRoutingFailure(ctx, "decision_output_refused", routingFailureImageRouteRequired.code, errors.New(reason))
	return r.routingFailureResponse(ctx, routingFailureImageRouteRequired)
}

// videosEntryRejection refuses a Videos API create that reached no decision
// producing a video job.
func (r *OpenAIRouter) videosEntryRejection(ctx *RequestContext) *ext_proc.ProcessingResponse {
	if ctx == nil || ctx.SourceFormat != llmprotocol.OpenAIVideosV1 || !ctx.videosCreate {
		return nil
	}
	decision := ctx.VSRSelectedDecision
	if decision != nil && decision.Output.GeneratesVideos() {
		return nil
	}
	reason := "the request matched no decision"
	if decision != nil {
		reason = fmt.Sprintf("decision %q answers in %s", decision.Name, decision.Output)
	} else if ctx.Routing.SelectedRecipe() == nil {
		reason = fmt.Sprintf("model %q is not a recipe entrypoint", ctx.RequestModel)
	}
	logRoutingFailure(ctx, "decision_output_refused", "unsupported_capability", errors.New(reason))
	return r.routingFailureResponse(ctx, routingFailureVideoRouteRequired)
}

// refuseDecisionOutput logs which decision a request reached and returns the
// error its client sees, which names no decision.
func refuseDecisionOutput(ctx *RequestContext, decision *config.Decision, category llmprotocol.ErrorCategory, code, message string) error {
	logRoutingFailure(ctx, "decision_output_refused", code, fmt.Errorf("decision %q answers in %s", decision.Name, decision.Output))
	return llmprotocol.NewError(category, code, message, nil)
}

func clientAPIName(format llmprotocol.WireFormat) string {
	switch format {
	case llmprotocol.OpenAIChatV1:
		return "Chat Completions"
	case llmprotocol.AnthropicMessagesV1:
		return "Anthropic Messages"
	default:
		return string(format)
	}
}
