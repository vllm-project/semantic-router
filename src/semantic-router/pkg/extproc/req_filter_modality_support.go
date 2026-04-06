package extproc

import (
	"fmt"
	"io"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/imagegen"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func resolveModalityDecisionState(ctx *RequestContext, cfg *config.RouterConfig) (ModalityClassificationResult, *config.Decision, ModalityModels, bool) {
	if cfg == nil || !cfg.ModalityDetector.Enabled {
		return ModalityClassificationResult{}, nil, ModalityModels{}, false
	}
	if ctx == nil || ctx.ModalityClassification == nil || ctx.ModalityClassification.Modality == "" {
		return ModalityClassificationResult{}, nil, ModalityModels{}, false
	}

	result := *ctx.ModalityClassification
	decision := ctx.VSRSelectedDecision
	models := ModalityModels{}
	if decision != nil {
		models = resolveAllModalityModels(decision, cfg.ModelConfig)
	}
	return result, decision, models, true
}

func (r *OpenAIRouter) handleDiffusionModality(ctx *RequestContext, cfg *config.RouterConfig, openAIRequest *openai.ChatCompletionNewParams, result ModalityClassificationResult, decision *config.Decision, models ModalityModels) (*ext_proc.ProcessingResponse, error) {
	if decision == nil {
		return nil, fmt.Errorf("modality DIFFUSION matched but no decision selected")
	}
	if models.OmniModel != "" {
		logging.Infof("[ModalityRouter] DIFFUSION: using omni model %s for image generation", models.OmniModel)
		return r.executeOmni(ctx, cfg, openAIRequest, result, models.OmniModel)
	}
	if models.DiffusionModel == "" {
		return nil, fmt.Errorf("decision %q has no diffusion or omni model in modelRefs", decision.Name)
	}
	return r.executeDiffusion(ctx, cfg, result, models.DiffusionModel)
}

func (r *OpenAIRouter) handleBothModality(ctx *RequestContext, cfg *config.RouterConfig, openAIRequest *openai.ChatCompletionNewParams, result ModalityClassificationResult, decision *config.Decision, models ModalityModels) (*ext_proc.ProcessingResponse, error) {
	if decision == nil {
		return nil, fmt.Errorf("modality BOTH matched but no decision selected")
	}
	if models.OmniModel != "" {
		logging.Infof("[ModalityRouter] BOTH: using omni model %s (single call for text+image)", models.OmniModel)
		return r.executeOmni(ctx, cfg, openAIRequest, result, models.OmniModel)
	}
	if models.ARModel == "" {
		return nil, fmt.Errorf("decision %q has no AR or omni model in modelRefs", decision.Name)
	}
	if models.DiffusionModel == "" {
		return nil, fmt.Errorf("decision %q has no diffusion or omni model in modelRefs", decision.Name)
	}
	return r.executeBoth(ctx, cfg, openAIRequest, result, models.ARModel, models.DiffusionModel)
}

func buildImageGenerateRequest(ctx *RequestContext, pluginCfg *config.ImageGenPluginConfig, promptPrefixes []string) *imagegen.GenerateRequest {
	genReq := &imagegen.GenerateRequest{
		Prompt: ExtractImagePrompt(ctx.UserContent, promptPrefixes),
		Width:  pluginCfg.DefaultWidth,
		Height: pluginCfg.DefaultHeight,
	}
	applyResponseAPIImageOverrides(ctx, genReq)
	return genReq
}

func applyResponseAPIImageOverrides(ctx *RequestContext, genReq *imagegen.GenerateRequest) {
	if ctx == nil || ctx.ResponseAPICtx == nil || ctx.ResponseAPICtx.ImageGenToolParams == nil {
		return
	}

	params := ctx.ResponseAPICtx.ImageGenToolParams
	if params.Size != "" && params.Size != "auto" {
		w, h := parseSizeString(params.Size)
		if w > 0 && h > 0 {
			genReq.Width = w
			genReq.Height = h
		}
	}
	if params.Quality != "" && params.Quality != "auto" {
		genReq.Quality = params.Quality
	}
	if params.Model != "" {
		genReq.Model = params.Model
	}
}

func closeHTTPBodyWithLog(body io.Closer, label string) {
	if body == nil {
		return
	}
	if err := body.Close(); err != nil {
		logging.Warnf("[ModalityRouter] failed to close %s: %v", label, err)
	}
}
