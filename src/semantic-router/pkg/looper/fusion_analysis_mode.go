package looper

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type fusionJudgeOutcome struct {
	analysis         *FusionAnalysis
	analysisResponse *ModelResponse
	finalResponse    *ModelResponse
	iterations       int
}

func (l *FusionLooper) runFusionJudgeStages(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	groundingScores []groundingScore,
) (fusionJudgeOutcome, error) {
	switch cfg.AnalysisMode {
	case config.FusionAnalysisModeSeparate:
		analysis, analysisResp := l.runFusionAnalysis(ctx, req, cfg, panelResponses, groundingScores)
		finalResp, err := l.runFusionFinal(ctx, req, cfg, panelResponses, analysis, groundingScores)
		return fusionJudgeOutcome{
			analysis:         analysis,
			analysisResponse: analysisResp,
			finalResponse:    finalResp,
			iterations:       2,
		}, err
	case config.FusionAnalysisModeOneCall, config.FusionAnalysisModeNone:
		finalResp, err := l.runFusionSingleJudge(ctx, req, cfg, panelResponses, groundingScores)
		return fusionJudgeOutcome{finalResponse: finalResp, iterations: 1}, err
	default:
		return fusionJudgeOutcome{}, fmt.Errorf("unsupported fusion analysis_mode %q", cfg.AnalysisMode)
	}
}

func (l *FusionLooper) runFusionSingleJudge(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	groundingScores []groundingScore,
) (*ModelResponse, error) {
	original := extractOriginalContent(req.OriginalRequest)
	outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
	prompt := buildFusionModeFinalPrompt(cfg, original, outputContract, panelResponses)
	if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	finalReq := appendFusionStageMessage(req.OriginalRequest, prompt)
	resp, err := l.callFusionModel(
		ctx,
		req,
		finalReq,
		cfg,
		cfg.Model,
		true,
		false,
		len(panelResponses)+1,
		config.FusionModelOverride{},
	)
	if err != nil {
		return nil, fmt.Errorf("fusion final synthesis failed for judge model %q: %w", cfg.Model, err)
	}
	applyJSONActionOutputContract(req.OutputContractSpec, resp, panelResponses)
	applyFinalOutputContract(req.OutputContractSpec, resp)
	return resp, nil
}

func buildFusionModeFinalPrompt(
	cfg fusionExecutionConfig,
	original string,
	outputContract string,
	responses []*ModelResponse,
) string {
	if cfg.SynthesisTemplate != "" {
		return appendOutputContractForPrompt(
			renderFusionPrompt(cfg.SynthesisTemplate, original, responses, nil),
			outputContract,
		)
	}

	var instruction string
	switch cfg.AnalysisMode {
	case config.FusionAnalysisModeOneCall:
		instruction = `You are the Fusion calling model. Compare the panel responses, resolve contradictions, identify the strongest supported information, and produce the final answer in this single call.`
	case config.FusionAnalysisModeNone:
		instruction = `You are the Fusion calling model. Synthesize a final answer directly from the panel responses without requesting or returning a distinct analysis artifact.`
	}

	prompt := fmt.Sprintf(`%s

Rules:
- Preserve the original output contract exactly.
- Do not reveal hidden reasoning, scratch work, panel reasoning, tool traces, or internal deliberation.
- Do not mention internal model names unless the user asks.
- Provide a concise explanation only when the original output contract asks for one.

Original prompt:
%s

Panel responses:
%s

Final answer:`, instruction, original, formatPanelResponses(responses))

	return appendOutputContractForPrompt(prompt, outputContract)
}
