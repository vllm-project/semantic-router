package config

import (
	"encoding/hex"
	"fmt"
	"math"
	"strings"
	"time"
)

func (b *AlgorithmBudget) Validate() error {
	if b == nil {
		return nil
	}
	if _, err := positiveNativeDuration(b.Deadline); err != nil {
		return fmt.Errorf("algorithm.budget.deadline: %w", err)
	}
	if b.MaxCalls <= 0 {
		return fmt.Errorf("algorithm.budget.max_calls must be positive")
	}
	return nil
}

func positiveNativeDuration(value string) (time.Duration, error) {
	duration, err := time.ParseDuration(value)
	if err != nil || duration <= 0 {
		return 0, fmt.Errorf("must be a positive duration, got %q", value)
	}
	return duration, nil
}

func validateNativeArtifact(source, digest string) error {
	if strings.TrimSpace(source) == "" || source != strings.TrimSpace(source) || strings.Contains(source, "://") || strings.ContainsRune(source, '\x00') {
		return fmt.Errorf("source must name a local artifact file")
	}
	decoded, err := hex.DecodeString(digest)
	if err != nil || len(decoded) != 32 {
		return fmt.Errorf("sha256 must be a 64-character SHA-256 digest")
	}
	return nil
}

// validateNativeAlgorithmConfig checks a System One escalation: its budget,
// its stages over the decision's modelRefs, and its quality policy.
func validateNativeAlgorithmConfig(name string, refs []ModelRef, algorithm *AlgorithmConfig) error {
	if algorithm.Budget == nil {
		return fmt.Errorf("decision %q: a System One escalation requires an explicit algorithm.budget", name)
	}
	if algorithm.MinimumCandidates != 0 {
		return fmt.Errorf("decision %q: a System One escalation uses authored stages; minimum_candidates is unsupported", name)
	}
	if err := algorithm.Budget.Validate(); err != nil {
		return fmt.Errorf("decision %q: %w", name, err)
	}
	if len(refs) == 0 {
		return fmt.Errorf("decision %q: a System One escalation requires modelRefs", name)
	}
	if algorithm.OnError != "" && algorithm.OnError != EscalateOnErrorFail {
		return fmt.Errorf("decision %q: a System One escalation returns unresolved when no stage accepts; algorithm.on_error must be %q", name, EscalateOnErrorFail)
	}
	if err := validateNativeQuality(algorithm.Escalate.Quality); err != nil {
		return fmt.Errorf("decision %q algorithm.escalate.quality: %w", name, err)
	}
	return validateNativeStages(name, refs, algorithm.Escalate.Stages)
}

func validateNativeStages(name string, refs []ModelRef, stages []EscalateStage) error {
	candidates := make(map[string]bool, len(refs))
	for _, ref := range refs {
		if ref.Model == "" || candidates[ref.Model] || ref.LoRAName != "" {
			return fmt.Errorf("decision %q: native modelRefs must contain distinct concrete model aliases without LoRA overrides", name)
		}
		candidates[ref.Model] = true
	}
	names := make(map[string]bool, len(stages))
	for index, stage := range stages {
		path := fmt.Sprintf("decision %q algorithm.escalate.stages[%d]", name, index)
		stageName := stage.StageName()
		if stageName == "" || stageName == "abstain" || stageName != strings.TrimSpace(stageName) || names[stageName] {
			return fmt.Errorf("%s: stage names (the model unless set) must be distinct and non-empty without surrounding whitespace", path)
		}
		names[stageName] = true
		if index == 0 && (stage.StageKind() != EscalateStageNative || !stage.IsEnabled()) {
			return fmt.Errorf("%s: first stage must be enabled and native", path)
		}
		if err := validateNativeStage(stage, candidates); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

func validateNativeStage(stage EscalateStage, candidates map[string]bool) error {
	if !candidates[stage.Model] {
		return fmt.Errorf("model %q is an undeclared modelRef", stage.Model)
	}
	switch stage.StageKind() {
	case EscalateStageNative:
		if stage.Generation != nil || stage.Instructions != "" {
			return fmt.Errorf("native stage cannot declare LLM generation or instructions")
		}
	case EscalateStageJudge:
		if stage.Generation == nil || stage.Generation.MaxOutputTokens <= 0 {
			return fmt.Errorf("LLM stage requires positive generation.max_output_tokens")
		}
	default:
		return fmt.Errorf("kind must be native or judge; typed_fallback requires a separate response contract")
	}
	if stage.Timeout != "" {
		if _, err := positiveNativeDuration(stage.Timeout); err != nil {
			return fmt.Errorf("timeout: %w", err)
		}
	}
	if stage.Accept != nil {
		return validateNativeAcceptance(stage.Accept)
	}
	return nil
}

func validateNativeQuality(quality *EscalateQuality) error {
	switch quality.EffectiveType() {
	case EscalateQualityUncalibrated:
		if quality == nil {
			return nil
		}
		if quality.Calibration != "" || quality.Loss != "" || quality.MaxRisk != nil {
			return fmt.Errorf("uncalibrated acceptance cannot declare calibrated evidence or risk")
		}
		if quality.Accept == nil {
			return nil
		}
		return validateNativeAcceptance(quality.Accept)
	case EscalateQualityCalibrated:
		if quality.Accept != nil || strings.TrimSpace(quality.Calibration) == "" || quality.Loss != "bundle_error" || quality.MaxRisk == nil {
			return fmt.Errorf("calibrated quality requires calibration, loss: bundle_error, and max_risk, without accept")
		}
		risk := *quality.MaxRisk
		if math.IsNaN(risk) || math.IsInf(risk, 0) || risk < 0 || risk > 1 {
			return fmt.Errorf("max_risk must be finite and in [0, 1]")
		}
	default:
		return fmt.Errorf("type must be uncalibrated or calibrated")
	}
	return nil
}

func validateNativeAcceptance(accept *EscalateAccept) error {
	if accept.Confidence != nil || accept.SystemOne != nil || accept.Refusal != nil {
		return fmt.Errorf("accept: a System One answer is accepted by native rules; confidence, systemone and refusal accept Chat answers")
	}
	if len(accept.Native) == 0 {
		return fmt.Errorf("accept.native must contain at least one typed rule")
	}
	for index, rule := range accept.Native {
		if rule.Question == "" && rule.QuestionType == "" {
			return fmt.Errorf("accept.native[%d] must name a question or question_type", index)
		}
		switch rule.QuestionType {
		case "", "choice", "score", "noul", "box":
		default:
			return fmt.Errorf("accept.native[%d].question_type must be choice, score, noul, or box", index)
		}
		if rule.Field != "confidence" && rule.Field != "top_probability" && rule.Field != "probability_margin" {
			return fmt.Errorf("accept.native[%d].field must be confidence, top_probability, or probability_margin; score value is not confidence", index)
		}
		if rule.QuestionType == "box" && rule.Field != "confidence" {
			return fmt.Errorf("accept.native[%d]: a box answer has only a confidence", index)
		}
		if rule.Field == "probability_margin" && rule.QuestionType != "" && rule.QuestionType != "noul" {
			return fmt.Errorf("accept.native[%d]: probability_margin applies only to noul", index)
		}
		if err := validateNumericPredicateContract(&rule.NumericPredicate); err != nil {
			return fmt.Errorf("accept.native[%d]: %w", index, err)
		}
		for _, bound := range []*float64{rule.GT, rule.GTE, rule.LT, rule.LTE} {
			if bound != nil && (math.IsNaN(*bound) || math.IsInf(*bound, 0) || *bound < 0 || *bound > 1) {
				return fmt.Errorf("accept.native[%d]: probability bounds must be finite and in [0, 1]", index)
			}
		}
	}
	return nil
}
