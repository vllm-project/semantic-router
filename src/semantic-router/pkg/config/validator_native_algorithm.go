package config

import (
	"encoding/hex"
	"fmt"
	"math"
	"strings"
	"time"
)

func (b *RoutingBudget) Validate() error {
	if b == nil {
		return nil
	}
	if _, err := positiveNativeDuration(b.Deadline); err != nil {
		return fmt.Errorf("routing.budget.deadline: %w", err)
	}
	if b.MaxCalls <= 0 {
		return fmt.Errorf("routing.budget.max_calls must be positive")
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

func validateNativeAlgorithmConfig(name string, refs []ModelRef, algorithm *AlgorithmConfig) error {
	if !algorithm.IsNative() {
		if algorithm.Quality != nil || len(algorithm.Stages) > 0 {
			return fmt.Errorf("decision %q: algorithm.quality and stages require cascade or policy", name)
		}
		return nil
	}
	if len(refs) == 0 || len(algorithm.Stages) == 0 {
		return fmt.Errorf("decision %q: native algorithms require modelRefs and stages", name)
	}
	if algorithm.OnError != "" {
		return fmt.Errorf("decision %q: native algorithms return unresolved on exhaustion; algorithm.on_error is unsupported", name)
	}
	if err := validateNativeQuality(algorithm.Quality); err != nil {
		return fmt.Errorf("decision %q algorithm.quality: %w", name, err)
	}
	if err := validateNativeStages(name, refs, algorithm.Stages); err != nil {
		return err
	}
	if algorithm.Type == DecisionAlgorithmPolicy {
		for index, stage := range algorithm.Stages {
			if stage.Kind == "judge" && index != len(algorithm.Stages)-1 {
				return fmt.Errorf("decision %q: policy permits one terminal judge after all native stages", name)
			}
		}
		if algorithm.Policy == nil {
			return fmt.Errorf("decision %q: policy requires algorithm.policy", name)
		}
		if err := validateNativeArtifact(algorithm.Policy.Source, algorithm.Policy.SHA256); err != nil {
			return fmt.Errorf("decision %q algorithm.policy: %w", name, err)
		}
		weight := algorithm.Policy.CostWeight
		if math.IsNaN(weight) || math.IsInf(weight, 0) || weight < 0 {
			return fmt.Errorf("decision %q algorithm.policy.cost_weight must be finite and nonnegative", name)
		}
	}
	return nil
}

func validateNativeStages(name string, refs []ModelRef, stages []CascadeStage) error {
	candidates := make(map[string]bool, len(refs))
	for _, ref := range refs {
		if ref.Model == "" || candidates[ref.Model] || ref.LoRAName != "" {
			return fmt.Errorf("decision %q: native modelRefs must contain distinct concrete model aliases without LoRA overrides", name)
		}
		candidates[ref.Model] = true
	}
	names := make(map[string]bool, len(stages))
	for index, stage := range stages {
		path := fmt.Sprintf("decision %q algorithm.stages[%d]", name, index)
		if stage.Name == "" || stage.Name == "abstain" || stage.Name != strings.TrimSpace(stage.Name) || names[stage.Name] {
			return fmt.Errorf("%s.name must be distinct and non-empty without surrounding whitespace", path)
		}
		names[stage.Name] = true
		if index == 0 && (stage.Kind != "native" || !stage.IsEnabled()) {
			return fmt.Errorf("%s: first stage must be enabled and native", path)
		}
		if err := validateNativeStage(stage, candidates); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

func validateNativeStage(stage CascadeStage, candidates map[string]bool) error {
	if !candidates[stage.Model] {
		return fmt.Errorf("model %q is an undeclared modelRef", stage.Model)
	}
	switch stage.Kind {
	case "native":
		if stage.Generation != nil || stage.Instructions != "" {
			return fmt.Errorf("native stage cannot declare LLM generation or instructions")
		}
	case "judge":
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

func validateNativeQuality(quality *NativeQualityConfig) error {
	if quality == nil {
		return fmt.Errorf("an explicit quality policy is required")
	}
	switch quality.Type {
	case "uncalibrated":
		if quality.Calibration != "" || quality.Loss != "" || quality.MaxRisk != nil {
			return fmt.Errorf("uncalibrated acceptance cannot declare calibrated evidence or risk")
		}
		return validateNativeAcceptance(quality.Acceptance)
	case "calibrated":
		if quality.Acceptance != nil || strings.TrimSpace(quality.Calibration) == "" || quality.Loss != "bundle_error" || quality.MaxRisk == nil {
			return fmt.Errorf("calibrated quality requires calibration, loss: bundle_error, and max_risk, without acceptance")
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

func validateNativeAcceptance(acceptance *NativeAcceptance) error {
	if acceptance == nil || len(acceptance.Rules) == 0 {
		return fmt.Errorf("acceptance must contain at least one typed rule")
	}
	for index, rule := range acceptance.Rules {
		if rule.Question == "" && rule.QuestionType == "" {
			return fmt.Errorf("rules[%d] must name a question or question_type", index)
		}
		switch rule.QuestionType {
		case "", "choice", "score", "noul":
		default:
			return fmt.Errorf("rules[%d].question_type must be choice, score, or noul", index)
		}
		if rule.Field != "confidence" && rule.Field != "top_probability" && rule.Field != "probability_margin" {
			return fmt.Errorf("rules[%d].field must be confidence, top_probability, or probability_margin; score value is not confidence", index)
		}
		if rule.Field == "probability_margin" && rule.QuestionType != "" && rule.QuestionType != "noul" {
			return fmt.Errorf("rules[%d]: probability_margin applies only to noul", index)
		}
		if err := validateNumericPredicateContract(&rule.Predicate); err != nil {
			return fmt.Errorf("rules[%d].predicate: %w", index, err)
		}
		for _, bound := range []*float64{rule.Predicate.GT, rule.Predicate.GTE, rule.Predicate.LT, rule.Predicate.LTE} {
			if bound != nil && (math.IsNaN(*bound) || math.IsInf(*bound, 0) || *bound < 0 || *bound > 1) {
				return fmt.Errorf("rules[%d].predicate: probability bounds must be finite and in [0, 1]", index)
			}
		}
	}
	return nil
}
