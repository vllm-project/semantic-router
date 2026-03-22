package main

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

type nativeTestBlockRunner struct {
	classifier *classification.Classifier
}

const signalGroupCentroidWarningThreshold = 0.7

func buildNativeTestBlockRunner(prog *dsl.Program) (dsl.TestBlockRunner, error) {
	if prog == nil {
		return nil, fmt.Errorf("program is nil")
	}

	cfg, compileErrs := dsl.CompileAST(prog)
	if len(compileErrs) > 0 {
		return nil, fmt.Errorf("TEST block compile failed: %v", compileErrs)
	}

	categoryMapping, piiMapping, jailbreakMapping, err := loadClassifierMappings(cfg)
	if err != nil {
		return nil, err
	}

	classifier, err := classification.NewClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier for TEST blocks: %w", err)
	}
	return &nativeTestBlockRunner{classifier: classifier}, nil
}

func loadClassifierMappings(
	cfg *config.RouterConfig,
) (
	*classification.CategoryMapping,
	*classification.PIIMapping,
	*classification.JailbreakMapping,
	error,
) {
	var (
		categoryMapping  *classification.CategoryMapping
		piiMapping       *classification.PIIMapping
		jailbreakMapping *classification.JailbreakMapping
		err              error
	)

	if cfg.CategoryMappingPath != "" {
		categoryMapping, err = classification.LoadCategoryMapping(cfg.CategoryMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load category mapping: %w", err)
		}
	}

	if cfg.PIIMappingPath != "" {
		piiMapping, err = classification.LoadPIIMapping(cfg.PIIMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load PII mapping: %w", err)
		}
	}

	if cfg.PromptGuard.JailbreakMappingPath != "" {
		jailbreakMapping, err = classification.LoadJailbreakMapping(cfg.PromptGuard.JailbreakMappingPath)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to load jailbreak mapping: %w", err)
		}
	}

	return categoryMapping, piiMapping, jailbreakMapping, nil
}

func (r *nativeTestBlockRunner) EvaluateTestBlockQuery(query string) (*dsl.TestBlockResult, error) {
	signals, err := r.classifier.EvaluateAllSignalsWithHeaders(query, query, nil, nil, false, "")
	if err != nil {
		return nil, err
	}

	result, err := r.classifier.EvaluateDecisionWithEngine(signals)
	if err != nil {
		return nil, err
	}
	if result == nil || result.Decision == nil {
		return nil, nil
	}

	return &dsl.TestBlockResult{
		DecisionName: result.Decision.Name,
		Confidence:   result.Confidence,
		MatchedRules: append([]string(nil), result.MatchedRules...),
	}, nil
}

func (r *nativeTestBlockRunner) ValidateSignalGroups(prog *dsl.Program) []dsl.Diagnostic {
	if r == nil || r.classifier == nil || prog == nil {
		return nil
	}

	warnings, err := r.classifier.AnalyzeSoftmaxSignalGroupCentroids(signalGroupCentroidWarningThreshold)
	if err != nil {
		return []dsl.Diagnostic{{
			Level:   dsl.DiagError,
			Message: fmt.Sprintf("SIGNAL_GROUP native centroid validation failed: %v", err),
			Pos:     firstSoftmaxSignalGroupPos(prog),
		}}
	}
	if len(warnings) == 0 {
		return nil
	}

	positions := signalGroupPositions(prog)
	diags := make([]dsl.Diagnostic, 0, len(warnings))
	for _, warning := range warnings {
		diags = append(diags, dsl.Diagnostic{
			Level: dsl.DiagWarning,
			Pos:   positions[warning.GroupName],
			Message: fmt.Sprintf(
				`SIGNAL_GROUP %s: members %q and %q candidate centroids have cosine similarity %.2f (threshold: %.1f) — softmax scores may be near-uniform on ambiguous queries`,
				warning.GroupName,
				warning.LeftMember,
				warning.RightMember,
				warning.Similarity,
				signalGroupCentroidWarningThreshold,
			),
		})
	}
	return diags
}

func signalGroupPositions(prog *dsl.Program) map[string]dsl.Position {
	positions := make(map[string]dsl.Position, len(prog.SignalGroups))
	for _, group := range prog.SignalGroups {
		positions[group.Name] = group.Pos
	}
	return positions
}

func firstSoftmaxSignalGroupPos(prog *dsl.Program) dsl.Position {
	for _, group := range prog.SignalGroups {
		if group.Semantics == "softmax_exclusive" {
			return group.Pos
		}
	}
	return dsl.Position{}
}
