package main

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

type nativeTestBlockRunner struct {
	classifier *classification.Classifier
}

func buildNativeTestBlockRunner(prog *dsl.Program) (dsl.TestBlockRunner, error) {
	if prog == nil {
		return nil, fmt.Errorf("program is nil")
	}

	cfg, compileErrs := dsl.CompileAST(prog)
	if len(compileErrs) > 0 {
		return nil, fmt.Errorf("TEST block compile failed: %v", compileErrs)
	}

	classifier, err := classification.NewClassifier(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create classifier for TEST blocks: %w", err)
	}
	return &nativeTestBlockRunner{classifier: classifier}, nil
}

func (r *nativeTestBlockRunner) EvaluateTestBlockQuery(query string) (*dsl.TestBlockResult, error) {
	signals, err := r.classifier.EvaluateAllSignalsWithHeaders(classification.SignalEvaluationInput{
		Text:            query,
		ContextText:     query,
		CurrentUserText: query,
	})
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
