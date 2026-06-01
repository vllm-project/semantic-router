/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package selection

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func buildMFSelector(cfg *MultiFactorConfig, params map[string]config.ModelParams,
	inflight func(string) int,
	tpot func(string, int) (float64, bool),
	ttft func(string, int) (float64, bool),
) *MultiFactorSelector {
	s := NewMultiFactorSelector(cfg)
	if params != nil {
		s.InitializeFromConfig(params)
	}
	if inflight != nil {
		s.getInflight = inflight
	}
	if tpot != nil {
		s.getTPOT = tpot
	}
	if ttft != nil {
		s.getTTFT = ttft
	}
	return s
}

func candidates(names ...string) []config.ModelRef {
	out := make([]config.ModelRef, 0, len(names))
	for _, n := range names {
		out = append(out, config.ModelRef{Model: n})
	}
	return out
}

func TestMultiFactor_PicksHighestQualityWhenQualityDominant(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Quality: 1.0}
	params := map[string]config.ModelParams{
		"a": {QualityScore: 0.5},
		"b": {QualityScore: 0.9},
		"c": {QualityScore: 0.3},
	}
	s := buildMFSelector(cfg, params,
		func(string) int { return 0 },
		func(string, int) (float64, bool) { return 0, false },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b", "c")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "b" {
		t.Errorf("expected b (highest quality), got %s; scores=%v", res.SelectedModel, res.AllScores)
	}
}

func TestMultiFactor_PicksCheapestWhenCostDominant(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Cost: 1.0}
	params := map[string]config.ModelParams{
		"expensive": {Pricing: config.ModelPricing{PromptPer1M: 30.0}},
		"cheap":     {Pricing: config.ModelPricing{PromptPer1M: 0.5}},
		"mid":       {Pricing: config.ModelPricing{PromptPer1M: 5.0}},
	}
	s := buildMFSelector(cfg, params,
		func(string) int { return 0 },
		func(string, int) (float64, bool) { return 0, false },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("expensive", "cheap", "mid")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "cheap" {
		t.Errorf("expected cheap, got %s; scores=%v", res.SelectedModel, res.AllScores)
	}
}

func TestMultiFactor_PicksLeastLoadedWhenLoadDominant(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Load: 1.0}
	loadMap := map[string]int{"a": 100, "b": 5, "c": 50}
	s := buildMFSelector(cfg, nil,
		func(m string) int { return loadMap[m] },
		func(string, int) (float64, bool) { return 0, false },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b", "c")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "b" {
		t.Errorf("expected b (lowest load=5), got %s; scores=%v", res.SelectedModel, res.AllScores)
	}
}

func TestMultiFactor_PicksFastestWhenLatencyDominant(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Latency: 1.0}
	tpotMap := map[string]float64{"slow": 0.5, "fast": 0.05, "mid": 0.2}
	s := buildMFSelector(cfg, nil,
		func(string) int { return 0 },
		func(m string, _ int) (float64, bool) { v, ok := tpotMap[m]; return v, ok },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("slow", "fast", "mid")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "fast" {
		t.Errorf("expected fast (lowest TPOT), got %s; scores=%v", res.SelectedModel, res.AllScores)
	}
}

func TestMultiFactor_SLOExcludesViolators(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.SLO.MaxTPOTMs = 100
	cfg.Weights = MultiFactorWeights{Quality: 1.0}
	params := map[string]config.ModelParams{
		"slow_high_q": {QualityScore: 0.99},
		"fast_low_q":  {QualityScore: 0.3},
	}
	tpot := map[string]float64{
		"slow_high_q": 0.5,
		"fast_low_q":  0.05,
	}
	s := buildMFSelector(cfg, params,
		func(string) int { return 0 },
		func(m string, _ int) (float64, bool) { v, ok := tpot[m]; return v, ok },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("slow_high_q", "fast_low_q")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "fast_low_q" {
		t.Errorf("SLO should exclude slow_high_q despite higher quality; got %s reasoning=%q",
			res.SelectedModel, res.Reasoning)
	}
	if !strings.Contains(res.Reasoning, "dropped=1") {
		t.Errorf("reasoning should report dropped=1, got %q", res.Reasoning)
	}
}

func TestMultiFactor_DefaultCheapestWhenAllSLOExcluded(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.SLO.MaxTPOTMs = 1
	cfg.OnNoCandidates = "cheapest"
	params := map[string]config.ModelParams{
		"a": {Pricing: config.ModelPricing{PromptPer1M: 5.0}},
		"b": {Pricing: config.ModelPricing{PromptPer1M: 0.5}},
	}
	s := buildMFSelector(cfg, params,
		func(string) int { return 0 },
		func(string, int) (float64, bool) { return 0.5, true },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "b" {
		t.Errorf("expected cheapest default to pick b, got %s", res.SelectedModel)
	}
	if !strings.Contains(res.Reasoning, "no-candidate policy") {
		t.Errorf("expected no-candidate policy marker in reasoning, got %q", res.Reasoning)
	}
}

func TestMultiFactor_FailWhenAllSLOExcluded(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.SLO.MaxTPOTMs = 1
	cfg.OnNoCandidates = "fail"
	s := buildMFSelector(cfg, nil,
		func(string) int { return 0 },
		func(string, int) (float64, bool) { return 0.5, true },
		func(string, int) (float64, bool) { return 0, false },
	)
	_, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("a", "b")})
	if err == nil {
		t.Fatal("expected error when on_no_candidates=fail and all candidates excluded")
	}
}

func TestMultiFactor_WeightsNormalized(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Quality: 4, Latency: 2, Cost: 2, Load: 2}
	s := NewMultiFactorSelector(cfg)
	w := s.config.Weights
	sum := w.Quality + w.Latency + w.Cost + w.Load
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("weights should sum to 1, got %.6f", sum)
	}
	if math.Abs(w.Quality-0.4) > 1e-9 {
		t.Errorf("quality should be 4/10=0.4, got %.6f", w.Quality)
	}
}

func TestMultiFactor_AllZeroWeightsResetToEqual(t *testing.T) {
	cfg := &MultiFactorConfig{Weights: MultiFactorWeights{}}
	s := NewMultiFactorSelector(cfg)
	w := s.config.Weights
	if w.Quality != 0.25 || w.Latency != 0.25 || w.Cost != 0.25 || w.Load != 0.25 {
		t.Errorf("zero weights should reset to equal 0.25 each, got %+v", w)
	}
}

func TestMultiFactor_NegativeWeightClamped(t *testing.T) {
	cfg := DefaultMultiFactorConfig()
	cfg.Weights = MultiFactorWeights{Quality: -1, Latency: 1, Cost: 1, Load: 1}
	s := NewMultiFactorSelector(cfg)
	if s.config.Weights.Quality != 0 {
		t.Errorf("negative quality should clamp to 0, got %.4f", s.config.Weights.Quality)
	}
}

func TestMultiFactor_NoCandidatesErrors(t *testing.T) {
	s := NewMultiFactorSelector(DefaultMultiFactorConfig())
	_, err := s.Select(context.Background(), &SelectionContext{})
	if err == nil {
		t.Fatal("expected error on empty candidate list")
	}
}

func TestMultiFactor_SingleCandidateReturnedDirectly(t *testing.T) {
	s := buildMFSelector(DefaultMultiFactorConfig(), nil,
		func(string) int { return 0 },
		func(string, int) (float64, bool) { return 0, false },
		func(string, int) (float64, bool) { return 0, false },
	)
	res, err := s.Select(context.Background(), &SelectionContext{CandidateModels: candidates("only")})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res.SelectedModel != "only" {
		t.Errorf("expected 'only', got %s", res.SelectedModel)
	}
}

func TestMultiFactor_DefaultPercentileNormalized(t *testing.T) {
	cfg := &MultiFactorConfig{LatencyPercentile: 0}
	s := NewMultiFactorSelector(cfg)
	if s.config.LatencyPercentile != defaultMFLatencyPercentile {
		t.Errorf("zero LatencyPercentile should normalize to %d, got %d",
			defaultMFLatencyPercentile, s.config.LatencyPercentile)
	}
	cfg2 := &MultiFactorConfig{LatencyPercentile: 150}
	s2 := NewMultiFactorSelector(cfg2)
	if s2.config.LatencyPercentile != defaultMFLatencyPercentile {
		t.Errorf("invalid LatencyPercentile=150 should normalize to %d, got %d",
			defaultMFLatencyPercentile, s2.config.LatencyPercentile)
	}
}

func TestMultiFactor_MethodAndTier(t *testing.T) {
	s := NewMultiFactorSelector(nil)
	if s.Method() != MethodMultiFactor {
		t.Errorf("Method() = %q, want %q", s.Method(), MethodMultiFactor)
	}
	if s.Tier() != TierSupported {
		t.Errorf("Tier() = %q, want %q", s.Tier(), TierSupported)
	}
	if deps := s.ExternalDependencies(); len(deps) != 0 {
		t.Errorf("ExternalDependencies() should be empty, got %v", deps)
	}
}

func TestMultiFactor_NormalizeMinEqualsMax(t *testing.T) {
	if got := normalizeDirect(5, 5, 5, true); got != 0.5 {
		t.Errorf("when min==max, normalizeDirect should be 0.5, got %.4f", got)
	}
	if got := normalizeInverted(5, 5, 5, true); got != 0.5 {
		t.Errorf("when min==max, normalizeInverted should be 0.5, got %.4f", got)
	}
}
