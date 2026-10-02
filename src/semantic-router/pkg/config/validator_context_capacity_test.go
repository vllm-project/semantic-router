package config

import "testing"

// capacityConfig builds a config with the given bands, the decisions gating
// them, and the declared model windows.
func capacityConfig(rules []ContextRule, models map[string]int, decisions ...Decision) *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ContextRules = rules
	cfg.Decisions = decisions
	cfg.ModelConfig = make(map[string]ModelParams, len(models))
	for name, window := range models {
		cfg.ModelConfig[name] = ModelParams{ContextWindowSize: window}
	}
	return cfg
}

func ruleDecision(name string, rules RuleCombination, models ...string) Decision {
	refs := make([]ModelRef, 0, len(models))
	for _, model := range models {
		refs = append(refs, ModelRef{Model: model})
	}
	return Decision{Name: name, Rules: rules, ModelRefs: refs}
}

func gatedDecision(name string, band string, models ...string) Decision {
	return ruleDecision(name, bandLeaf(band), models...)
}

func bandLeaf(band string) RuleNode {
	return RuleNode{Type: SignalTypeContext, Name: band}
}

func notRule(child RuleNode) RuleCombination {
	return RuleCombination{Operator: RuleOperatorNot, Conditions: []RuleNode{child}}
}

func TestContextCapacityWarnsWhenALargerModelIsUnreachable(t *testing.T) {
	cfg := capacityConfig(
		[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
		map[string]int{"frontier": 200000, "midtier": 200000, "large-window": 262000},
		gatedDecision("large-context", "oversized", "frontier"),
		gatedDecision("overflow", "other-band", "large-window"),
	)

	findings := contextCapacityIssues(cfg)
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}
	finding := findings[0]
	if !finding.gainsNoCapacity() {
		t.Errorf("expected a no-capacity finding, got %+v", finding)
	}
	if finding.unreachableModel != "large-window" || finding.unreachableWindow != 262000 {
		t.Errorf("expected the unreachable model named, got %q at %d", finding.unreachableModel, finding.unreachableWindow)
	}
	if finding.reachableWindow != 200000 {
		t.Errorf("expected reachable window 200000, got %d", finding.reachableWindow)
	}
	if finding.cannotServe() {
		t.Errorf("180K fits a 200000 window; should not report cannot-serve: %+v", finding)
	}
}

func TestContextCapacitySilentWhenBandReachesTheLargestModel(t *testing.T) {
	// The shape of the shipped accuracy recipe: a deliberate single-model
	// long-context route that already uses the biggest declared window,
	// alongside lower-priority decisions on equal-or-smaller models.
	cfg := capacityConfig(
		[]ContextRule{{Name: "long", MinTokens: "16K", MaxTokens: "1M"}},
		map[string]int{"big": 1048576, "peer": 1000000, "small": 32768},
		gatedDecision("long-direct", "long", "big"),
		gatedDecision("workflow", "other-band", "peer", "small"),
	)

	if findings := contextCapacityIssues(cfg); len(findings) != 0 {
		t.Errorf("expected silence when the band already routes to the largest model, got %+v", findings)
	}
}

func TestContextCapacityIgnoresALargerModelNoDecisionRoutesTo(t *testing.T) {
	// model_config is a catalogue and may be wider than one recipe's
	// decisions. A model routed nowhere is not an alternative the operator
	// could switch a band to, so it must not trigger the advisory.
	cfg := capacityConfig(
		[]ContextRule{{Name: "band_a", MinTokens: "10K"}, {Name: "band_b", MinTokens: "50K"}},
		map[string]int{"worker": 200000, "declared-but-unrouted": 1000000},
		gatedDecision("d_a", "band_a", "worker"),
		gatedDecision("d_b", "band_b", "worker"),
	)

	if findings := contextCapacityIssues(cfg); len(findings) != 0 {
		t.Errorf("a model no decision routes to must not be reported as a missed alternative, got %+v", findings)
	}
}

func TestContextCapacityExcludesDecisionsThatOnlyMatchOutsideTheBand(t *testing.T) {
	// A decision gated on NOT oversized only matches requests outside the
	// band, so its larger model is not reachable from inside it.
	cfg := capacityConfig(
		[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
		map[string]int{"frontier": 200000, "large-window": 262000},
		gatedDecision("large-context", "oversized", "frontier"),
		ruleDecision("everything-else", notRule(bandLeaf("oversized")), "large-window"),
	)

	findings := contextCapacityIssues(cfg)
	if len(findings) != 1 || findings[0].unreachableModel != "large-window" {
		t.Fatalf("expected one finding naming large-window, got %+v", findings)
	}
}

func TestContextCapacityKeepsDecisionsReachableThroughANegatedConjunction(t *testing.T) {
	// NOT (oversized AND x) still matches inside the band whenever x is false,
	// so reachability has to be evaluated over the tree, not read off the
	// nearest NOT.
	negated := notRule(RuleNode{
		Operator:   RuleOperatorAnd,
		Conditions: []RuleNode{bandLeaf("oversized"), {Type: SignalTypeKeyword, Name: "x"}},
	})
	cfg := capacityConfig(
		[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
		map[string]int{"frontier": 200000, "large-window": 262000},
		gatedDecision("large-context", "oversized", "frontier"),
		ruleDecision("mixed", negated, "large-window"),
	)

	if findings := contextCapacityIssues(cfg); len(findings) != 0 {
		t.Errorf("large-window is reachable from inside the band when x is false, got %+v", findings)
	}
}

func TestContextCapacityWarnsWhenTheBandCannotBeServed(t *testing.T) {
	cfg := capacityConfig(
		[]ContextRule{{Name: "huge", MinTokens: "500K"}},
		map[string]int{"only": 200000},
		gatedDecision("huge-context", "huge", "only"),
	)

	findings := contextCapacityIssues(cfg)
	if len(findings) != 1 {
		t.Fatalf("expected 1 finding, got %d: %+v", len(findings), findings)
	}
	if !findings[0].cannotServe() || findings[0].minTokens != 500000 {
		t.Errorf("expected a cannot-serve finding at 500000, got %+v", findings[0])
	}
}

func TestContextCapacityIgnoresBandsAndConfigsWithNothingToCheck(t *testing.T) {
	cases := map[string]*RouterConfig{
		"nil config": nil,
		"no context rules": capacityConfig(nil,
			map[string]int{"a": 200000, "b": 262000},
			gatedDecision("d", "missing", "a")),
		"no decisions": capacityConfig(
			[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
			map[string]int{"a": 200000, "b": 262000}),
		"band with no lower bound gates nothing on size": capacityConfig(
			[]ContextRule{{Name: "short", MaxTokens: "1K"}},
			map[string]int{"a": 200000, "b": 262000},
			gatedDecision("d", "short", "a"),
			gatedDecision("other", "elsewhere", "b")),
		"band no decision references": capacityConfig(
			[]ContextRule{{Name: "unused", MinTokens: "180K"}},
			map[string]int{"a": 200000, "b": 262000},
			gatedDecision("d", "other", "a"),
			gatedDecision("other", "elsewhere", "b")),
		"reachable model declares no window": capacityConfig(
			[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
			map[string]int{"b": 262000},
			gatedDecision("d", "oversized", "undeclared"),
			gatedDecision("other", "elsewhere", "b")),
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if findings := contextCapacityIssues(cfg); len(findings) != 0 {
				t.Errorf("expected no findings, got %+v", findings)
			}
			if err := validateContextCapacity(cfg); err != nil {
				t.Errorf("validator must never reject a configuration: %v", err)
			}
		})
	}
}

func TestContextCapacityTakesTheBestWindowAcrossEveryGatingDecision(t *testing.T) {
	// Two decisions gate the same band; the larger of their models counts.
	cfg := capacityConfig(
		[]ContextRule{{Name: "oversized", MinTokens: "180K"}},
		map[string]int{"small": 200000, "roomy": 262000},
		gatedDecision("first", "oversized", "small"),
		gatedDecision("second", "oversized", "roomy"),
	)

	if findings := contextCapacityIssues(cfg); len(findings) != 0 {
		t.Errorf("the band reaches the largest model through its second decision: %+v", findings)
	}
}
