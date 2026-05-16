package classification

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// SignalSessionContext carries request-scoped SESSION_STATE snapshots and optional
// table lookup resolution for session_metric rule evaluation.
type SignalSessionContext struct {
	// Scalars maps dotted SESSION_STATE paths for numeric fields, e.g.
	// "session_routing.cumulative_cost_usd" → 1.23
	Scalars map[string]float64
	// Strings maps dotted SESSION_STATE paths for string fields used as lookup keys.
	Strings map[string]string
	Lookup  LookupResolver
}

// LookupResolver resolves lookup table rows into a single float64 routing input.
type LookupResolver interface {
	LookupFloat64(table string, key []string) (float64, error)
}

func (c *Classifier) hydrateSessionLookupRulesForProjections(results *SignalResults, ctx *SignalSessionContext) []error {
	if c == nil || c.Config == nil || results == nil {
		return nil
	}
	if len(c.Config.SessionMetricRules) == 0 {
		return nil
	}
	if ctx == nil {
		return nil
	}
	return HydrateSessionLookupRuleValues(c.Config, results, ctx)
}

// HydrateSessionLookupRuleValues evaluates configured session_metric rules into
// results.SignalValues and matched slices. Errors describe resolution failures
// (missing scalars, incomplete keys, lookup resolver errors).
func HydrateSessionLookupRuleValues(cfg *config.RouterConfig, results *SignalResults, ctx *SignalSessionContext) []error {
	var errs []error
	for _, rule := range cfg.SessionMetricRules {
		if err := hydrateOneSessionMetricRule(rule, results, ctx); err != nil {
			errs = append(errs, err)
		}
	}
	return errs
}

func sessionMetricSignalKey(ruleName string) string {
	return strings.ToLower(config.SignalTypeSessionMetric) + ":" + ruleName
}

func recordSessionMetricHit(results *SignalResults, ruleName string, v float64) {
	key := sessionMetricSignalKey(ruleName)
	results.SignalValues[key] = v
	results.MatchedSessionMetricRules = append(results.MatchedSessionMetricRules, ruleName)
	if results.SignalConfidences == nil {
		results.SignalConfidences = make(map[string]float64)
	}
	results.SignalConfidences[key] = 1.0
}

func hydrateOneSessionMetricRule(rule config.SessionMetricRule, results *SignalResults, ctx *SignalSessionContext) error {
	switch normalizedSessionMetricKind(rule) {
	case "state":
		v, err := evalStateSessionMetric(rule, ctx.Scalars)
		if err != nil {
			return fmt.Errorf("session_metric rule %q (state): %w", rule.Name, err)
		}
		recordSessionMetricHit(results, rule.Name, v)
		return nil
	case "lookup":
		if ctx.Lookup == nil {
			return fmt.Errorf("session_metric rule %q (lookup): lookup resolver not configured", rule.Name)
		}
		keyParts, ok := resolveLookupKeyParts(rule.Key, ctx)
		if !ok {
			return fmt.Errorf("session_metric rule %q (lookup): incomplete key from session context", rule.Name)
		}
		v, err := ctx.Lookup.LookupFloat64(rule.Table, keyParts)
		if err != nil {
			return fmt.Errorf("session_metric rule %q (lookup): %w", rule.Name, err)
		}
		recordSessionMetricHit(results, rule.Name, v)
		return nil
	default:
		return fmt.Errorf("session_metric rule %q: invalid kind %q (want state or lookup)", rule.Name, rule.Kind)
	}
}

func normalizedSessionMetricKind(rule config.SessionMetricRule) string {
	k := strings.ToLower(strings.TrimSpace(rule.Kind))
	if k == "state" || k == "lookup" {
		return k
	}
	// Infer when kind omitted (YAML fragments may omit explicit kind).
	if strings.TrimSpace(rule.Table) != "" {
		return "lookup"
	}
	if strings.TrimSpace(rule.State) != "" {
		return "state"
	}
	return ""
}

func evalStateSessionMetric(rule config.SessionMetricRule, scalars map[string]float64) (float64, error) {
	if scalars == nil {
		return 0, fmt.Errorf("missing session scalar %q", rule.State)
	}
	v, ok := scalars[rule.State]
	if !ok {
		return 0, fmt.Errorf("missing session scalar %q", rule.State)
	}
	norm := strings.ToLower(strings.TrimSpace(rule.Normalize))
	if norm == "" || norm == "identity" {
		return v, nil
	}
	if norm != "minmax" {
		return 0, fmt.Errorf("unknown normalize %q", rule.Normalize)
	}
	if rule.Min == nil || rule.Max == nil {
		return 0, fmt.Errorf("minmax requires min and max")
	}
	if *rule.Max <= *rule.Min {
		return 0, fmt.Errorf("invalid minmax bounds")
	}
	x := (v - *rule.Min) / (*rule.Max - *rule.Min)
	if x < 0 {
		x = 0
	}
	if x > 1 {
		x = 1
	}
	return x, nil
}

func resolveLookupKeyParts(parts []string, ctx *SignalSessionContext) ([]string, bool) {
	if ctx == nil {
		return nil, false
	}
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if ctx.Strings != nil {
			if s, ok := ctx.Strings[p]; ok && s != "" {
				out = append(out, s)
				continue
			}
		}
		if ctx.Scalars != nil {
			if v, ok := ctx.Scalars[p]; ok {
				out = append(out, strconv.FormatFloat(v, 'f', -1, 64))
				continue
			}
		}
		return nil, false
	}
	return out, true
}
