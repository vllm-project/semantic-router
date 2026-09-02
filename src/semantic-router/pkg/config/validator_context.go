package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// validateContextContracts checks routing.signals.context bands.
//
// Errors: empty or duplicate names, a rule with neither limit, unparsable or
// negative values, and min_tokens > max_tokens. min_tokens == max_tokens is a
// valid exact-match band, omitting max_tokens is a valid open-ended band, and
// omitting min_tokens means 0.
//
// Gaps and overlaps between bands are allowed but logged, because both change
// what a request can match: a gap leaves counts with no context signal, and an
// overlap reports every matching rule in configuration order. A band that
// fully contains another is a common intentional layout and is logged at info.
func validateContextContracts(cfg *RouterConfig) error {
	if cfg == nil || len(cfg.ContextRules) == 0 {
		return nil
	}
	bands, err := collectContextBands(cfg.ContextRules)
	if err != nil {
		return err
	}
	logContextBandIssues(bands)
	logContextBandCoverage(bands)
	return nil
}

// collectContextBands parses every rule, rejecting empty or duplicate names
// and invalid ranges.
func collectContextBands(rules []ContextRule) ([]NamedContextBand, error) {
	seen := make(map[string]struct{}, len(rules))
	bands := make([]NamedContextBand, 0, len(rules))
	for index, rule := range rules {
		name := strings.TrimSpace(rule.Name)
		if name == "" {
			return nil, fmt.Errorf("routing.signals.context[%d]: name cannot be empty", index)
		}
		if _, exists := seen[name]; exists {
			return nil, fmt.Errorf("routing.signals.context[%q]: duplicate rule name", name)
		}
		seen[name] = struct{}{}

		bounds, err := rule.Bounds()
		if err != nil {
			return nil, fmt.Errorf("routing.signals.context[%q]: %w", name, err)
		}
		bands = append(bands, NamedContextBand{Name: name, Bounds: bounds})
	}
	return bands, nil
}

// logContextBandIssues reports overlaps and gaps. A band that fully contains
// another is logged at info because it is a common intentional layout.
func logContextBandIssues(bands []NamedContextBand) {
	overlaps, gaps := ContextBandIssues(bands)
	for _, overlap := range overlaps {
		if overlap.Contains {
			logging.Infof(
				"routing.signals.context: rule %q %s contains %q %s; requests in the inner band match both and are reported in configuration order",
				overlap.Outer.Name, overlap.Outer.Bounds, overlap.Inner.Name, overlap.Inner.Bounds,
			)
			continue
		}
		logging.Warnf(
			"routing.signals.context: rules %q %s and %q %s overlap; requests in the shared range match both and are reported in configuration order",
			overlap.Outer.Name, overlap.Outer.Bounds, overlap.Inner.Name, overlap.Inner.Bounds,
		)
	}
	for _, gap := range gaps {
		logging.Warnf(
			"routing.signals.context: no rule covers token counts %d to %d (before %q); requests in that range match no context signal",
			gap.From, gap.To, gap.Before.Name,
		)
	}
}

// logContextBandCoverage reports how overflow above the largest bounded band
// is handled: several open-ended bands, or none at all.
func logContextBandCoverage(bands []NamedContextBand) {
	unbounded := 0
	coveredTo := -1
	for _, band := range bands {
		if band.Bounds.Unbounded {
			unbounded++
		} else if band.Bounds.Max > coveredTo {
			coveredTo = band.Bounds.Max
		}
	}
	switch {
	case unbounded > 1:
		logging.Warnf("routing.signals.context: %d rules have no max_tokens; each matches every request at or above its min_tokens", unbounded)
	case unbounded == 0:
		logging.Infof("routing.signals.context: no rule is open-ended; requests above %d tokens match no context signal (omit max_tokens on the last band to catch overflow)", coveredTo)
	}
}
