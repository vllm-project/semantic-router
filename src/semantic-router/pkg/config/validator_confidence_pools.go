package config

import (
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// confidencePoolFallback records a ranking pool that cannot order by
// confidence because its members report different score kinds.
type confidencePoolFallback struct {
	Pool      int
	Decisions []string
	Kinds     []string
}

// reportAmbiguousConfidencePools warns about pools whose members report
// different score kinds. Selection compares confidence only within one kind,
// so such a pool always falls back to priority ordering. Saying so at load
// time keeps the configured strategy and the runtime contract aligned.
func reportAmbiguousConfidencePools(cfg *RouterConfig) {
	_ = visitRoutingProfileConfigs(cfg, func(scoped *RouterConfig) error {
		if scoped == nil {
			return nil
		}
		for _, fallback := range ambiguousConfidencePools(scoped.Decisions) {
			logging.Warnf(
				"routing profile %q tier %d ranks by priority: %s report different score kinds (%s). "+
					"Confidence orders a pool only when every member reports the same kind",
				string(scoped.RoutingScope), fallback.Pool,
				strings.Join(fallback.Decisions, ", "), strings.Join(fallback.Kinds, ", "),
			)
		}
		return nil
	})
}

// ambiguousConfidencePools groups decisions the way selection does, by tier
// when any decision sets one and into a single pool otherwise, and returns
// the pools that hold more than one score kind.
func ambiguousConfidencePools(decisions []Decision) []confidencePoolFallback {
	tiered := false
	for i := range decisions {
		if decisions[i].Tier > 0 {
			tiered = true
			break
		}
	}

	members := map[int][]string{}
	kinds := map[int]map[ScoreKind]struct{}{}
	for i := range decisions {
		decision := &decisions[i]
		if decision.Rules.IsCatchAll() {
			continue
		}
		pool := 0
		if tiered {
			pool = decision.Tier
		}
		members[pool] = append(members[pool], decision.Name)
		if kinds[pool] == nil {
			kinds[pool] = map[ScoreKind]struct{}{}
		}
		collectScoreKinds(&decision.Rules, kinds[pool])
	}

	pools := make([]int, 0, len(members))
	for pool := range members {
		pools = append(pools, pool)
	}
	sort.Ints(pools)

	fallbacks := make([]confidencePoolFallback, 0, len(pools))
	for _, pool := range pools {
		if len(members[pool]) < 2 || len(kinds[pool]) < 2 {
			continue
		}
		names := append([]string(nil), members[pool]...)
		sort.Strings(names)
		fallbacks = append(fallbacks, confidencePoolFallback{
			Pool:      pool,
			Decisions: names,
			Kinds:     sortedScoreKinds(kinds[pool]),
		})
	}
	return fallbacks
}

// collectScoreKinds gathers the score kinds a rule tree can report. A
// predicate leaf answers a threshold rather than reporting a measurement, so
// it declares no kind.
func collectScoreKinds(node *RuleNode, kinds map[ScoreKind]struct{}) {
	if node == nil {
		return
	}
	if len(node.Conditions) > 0 {
		for i := range node.Conditions {
			collectScoreKinds(&node.Conditions[i], kinds)
		}
		return
	}
	if node.Predicate != nil {
		return
	}
	if kind := SignalScoreKind(strings.ToLower(strings.TrimSpace(node.Type))); kind != ScoreKindNone {
		kinds[kind] = struct{}{}
	}
}

func sortedScoreKinds(set map[ScoreKind]struct{}) []string {
	kinds := make([]string, 0, len(set))
	for _, kind := range sortedKinds(set) {
		kinds = append(kinds, string(kind))
	}
	return kinds
}

func sortedKinds(set map[ScoreKind]struct{}) []ScoreKind {
	kinds := make([]ScoreKind, 0, len(set))
	for kind := range set {
		kinds = append(kinds, kind)
	}
	sort.Slice(kinds, func(i, j int) bool { return kinds[i] < kinds[j] })
	return kinds
}
