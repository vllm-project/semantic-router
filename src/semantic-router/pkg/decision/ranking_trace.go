package decision

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// RankingTrace records how selection ordered the matched decisions: which
// strategy ran, whether the winning pool could be ranked by confidence and why
// not, and which key separated the winner from the decision behind it.
type RankingTrace struct {
	Strategy   string `json:"strategy"`
	Tiered     bool   `json:"tiered"`
	Tier       int    `json:"tier"`
	Comparable bool   `json:"comparable"`
	Fallback   string `json:"fallback_reason,omitempty"`
	ScoreKind  string `json:"score_kind,omitempty"`
	DecidedBy  string `json:"decided_by"`
	Winner     string `json:"winner"`
	Candidates int    `json:"candidates"`
}

func (e *DecisionEngine) rankingTrace(
	winner DecisionResult,
	runnerUp *DecisionResult,
	useTieredSelection bool,
	comparable map[int]bool,
	fallbacks map[int]string,
	candidates int,
) *RankingTrace {
	if winner.Decision == nil {
		return nil
	}
	pool := 0
	if useTieredSelection {
		pool = winner.Decision.Tier
	}
	trace := &RankingTrace{
		Strategy:   string(e.rankingStrategy()),
		Tiered:     useTieredSelection,
		Tier:       pool,
		Comparable: comparable[pool],
		Fallback:   fallbacks[pool],
		ScoreKind:  string(winner.ScoreKind),
		DecidedBy:  "only_candidate",
		Winner:     winner.Decision.Name,
		Candidates: candidates,
	}
	if runnerUp != nil {
		trace.DecidedBy = e.decidingKey(winner, *runnerUp, useTieredSelection, comparable[pool])
	}
	return trace
}

// rankingStrategy reports the strategy selection ran under. An unset strategy
// orders by priority, the same as the explicit value.
func (e *DecisionEngine) rankingStrategy() config.RoutingStrategy {
	if e.strategy == config.RoutingStrategyConfidence {
		return config.RoutingStrategyConfidence
	}
	return config.RoutingStrategyPriority
}

// decidingKey names the first key that separated the winner from the decision
// ranked behind it, in the order the comparator consults them.
func (e *DecisionEngine) decidingKey(winner, runnerUp DecisionResult, useTieredSelection, comparable bool) string {
	if useTieredSelection && winner.Decision.Tier != runnerUp.Decision.Tier {
		return "tier"
	}
	if winner.CatchAll != runnerUp.CatchAll {
		return "catch_all"
	}
	byConfidence := comparable && winner.Confidence != runnerUp.Confidence
	byPriority := winner.Decision.Priority != runnerUp.Decision.Priority
	keys := []string{"priority", "confidence"}
	if e.rankingStrategy() == config.RoutingStrategyConfidence {
		keys = []string{"confidence", "priority"}
	}
	for _, key := range keys {
		if (key == "confidence" && byConfidence) || (key == "priority" && byPriority) {
			return key
		}
	}
	return "name"
}
