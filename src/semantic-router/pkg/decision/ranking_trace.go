package decision

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
	// RunnerUp is the decision ranked second; DecidedBy separated it from
	// Winner. Empty when only one decision matched.
	RunnerUp string `json:"runner_up,omitempty"`
	// Reason states the values compared, without names (#3658). It is
	// prose: filter on DecidedBy and RunnerUp instead.
	Reason string `json:"reason,omitempty"`
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
		trace.RunnerUp = runnerUp.Decision.Name
		trace.Reason = e.rankingReason(trace.DecidedBy, winner, *runnerUp, comparable[pool])
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
	for _, key := range e.evidenceKeys() {
		if (key == "confidence" && byConfidence) || (key == "priority" && byPriority) {
			return key
		}
	}
	return "name"
}

// evidenceKeys lists priority and confidence in the order routing.strategy
// consults them.
func (e *DecisionEngine) evidenceKeys() []string {
	if e.rankingStrategy() == config.RoutingStrategyConfidence {
		return []string{"confidence", "priority"}
	}
	return []string{"priority", "confidence"}
}

// rankingReason states the comparison that settled the ranking. Keys the
// comparator consulted first read as equal, so a name tie-break shows as
// the last resort it is.
func (e *DecisionEngine) rankingReason(decidedBy string, winner, runnerUp DecisionResult, comparable bool) string {
	switch decidedBy {
	case "tier":
		return fmt.Sprintf("tier %d ranks before tier %d", winner.Decision.Tier, runnerUp.Decision.Tier)
	case "catch_all":
		return "runner-up is a catch-all"
	}
	var clauses []string
	for _, key := range e.evidenceKeys() {
		if key == decidedBy {
			return strings.Join(append(clauses, comparedClause(key, winner, runnerUp)), ", ")
		}
		clauses = append(clauses, equalClause(key, winner, comparable))
	}
	return strings.Join(append(clauses, "decision name ordering"), ", ")
}

func comparedClause(key string, winner, runnerUp DecisionResult) string {
	if key == "priority" {
		return fmt.Sprintf("priority %d > %d", winner.Decision.Priority, runnerUp.Decision.Priority)
	}
	return "confidence " + formatConfidence(winner.Confidence) + " > " + formatConfidence(runnerUp.Confidence)
}

// equalClause describes a key that did not separate the pair. Confidence
// that was never comparable is said so, not reported as equal.
func equalClause(key string, winner DecisionResult, comparable bool) string {
	switch {
	case key == "priority":
		return fmt.Sprintf("equal priority %d", winner.Decision.Priority)
	case comparable:
		return "equal confidence " + formatConfidence(winner.Confidence)
	default:
		return "no comparable confidence"
	}
}

func formatConfidence(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
