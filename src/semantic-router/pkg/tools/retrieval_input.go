package tools

// Default candidate-pool heuristics match extproc when advanced filtering is off.
const (
	DefaultCandidatePoolMultiplier = 5
	DefaultCandidatePoolMin        = 20
)

// StrategyDefault is the built-in embedding-similarity retriever name.
const StrategyDefault = "default"

// RetrievalInput carries shared context for tool retrieval. Strategies use the
// fields they need; others may be left empty.
type RetrievalInput struct {
	Query              string
	HistorySummary     string
	Category           string
	DecisionName       string
	DecisionConfidence float64
	// TopK is the desired number of tools after post-retrieval filtering.
	TopK int
	// PoolSize is how many candidates to pull from the index (before advanced filters).
	PoolSize int
}

// EffectivePoolSize returns PoolSize, or a conservative default when unset
// (aligned with the legacy single-path behavior).
func (in RetrievalInput) EffectivePoolSize() int {
	if in.PoolSize > 0 {
		return in.PoolSize
	}
	if in.TopK > 0 {
		if n := in.TopK * DefaultCandidatePoolMultiplier; n > DefaultCandidatePoolMin {
			return n
		}
		return DefaultCandidatePoolMin
	}
	return DefaultCandidatePoolMin
}
