package selection

import (
	"math"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
)

// Cache economics decisions: keep the current model, allow the switch, or
// leave the switch to the heuristic session scores because a value is unknown.
const (
	CacheEconomicsHold     = "hold"
	CacheEconomicsSwitch   = "switch"
	CacheEconomicsFallback = "fallback"
)

// Cache economics reasons recorded in replay.
const (
	CacheReasonSwitchPays         = "switch_pays"
	CacheReasonSwitchDoesNotPay   = "switch_does_not_pay"
	CacheReasonProposalNotCheaper = "proposal_not_cheaper"
	CacheReasonCacheExpired       = "cache_expired"
	CacheReasonNoWarmCache        = "no_warm_cache"
	CacheReasonNoReceipt          = "no_receipt"
	CacheReasonReceiptOtherModel  = "receipt_other_model"
	CacheReasonCacheNotReported   = "cache_not_reported"
	CacheReasonNoCachePrice       = "no_cache_price"
	CacheReasonCurrencyMismatch   = "currency_mismatch"

	CacheTurnsSourcePrior   = "prior"
	CacheTurnsSourceDefault = "default"

	// Session decision reasons for a switch the receipts decided.
	SessionReasonCacheSwitchPays       = "cache_switch_pays"
	SessionReasonCacheSwitchDoesNotPay = "cache_switch_does_not_pay"
)

// oneHourCacheTTLSeconds is the lifetime of an entry written with a 1-hour
// cache_control ttl.
const oneHourCacheTTLSeconds = 3600

// CacheSwitchInput is what pricing one switch needs: the current model's last
// receipt, the rates of both models, and the turns left to repay a rewrite.
type CacheSwitchInput struct {
	Config              config.CacheEconomicsConfig
	Current, Proposed   string
	Receipt             *modelpricing.Receipt
	ReceiptAge          time.Duration
	Rates               map[string]modelpricing.Rates
	RemainingTurns      float64
	RemainingTurnsKnown bool
}

// CacheSwitchEconomics prices one switch the way a live cache is priced: the
// model holding it reads the prefix at its cache-read rate, and the proposed
// model writes it once before reading it. Costs are in Currency.
type CacheSwitchEconomics struct {
	Decision      string
	Reason        string
	CurrentModel  string
	ProposedModel string
	Currency      string

	// CachedPrefixTokens is what the current model's cache holds after the
	// receipt: its cache reads plus its cache writes.
	CachedPrefixTokens   int
	CacheAgeSeconds      float64
	CacheTTLSeconds      float64
	RemainingTurns       float64
	RemainingTurnsSource string

	// CurrentTurnCost and ProposedTurnCost price the receipt's turn on each
	// model with its cache warm; RewriteCost is the proposed model's first
	// request writing the prefix instead of reading it, and Saving is the
	// turn-cost difference over the remaining turns.
	CurrentTurnCost  float64
	ProposedTurnCost float64
	RewriteCost      float64
	Saving           float64
}

// EvaluateCacheSwitch decides whether a switch repays the prompt cache it
// throws away. A switch to a model that costs no less per turn is the routing
// paying for quality, so it is allowed with its cache cost recorded.
func EvaluateCacheSwitch(in CacheSwitchInput) CacheSwitchEconomics {
	out := CacheSwitchEconomics{
		Decision:      CacheEconomicsFallback,
		CurrentModel:  in.Current,
		ProposedModel: in.Proposed,
	}
	if out.Reason = cacheSwitchUnknown(in); out.Reason != "" {
		return out
	}
	current, proposed := in.Rates[in.Current], in.Rates[in.Proposed]
	usage := modelpricing.Normalize(in.Receipt.Usage)
	writeRate := proposed.EffectiveCacheWritePer1M()
	out.CacheTTLSeconds = float64(in.Config.CacheTTLSeconds)
	if usage.CacheWrite1hTokens > 0 {
		writeRate = proposed.EffectiveCacheWrite1hPer1M()
		out.CacheTTLSeconds = math.Max(out.CacheTTLSeconds, oneHourCacheTTLSeconds)
	}
	out.Currency = current.Currency
	out.CachedPrefixTokens = usage.CachedInputTokens + usage.CacheWriteTokens
	out.CacheAgeSeconds = math.Max(in.ReceiptAge.Seconds(), 0)
	out.RemainingTurns, out.RemainingTurnsSource = float64(in.Config.DefaultRemainingTurns), CacheTurnsSourceDefault
	if in.RemainingTurnsKnown {
		out.RemainingTurns, out.RemainingTurnsSource = math.Max(in.RemainingTurns, 1), CacheTurnsSourcePrior
	}
	out.CurrentTurnCost = modelpricing.Cost(in.Receipt.Usage, current)
	out.ProposedTurnCost = modelpricing.Cost(in.Receipt.Usage, proposed)
	out.RewriteCost = float64(out.CachedPrefixTokens) * math.Max(writeRate-math.Max(proposed.CachedInputPer1M, 0), 0) / 1_000_000
	out.Saving = out.RemainingTurns * (out.CurrentTurnCost - out.ProposedTurnCost)
	out.Decision, out.Reason = cacheSwitchVerdict(out)
	return out
}

func cacheSwitchVerdict(e CacheSwitchEconomics) (string, string) {
	switch {
	case e.CacheAgeSeconds >= e.CacheTTLSeconds:
		return CacheEconomicsSwitch, CacheReasonCacheExpired
	case e.CachedPrefixTokens == 0:
		return CacheEconomicsSwitch, CacheReasonNoWarmCache
	case e.ProposedTurnCost >= e.CurrentTurnCost:
		return CacheEconomicsSwitch, CacheReasonProposalNotCheaper
	case e.Saving >= e.RewriteCost:
		return CacheEconomicsSwitch, CacheReasonSwitchPays
	default:
		return CacheEconomicsHold, CacheReasonSwitchDoesNotPay
	}
}

// cacheSwitchUnknown names the missing value that leaves a switch to the
// heuristic session scores, or "" when the switch can be priced.
func cacheSwitchUnknown(in CacheSwitchInput) string {
	switch {
	case in.Receipt == nil:
		return CacheReasonNoReceipt
	case in.Receipt.Model != in.Current:
		return CacheReasonReceiptOtherModel
	case !in.Receipt.CacheReported:
		return CacheReasonCacheNotReported
	}
	current, currentOK := in.Rates[in.Current]
	proposed, proposedOK := in.Rates[in.Proposed]
	if !currentOK || !proposedOK || !hasCachePrice(current) || !hasCachePrice(proposed) {
		return CacheReasonNoCachePrice
	}
	if current.Currency != proposed.Currency {
		return CacheReasonCurrencyMismatch
	}
	return ""
}

// hasCachePrice reports whether a priced model also prices its cache reads, or
// takes input for free.
func hasCachePrice(rates modelpricing.Rates) bool {
	return rates.IsConfigured() &&
		(rates.CachedInputPer1M > 0 || rates.CacheWritePer1M != nil || rates.PromptPer1M <= 0)
}

// ToMap is the replay shape of one priced switch.
func (e *CacheSwitchEconomics) ToMap() map[string]interface{} {
	if e == nil {
		return nil
	}
	return map[string]interface{}{
		"decision": e.Decision, "reason": e.Reason,
		"current_model": e.CurrentModel, "proposed_model": e.ProposedModel, "currency": e.Currency,
		"cached_prefix_tokens": e.CachedPrefixTokens,
		"cache_age_seconds":    e.CacheAgeSeconds, "cache_ttl_seconds": e.CacheTTLSeconds,
		"remaining_turns": e.RemainingTurns, "remaining_turns_source": e.RemainingTurnsSource,
		"current_turn_cost": e.CurrentTurnCost, "proposed_turn_cost": e.ProposedTurnCost,
		"rewrite_cost": e.RewriteCost, "saving": e.Saving,
	}
}

// cacheSwitch prices a switch from current to candidate with the session's
// last receipt. It returns nil when the policy is off, the candidate keeps the
// current model, or a client release asked for a fresh decision.
func (s *SessionAwareSelector) cacheSwitch(
	session *AgenticSessionContext,
	current string,
	candidate string,
	continuation continuationEvidence,
) *CacheSwitchEconomics {
	if !s.config.CacheEconomics.Enabled || session == nil || session.ReleaseHonored || candidate == current {
		return nil
	}
	economics := EvaluateCacheSwitch(CacheSwitchInput{
		Config:              s.config.CacheEconomics,
		Current:             current,
		Proposed:            candidate,
		Receipt:             session.CacheReceipt,
		ReceiptAge:          session.CacheReceiptAge,
		Rates:               session.ModelRates,
		RemainingTurns:      continuation.RemainingTurnsEstimate,
		RemainingTurnsKnown: continuation.RemainingTurnPriorOK,
	})
	return &economics
}

// priceCacheSwitch replaces a switch candidate's heuristic cache terms, its
// prefix penalty and the current model's prefix benefit, with the switch
// priced from receipts when every value is known.
func (s *SessionAwareSelector) priceCacheSwitch(
	session *AgenticSessionContext,
	current string,
	candidate string,
	continuation continuationEvidence,
	score float64,
	currentTrace SessionCandidateTrace,
	trace SessionCandidateTrace,
) (float64, SessionCandidateTrace) {
	trace.CacheEconomics = s.cacheSwitch(session, current, candidate, continuation)
	if trace.CacheEconomics == nil || trace.CacheEconomics.Decision == CacheEconomicsFallback {
		return score, trace
	}
	score += trace.PrefixCachePenalty + currentTrace.PrefixCacheBenefit
	trace.PrefixCachePenalty = 0
	trace.NetSwitchAdvantage = score - currentTrace.FinalScore
	trace.FinalScore = score
	return score, trace
}

// withoutUnpaidSwitches drops the candidates whose switch the receipts show
// does not repay its cache rewrite.
func withoutUnpaidSwitches(scores CandidateScores, traces map[string]SessionCandidateTrace) CandidateScores {
	unpaid := map[string]bool{}
	for _, trace := range traces {
		if e := trace.CacheEconomics; e != nil && e.Decision == CacheEconomicsHold {
			unpaid[e.ProposedModel] = true
		}
	}
	if len(unpaid) == 0 {
		return scores
	}
	kept := make(CandidateScores, 0, len(scores))
	for _, row := range scores {
		if !unpaid[row.Candidate.Model] {
			kept = append(kept, row)
		}
	}
	return kept
}

// recordCacheEconomics puts the routing's proposed switch on the policy trace.
// The decision reason names the receipts only where they changed or settled
// the outcome: they refused the switch the scores would have made, or they
// priced the switch that was made.
func recordCacheEconomics(trace *SessionPolicyTrace, traces map[string]SessionCandidateTrace, proposal, scoreBest, selected string) {
	for _, candidate := range traces {
		if e := candidate.CacheEconomics; e != nil && e.ProposedModel == proposal {
			trace.CacheEconomics = e
			break
		}
	}
	switch e := trace.CacheEconomics; {
	case e == nil:
	case e.Decision == CacheEconomicsHold && scoreBest == proposal && selected == trace.CurrentModel:
		trace.DecisionReason = SessionReasonCacheSwitchDoesNotPay
	case e.Reason == CacheReasonSwitchPays && selected == proposal:
		trace.DecisionReason = SessionReasonCacheSwitchPays
	}
}
