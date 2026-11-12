package sessiontools

import (
	"fmt"
	"slices"
	"time"
)

// Merge deterministically plans against a single snapshot. Compatible tools
// keep their relative order. Missing required names, then called names, are
// admitted in name order, replacing the newest unpinned/unrequired entry if
// needed. Relevance additions follow score/name order and never replace.
// Pin overflow resets continuity and seeds a bounded fresh ranking instead.
// This promises determinism for an input and committed operation order, not
// identical choices across different concurrent schedules.
func Merge(input MergeInput) (MergeResult, error) {
	p, err := prepareSelection(input.SelectionInput)
	if err != nil {
		return MergeResult{}, err
	}
	return p.merge(input.Prior, input.Now, input.TTL)
}

func (p preparedSelection) merge(prior *State, now time.Time, ttl time.Duration) (MergeResult, error) {
	if now.IsZero() || ttl <= 0 {
		return MergeResult{}, fmt.Errorf("%w: clock and TTL must be set", ErrInvalidSelection)
	}
	now = now.UTC()
	reason := p.resetReason(prior, now)
	state := p.newState(now, ttl)
	if reason == ReasonNone {
		if prior.Turn == MaxSelectionTurn {
			return MergeResult{}, ErrTurnExhausted
		}
		state.CreatedAt, state.Turn = prior.CreatedAt.UTC(), prior.Turn+1
		state.StrategyID = prior.StrategyID
		state.Tools = p.retain(prior.Tools)
	}
	b := mergeBuilder{selection: p, state: state}
	if err := b.fill(reason != ReasonNone, true); err != nil {
		return MergeResult{}, err
	}
	if b.pinOverflow {
		reason = ReasonPinCapacity
		b = mergeBuilder{selection: p, state: p.newState(now, ttl)}
		if err := b.fill(true, false); err != nil {
			return MergeResult{}, err
		}
	}
	return makeMergeResult(b.state, prior, reason), nil
}

func (p preparedSelection) resetReason(prior *State, now time.Time) SelectionReason {
	if prior == nil {
		return ReasonMissing
	}
	if prior.Validate(p.input.Bounds.MaxTools, p.input.Bounds.MaxStateBytes) != nil {
		return ReasonInvalidState
	}
	if !prior.ExpiresAt.After(now) {
		return ReasonExpired
	}
	if now.Before(prior.LastSeenAt) {
		return ReasonClockRegressed
	}
	if prior.PolicyFingerprint != p.input.Fingerprints.Policy {
		return ReasonPolicyChanged
	}
	if prior.CatalogFingerprint != p.input.Fingerprints.Catalog {
		return ReasonCatalogChanged
	}
	if prior.CapabilityFingerprint != p.input.Fingerprints.Capability {
		return ReasonCapabilityChanged
	}
	return ReasonNone
}

func (p preparedSelection) newState(now time.Time, ttl time.Duration) State {
	return State{
		SchemaVersion: SchemaVersion, Turn: 1,
		PolicyFingerprint: p.input.Fingerprints.Policy, CatalogFingerprint: p.input.Fingerprints.Catalog,
		CapabilityFingerprint: p.input.Fingerprints.Capability,
		CreatedAt:             now, LastSeenAt: now, ExpiresAt: now.Add(ttl),
		Tools: make([]ToolState, 0),
	}
}

func (p preparedSelection) retain(prior []ToolState) []ToolState {
	result := make([]ToolState, 0, len(prior))
	for _, tool := range prior {
		current, ok := p.eligible[tool.Name]
		if !ok || current.DefinitionFingerprint != tool.DefinitionFingerprint {
			continue
		}
		tool.Pinned = p.input.Bounds.PinCalledTools && (tool.Pinned || p.called[tool.Name])
		result = append(result, tool)
	}
	return result
}

func makeMergeResult(state State, prior *State, reason SelectionReason) MergeResult {
	result := MergeResult{State: state, Tools: make([]ToolIdentity, 0, len(state.Tools))}
	result.Receipt = SelectionReceipt{Outcome: OutcomeReset, Reason: reason, Selected: len(state.Tools)}
	previous := make(map[ToolIdentity]bool)
	if reason == ReasonMissing {
		result.Receipt.Outcome = OutcomeSeeded
	}
	if reason == ReasonNone {
		result.Receipt.Outcome = OutcomeUpdated
		for _, tool := range prior.Tools {
			previous[ToolIdentity{tool.Name, tool.DefinitionFingerprint}] = true
		}
		if slices.Equal(prior.Tools, state.Tools) {
			result.Receipt.Outcome = OutcomeReused
		}
	}
	for _, tool := range state.Tools {
		identity := ToolIdentity{tool.Name, tool.DefinitionFingerprint}
		result.Tools = append(result.Tools, identity)
		if previous[identity] {
			result.Receipt.Reused++
		}
		if tool.Pinned {
			result.Receipt.Pinned++
		}
	}
	result.Receipt.Added = result.Receipt.Selected - result.Receipt.Reused
	if prior != nil {
		// Even malformed snapshots cannot inject unbounded receipt counts.
		result.Receipt.Removed = min(len(prior.Tools), configMaxRetainedTools) - result.Receipt.Reused
	}
	return result
}
