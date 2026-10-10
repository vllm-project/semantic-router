package sessiontools

import (
	"math"
	"slices"
	"time"
)

type mergeBuilder struct {
	selection   preparedSelection
	state       State
	pinOverflow bool
}

func (b *mergeBuilder) fill(seed, admitPins bool) error {
	empty := b.state
	empty.Tools = nil
	if !b.fits(empty) {
		return ErrStateTooLarge
	}
	if admitPins && b.pinCount() > b.selection.input.Bounds.MaxTools {
		b.pinOverflow = true
		return nil
	}
	for !b.fits(b.state) {
		index := b.victim(b.state.Tools)
		if index < 0 {
			b.pinOverflow = true
			return nil
		}
		b.state.Tools = slices.Delete(b.state.Tools, index, index+1)
	}
	for _, name := range b.selection.input.Required {
		if !b.admit(name, true) {
			return &RequiredToolsError{}
		}
	}
	if admitPins {
		for _, name := range b.selection.input.Called {
			if b.selection.called[name] && !b.admit(name, true) {
				b.pinOverflow = true
				return nil
			}
		}
	}
	b.addRanked(seed)
	return b.state.Validate(b.selection.input.Bounds.MaxTools, b.selection.input.Bounds.MaxStateBytes)
}

func (b *mergeBuilder) pinCount() int {
	pins := make(map[string]bool, len(b.state.Tools)+len(b.selection.called))
	for _, tool := range b.state.Tools {
		if tool.Pinned {
			pins[tool.Name] = true
		}
	}
	for name := range b.selection.called {
		pins[name] = true
	}
	return len(pins)
}

func (b *mergeBuilder) addRanked(seed bool) {
	budget := b.selection.input.Bounds.MaxNewToolsPerTurn
	if seed {
		budget = b.selection.input.Bounds.MaxTools
	}
	for _, candidate := range b.selection.input.Ranked {
		if budget == 0 || len(b.state.Tools) == b.selection.input.Bounds.MaxTools {
			break
		}
		if b.contains(candidate.Name) {
			continue
		}
		if b.admit(candidate.Name, false) {
			budget--
			b.state.StrategyID = b.selection.input.StrategyID
		}
	}
}

func (b *mergeBuilder) contains(name string) bool {
	for _, tool := range b.state.Tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

// admit changes state only when the complete trial fits. Required/pinned
// additions may evict the newest unprotected entry; relevance cannot evict.
func (b *mergeBuilder) admit(name string, replace bool) bool {
	if b.contains(name) {
		return true
	}
	identity := b.selection.eligible[name]
	trial := b.state.Clone()
	if trial.Turn > MaxSelectionTurn {
		return false
	}
	tool := ToolState{
		Name: name, DefinitionFingerprint: identity.DefinitionFingerprint,
		Pinned: b.selection.called[name], FirstSeenTurn: int(trial.Turn),
	}
	for {
		candidate := trial.Clone()
		candidate.Tools = append(candidate.Tools, tool)
		if !replace {
			candidate.StrategyID = b.selection.input.StrategyID
		}
		if b.fits(candidate) {
			b.state = candidate
			return true
		}
		index := b.victim(trial.Tools)
		if !replace || index < 0 {
			return false
		}
		trial.Tools = slices.Delete(trial.Tools, index, index+1)
	}
}

// Equal first-seen turns are broken by later retained position, preserving
// the longest possible old prefix. Current requirements are never victims.
func (b *mergeBuilder) victim(tools []ToolState) int {
	victim := -1
	for i, tool := range tools {
		if tool.Pinned || b.selection.required[tool.Name] {
			continue
		}
		if victim < 0 || tool.FirstSeenTurn >= tools[victim].FirstSeenTurn {
			victim = i
		}
	}
	return victim
}

func (b *mergeBuilder) fits(state State) bool {
	if len(state.Tools) > b.selection.input.Bounds.MaxTools {
		return false
	}
	return plannedStateSize(state) <= b.selection.input.Bounds.MaxStateBytes
}

// plannedStateSize reserves the maximum JSON width of the store-assigned
// uint64 revision and UTC timestamps. This conservative overhead prevents a
// fitting proposal from overflowing merely because the store stamps a larger
// token or fractional second. The Store also checks the actual encoded size.
func plannedStateSize(state State) int {
	state.Revision = math.MaxUint64
	state.CreatedAt = fullPrecisionTime(state.CreatedAt)
	state.LastSeenAt = fullPrecisionTime(state.LastSeenAt)
	state.ExpiresAt = fullPrecisionTime(state.ExpiresAt)
	size, err := state.encodedSize()
	if err != nil {
		return math.MaxInt
	}
	return size
}

func fullPrecisionTime(value time.Time) time.Time {
	return value.UTC().Add(time.Duration(999999999 - value.Nanosecond()))
}
