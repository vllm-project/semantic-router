package sessiontools

import "fmt"

// ToolCandidate is the identity-only projection of one tool that is valid for
// the current request. The request-phase adapter supplies the definition
// fingerprint after it has performed the current authorization and capability
// checks; MergeToolSet never treats a stored ToolState as authorization.
type ToolCandidate struct {
	Name                  string
	DefinitionFingerprint string
}

// MergeInput contains the bounded, deterministic inputs to MergeToolSet.
// Authorized must be the current request's re-authorized catalog. Selected is
// the ordinary request-time selection in relevance order. Keeping those two
// views separate lets sticky reuse retain a valid historical tool that was not
// selected by this turn without allowing an old identity to bypass policy.
type MergeInput struct {
	Previous State

	Authorized []ToolCandidate
	// Selected is advisory: entries whose exact identity is absent from
	// Authorized are ignored, and the first occurrence of a duplicate valid
	// identity wins. This preserves request-time relevance order while making
	// malformed selection output fail closed instead of failing the request.
	Selected []ToolCandidate
	// CalledToolNames is the ordered list of tool names observed in the
	// request's conversation history. Empty and duplicate names are ignored;
	// names that are not in Authorized are never admitted.
	CalledToolNames []string

	// Turn is the non-negative request turn index used for newly admitted
	// identities. The caller owns turn-index monotonicity.
	Turn int

	MaxTools           int
	MaxNewToolsPerTurn int
	PinCalledTools     bool
}

// MergeReceipt is content-minimized accounting for one merge. It contains
// counts only; tool names, schemas, prompts, arguments, and results never
// cross the sessiontools boundary.
type MergeReceipt struct {
	Reused          int
	Added           int
	Pinned          int
	Replaced        int
	Evicted         int
	Dropped         int
	SkippedGrowth   int
	PinnedOverflow  int
	PinnedEvictions int
}

// MergeResult contains the next identity-only tool state and its bounded
// accounting receipt. State envelope metadata is copied from MergeInput's
// Previous value; the request manager updates fingerprints, revision, and TTL
// when it commits the result.
type MergeResult struct {
	State   State
	Receipt MergeReceipt
}

// MergeToolSet applies the deterministic sticky-selection rules to one
// request. It is deliberately pure: it does not read a store, compare
// fingerprints, authorize tools, or mutate the input slices.
//
// The rules are intentionally narrow:
//   - valid retained identities keep their previous order;
//   - a definition-mismatched historical identity is dropped rather than
//     refreshed in place;
//   - called tools are pinned (when configured) and may bypass the growth
//     quota, but never the max_tools bound;
//   - on a non-initial turn, ordinary selected additions are appended in the
//     supplied order up to max_new_tools_per_turn; when full they replace the
//     oldest unpinned identity, breaking ties by retained order;
//   - a forced called addition also evicts the oldest unpinned identity when
//     necessary; the replacement is appended so surviving identities never
//     change relative order;
//   - if a bound is already exceeded, unpinned identities are evicted first,
//     and pinned identities only as a last-resort hard-bound safeguard.
//
// A first turn seeds the state from the ordinary selection (up to max_tools),
// rather than applying max_new_tools_per_turn to the initial stateless result.
func MergeToolSet(input MergeInput) (MergeResult, error) {
	if err := validateMergeInput(input); err != nil {
		return MergeResult{}, err
	}

	authorized, err := indexCandidates(input.Authorized)
	if err != nil {
		return MergeResult{}, err
	}
	selected := selectedCandidates(input.Selected, authorized)
	called := calledSet(input.CalledToolNames)

	result := MergeResult{State: input.Previous.Clone()}
	entries := make([]ToolState, 0, minInt(input.MaxTools, len(input.Previous.Tools)+len(selected)))
	retained := make(map[string]int, len(input.Previous.Tools))

	// Revalidate every historical identity against the current authorized
	// catalog. A same-name definition change is not the same identity: discard
	// it here, even though the manager also invalidates state on a whole-catalog
	// fingerprint mismatch. Current selection or call-pinning may admit the new
	// definition below as a newly seen identity.
	for _, previous := range input.Previous.Tools {
		candidate, ok := authorized[previous.Name]
		if !ok || previous.DefinitionFingerprint != candidate.DefinitionFingerprint {
			result.Receipt.Dropped++
			continue
		}
		result.Receipt.Reused++
		if !input.PinCalledTools {
			previous.Pinned = false
		}
		if input.PinCalledTools {
			if _, observed := called[previous.Name]; observed && !previous.Pinned {
				previous.Pinned = true
				result.Receipt.Pinned++
			}
		}
		retained[previous.Name] = len(entries)
		entries = append(entries, previous)
	}

	// A called identity has priority over relevance-driven growth. Pinning is
	// applied before trimming so a called retained tool cannot be evicted merely
	// because an oversized/corrupt input state reached this pure seam.
	trimEntries(&entries, input.MaxTools, &result.Receipt)
	retained = rebuildIndex(entries)
	// If no historical identity survived current authorization/fingerprint
	// checks, seed from the ordinary stateless selection below. Called-tool
	// pinning is applied first, but must not make a fresh state consume the
	// per-turn growth allowance.
	freshSeed := len(entries) == 0

	if input.PinCalledTools {
		for _, name := range orderedCalledNames(input.CalledToolNames, called) {
			candidate, authorizedNow := authorized[name]
			if !authorizedNow {
				continue
			}
			if index, alreadyRetained := retained[name]; alreadyRetained {
				if !entries[index].Pinned {
					entries[index].Pinned = true
					result.Receipt.Pinned++
				}
				continue
			}
			if appendCandidate(&entries, candidate, true, input, &result.Receipt) {
				retained = rebuildIndex(entries)
			}
		}
	}

	selectedLimit := input.MaxNewToolsPerTurn
	if freshSeed {
		selectedLimit = input.MaxTools
	}
	for _, candidate := range selected {
		if _, alreadyRetained := retained[candidate.Name]; alreadyRetained {
			continue
		}
		if selectedLimit <= 0 {
			result.Receipt.SkippedGrowth++
			continue
		}
		if appendCandidate(&entries, candidate, false, input, &result.Receipt) {
			selectedLimit--
			retained = rebuildIndex(entries)
		} else {
			result.Receipt.SkippedGrowth++
		}
	}

	result.State.Tools = entries
	return result, nil
}

func validateMergeInput(input MergeInput) error {
	if input.Turn < 0 {
		return fmt.Errorf("sessiontools: merge turn must not be negative")
	}
	if input.MaxTools <= 0 {
		return fmt.Errorf("sessiontools: merge max_tools must be greater than zero")
	}
	if input.MaxNewToolsPerTurn < 0 || input.MaxNewToolsPerTurn > input.MaxTools {
		return fmt.Errorf("sessiontools: merge max_new_tools_per_turn must be between 0 and max_tools (%d)", input.MaxTools)
	}
	seen := make(map[string]struct{}, len(input.Previous.Tools))
	for i, tool := range input.Previous.Tools {
		if tool.Name == "" {
			return fmt.Errorf("sessiontools: merge previous tool at index %d has an empty name", i)
		}
		if tool.DefinitionFingerprint == "" {
			return fmt.Errorf("sessiontools: merge previous tool %q has an empty definition_fingerprint", tool.Name)
		}
		if tool.FirstSeenTurn < 0 {
			return fmt.Errorf("sessiontools: merge previous tool %q has a negative first_seen_turn", tool.Name)
		}
		if _, duplicate := seen[tool.Name]; duplicate {
			return fmt.Errorf("sessiontools: merge previous tools contain duplicate name %q", tool.Name)
		}
		seen[tool.Name] = struct{}{}
	}
	return validateCandidates(input.Authorized, "authorized")
}

func validateCandidates(candidates []ToolCandidate, label string) error {
	seen := make(map[string]struct{}, len(candidates))
	for i, candidate := range candidates {
		if candidate.Name == "" {
			return fmt.Errorf("sessiontools: merge %s tool at index %d has an empty name", label, i)
		}
		if candidate.DefinitionFingerprint == "" {
			return fmt.Errorf("sessiontools: merge %s tool %q has an empty definition_fingerprint", label, candidate.Name)
		}
		if _, duplicate := seen[candidate.Name]; duplicate {
			return fmt.Errorf("sessiontools: merge %s tools contain duplicate name %q", label, candidate.Name)
		}
		seen[candidate.Name] = struct{}{}
	}
	return nil
}

func indexCandidates(candidates []ToolCandidate) (map[string]ToolCandidate, error) {
	if err := validateCandidates(candidates, "authorized"); err != nil {
		return nil, err
	}
	indexed := make(map[string]ToolCandidate, len(candidates))
	for _, candidate := range candidates {
		indexed[candidate.Name] = candidate
	}
	return indexed, nil
}

func selectedCandidates(candidates []ToolCandidate, authorized map[string]ToolCandidate) []ToolCandidate {
	selected := make([]ToolCandidate, 0, len(candidates))
	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		current, ok := authorized[candidate.Name]
		if !ok || current.DefinitionFingerprint != candidate.DefinitionFingerprint {
			continue
		}
		if _, duplicate := seen[candidate.Name]; duplicate {
			continue
		}
		seen[candidate.Name] = struct{}{}
		selected = append(selected, current)
	}
	return selected
}

func calledSet(names []string) map[string]struct{} {
	called := make(map[string]struct{}, len(names))
	for _, name := range names {
		if name != "" {
			called[name] = struct{}{}
		}
	}
	return called
}

func orderedCalledNames(names []string, called map[string]struct{}) []string {
	ordered := make([]string, 0, len(called))
	seen := make(map[string]struct{}, len(called))
	for _, name := range names {
		if _, exists := called[name]; !exists {
			continue
		}
		if _, duplicate := seen[name]; duplicate {
			continue
		}
		seen[name] = struct{}{}
		ordered = append(ordered, name)
	}
	return ordered
}

func appendCandidate(entries *[]ToolState, candidate ToolCandidate, forced bool, input MergeInput, receipt *MergeReceipt) bool {
	if len(*entries) >= input.MaxTools {
		victim := unpinnedEvictionIndex(*entries)
		if victim < 0 {
			if forced {
				receipt.PinnedOverflow++
			}
			return false
		}
		*entries = append((*entries)[:victim], (*entries)[victim+1:]...)
		receipt.Evicted++
		receipt.Replaced++
	}

	*entries = append(*entries, ToolState{
		Name:                  candidate.Name,
		DefinitionFingerprint: candidate.DefinitionFingerprint,
		Pinned:                forced && input.PinCalledTools,
		FirstSeenTurn:         input.Turn,
	})
	receipt.Added++
	if forced && input.PinCalledTools {
		receipt.Pinned++
	}
	return true
}

func trimEntries(entries *[]ToolState, maxTools int, receipt *MergeReceipt) {
	for len(*entries) > maxTools {
		victim := evictionIndex(*entries)
		if victim < 0 {
			return
		}
		if (*entries)[victim].Pinned {
			receipt.PinnedEvictions++
		}
		*entries = append((*entries)[:victim], (*entries)[victim+1:]...)
		receipt.Evicted++
	}
}

func evictionIndex(entries []ToolState) int {
	if victim := unpinnedEvictionIndex(entries); victim >= 0 {
		return victim
	}
	// A hard max_tools bound wins over pinning when an invalid caller supplies
	// more pinned entries than the bound. The first-seen tie-break remains
	// deterministic, and this path is accounted for separately.
	victim := -1
	for index, entry := range entries {
		if victim < 0 || entry.FirstSeenTurn < entries[victim].FirstSeenTurn {
			victim = index
		}
	}
	return victim
}

func unpinnedEvictionIndex(entries []ToolState) int {
	victim := -1
	for index, entry := range entries {
		if entry.Pinned {
			continue
		}
		if victim < 0 || entry.FirstSeenTurn < entries[victim].FirstSeenTurn {
			victim = index
		}
	}
	return victim
}

func rebuildIndex(entries []ToolState) map[string]int {
	indexed := make(map[string]int, len(entries))
	for index, entry := range entries {
		indexed[entry.Name] = index
	}
	return indexed
}

func minInt(left, right int) int {
	if left < right {
		return left
	}
	return right
}
