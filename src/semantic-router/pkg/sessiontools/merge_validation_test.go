package sessiontools

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestMergeMaximumToolCount(t *testing.T) {
	input := selectionFixture()
	input.Bounds.MaxTools, input.Bounds.MaxStateBytes = 128, 65536
	input.Eligible, input.Ranked = nil, nil
	for i := 0; i < 129; i++ {
		identity := ToolIdentity{Name: fmt.Sprintf("tool-%03d", i), DefinitionFingerprint: "fingerprint"}
		input.Eligible = append(input.Eligible, identity)
		input.Ranked = append(input.Ranked, RankedTool{ToolIdentity: identity, Score: float64(129 - i)})
	}
	result := mergeOK(t, input)
	require.Len(t, result.Tools, 128)
	require.Equal(t, "tool-127", result.Tools[127].Name)
}

func TestMergeInvalidation(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*State)
		reason SelectionReason
	}{
		{"schema", func(s *State) { s.SchemaVersion-- }, ReasonInvalidState},
		{"duplicate", func(s *State) { s.Tools[1] = s.Tools[0] }, ReasonInvalidState},
		{"metadata", func(s *State) { s.CreatedAt = time.Time{} }, ReasonInvalidState},
		{"unset turn", func(s *State) { s.Turn = 0 }, ReasonInvalidState},
		{"future first seen", func(s *State) { s.Turn, s.Tools[0].FirstSeenTurn = 1, 2 }, ReasonInvalidState},
		{"missing fingerprint", func(s *State) { s.CapabilityFingerprint = "" }, ReasonInvalidState},
		{"malformed tool fingerprint", func(s *State) { s.Tools[0].DefinitionFingerprint = " fp " }, ReasonInvalidState},
		{"noncanonical name", func(s *State) { s.Tools[0].Name = " a " }, ReasonInvalidState},
		{"unset first seen", func(s *State) { s.Tools[0].FirstSeenTurn = 0 }, ReasonInvalidState},
		{"policy", func(s *State) { s.PolicyFingerprint += "-changed" }, ReasonPolicyChanged},
		{"catalog", func(s *State) { s.CatalogFingerprint += "-changed" }, ReasonCatalogChanged},
		{"capability", func(s *State) { s.CapabilityFingerprint += "-changed" }, ReasonCapabilityChanged},
		{"expired", func(s *State) { s.ExpiresAt = s.LastSeenAt }, ReasonExpired},
		{"clock rollback", func(s *State) { s.LastSeenAt = s.LastSeenAt.Add(time.Second) }, ReasonClockRegressed},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			input := selectionFixture()
			prior := mergeOK(t, input).State
			test.mutate(&prior)
			input.Prior, input.Ranked = &prior, input.Ranked[4:]
			before := prior.Clone()
			result := mergeOK(t, input)
			require.Equal(t, test.reason, result.Receipt.Reason)
			require.Equal(t, []string{"e", "f"}, selectedNames(result))
			require.Equal(t, before, prior)
		})
	}
}

func TestMergeRequiredTools(t *testing.T) {
	input := selectionFixture()
	input.Required = []string{"f"}
	result := mergeOK(t, input)
	require.Equal(t, []string{"f", "a", "b", "c"}, selectedNames(result))
	for _, unavailable := range []bool{false, true} {
		input.Required = []string{"a", "b", "c", "d", "e"}
		if unavailable {
			input.Required = []string{"forbidden"}
		}
		result, err := Merge(input)
		var typed *RequiredToolsError
		require.ErrorAs(t, err, &typed)
		require.Equal(t, unavailable, typed.Unavailable)
		require.Empty(t, result.Tools)
	}
}

func TestMergeByteBudgetIncludesMetadata(t *testing.T) {
	input := selectionFixture()
	input.Ranked = input.Ranked[:1]
	first := mergeOK(t, input)
	input.Bounds.MaxStateBytes = plannedStateSize(first.State)
	require.Equal(t, first.Tools, mergeOK(t, input).Tools, "exact reserved boundary fits")
	input.Bounds.MaxStateBytes--
	require.Empty(t, mergeOK(t, input).Tools)
	input.Required = []string{"a"}
	first.State.StrategyID = "" // Requirements do not contribute a ranking strategy.
	input.Bounds.MaxStateBytes = plannedStateSize(first.State) - 1
	_, err := Merge(input)
	var typed *RequiredToolsError
	require.ErrorAs(t, err, &typed)
	input.Required = nil
	input.Bounds.MaxStateBytes = 1
	_, err = Merge(input)
	require.ErrorIs(t, err, ErrStateTooLarge)
}

func TestMergeByteCapacityPinResetAndStrategyGrowth(t *testing.T) {
	input := selectionFixture()
	input.Ranked = input.Ranked[:1]
	first := mergeOK(t, input)
	first.State.Tools[0].Pinned = true
	input.Bounds.MaxStateBytes = plannedStateSize(first.State) + 1
	input.Prior, input.Called = &first.State, []string{"b"}
	input.Ranked = selectionFixture().Ranked[1:]
	result := mergeOK(t, input)
	require.Equal(t, ReasonPinCapacity, result.Receipt.Reason)
	require.Equal(t, []string{"b"}, selectedNames(result))

	input.Called, input.Prior = nil, &result.State
	input.StrategyID = strings.Repeat("x", 128)
	result = mergeOK(t, input)
	require.Equal(t, []string{"b"}, selectedNames(result))
	require.Equal(t, "semantic", result.State.StrategyID, "a rejected growth proposal cannot change strategy metadata")
}

func TestMergeTurnExhaustionIsIndependentOfRevision(t *testing.T) {
	input := selectionFixture()
	prior := mergeOK(t, input).State
	prior.Revision, prior.Turn = math.MaxUint64, MaxSelectionTurn-1
	input.Prior = &prior
	last := mergeOK(t, input)
	require.Equal(t, uint64(MaxSelectionTurn), last.State.Turn)
	input.Prior = &last.State
	_, err := Merge(input)
	require.ErrorIs(t, err, ErrTurnExhausted)
	input.Prior.Turn++
	require.Error(t, input.Prior.Validate(4, 16384))
}

func TestMergeRejectsInvalidEvidence(t *testing.T) {
	tests := map[string]func(*MergeInput){
		"duplicate eligible": func(i *MergeInput) { i.Eligible[1].Name = " a " },
		"duplicate ranking":  func(i *MergeInput) { i.Ranked[1].Name = " a " },
		"nan":                func(i *MergeInput) { i.Ranked[0].Score = math.NaN() },
		"infinity":           func(i *MergeInput) { i.Ranked[0].Score = math.Inf(1) },
		"name":               func(i *MergeInput) { i.Called = []string{" "} },
		"fingerprint":        func(i *MergeInput) { i.Fingerprints.Catalog = "" },
		"count":              func(i *MergeInput) { i.Bounds.MaxTools = 0 },
		"count ceiling":      func(i *MergeInput) { i.Bounds.MaxTools = 129 },
		"negative growth":    func(i *MergeInput) { i.Bounds.MaxNewToolsPerTurn = -1 },
		"byte ceiling":       func(i *MergeInput) { i.Bounds.MaxStateBytes = 65537 },
		"input ceiling":      func(i *MergeInput) { i.Called = make([]string, MaxSelectionInputs+1) },
		"clock":              func(i *MergeInput) { i.Now = time.Time{} },
		"ttl":                func(i *MergeInput) { i.TTL = 0 },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			input := selectionFixture()
			mutate(&input)
			_, err := Merge(input)
			require.ErrorIs(t, err, ErrInvalidSelection)
		})
	}
}
