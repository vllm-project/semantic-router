package sessiontools

import (
	"slices"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func selectionFixture() MergeInput {
	input := MergeInput{
		SelectionInput: SelectionInput{
			Bounds:       SelectionBounds{MaxTools: 4, MaxNewToolsPerTurn: 1, PinCalledTools: true, MaxStateBytes: 16384},
			Fingerprints: Fingerprints{Policy: "policy", Catalog: "catalog", Capability: "capability"},
			StrategyID:   "semantic",
		},
		Now: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC), TTL: time.Hour,
	}
	for i, name := range []string{"a", "b", "c", "d", "e", "f"} {
		identity := ToolIdentity{Name: name, DefinitionFingerprint: "fp-" + name}
		input.Eligible = append(input.Eligible, identity)
		input.Ranked = append(input.Ranked, RankedTool{ToolIdentity: identity, Score: float64(6 - i)})
	}
	return input
}

func mergeOK(t *testing.T, input MergeInput) MergeResult {
	t.Helper()
	result, err := Merge(input)
	require.NoError(t, err)
	require.NoError(t, result.State.Validate(input.Bounds.MaxTools, input.Bounds.MaxStateBytes))
	require.Zero(t, result.State.Revision, "the planner must not invent a store token")
	return result
}

func selectedNames(result MergeResult) []string {
	names := make([]string, 0, len(result.Tools))
	for _, tool := range result.Tools {
		names = append(names, tool.Name)
	}
	return names
}

func TestMergeSeedAndCompatibleGrowth(t *testing.T) {
	input := selectionFixture()
	input.Ranked = input.Ranked[:2]
	first := mergeOK(t, input)
	require.Equal(t, []string{"a", "b"}, selectedNames(first))
	require.Equal(t, OutcomeSeeded, first.Receipt.Outcome)
	first.State.Revision = 1000000 // An opaque token, unrelated to logical turns.
	input.Prior = &first.State
	input.Now = input.Now.Add(time.Second)
	input.Ranked = selectionFixture().Ranked
	input.Called = []string{" e ", "a", "e", "not-eligible"}
	before := first.State.Clone()
	second := mergeOK(t, input)
	require.Equal(t, []string{"a", "b", "e", "c"}, selectedNames(second))
	require.Equal(t, uint64(2), second.State.Turn)
	require.Equal(t, 2, second.State.Tools[2].FirstSeenTurn)
	require.Equal(t, 2, second.Receipt.Pinned)
	require.Equal(t, 2, second.Receipt.Reused)
	require.Equal(t, before, first.State, "planning must not mutate its snapshot")

	input.Prior = &second.State
	input.Ranked = []RankedTool{selectionFixture().Ranked[5]}
	third := mergeOK(t, input)
	require.Equal(t, selectedNames(second), selectedNames(third), "ordinary relevance cannot replace a full set")
	require.Equal(t, OutcomeReused, third.Receipt.Outcome)
	require.Equal(t, second.State.StrategyID, third.State.StrategyID)
}

func TestMergeZeroGrowthAndPinningDisabled(t *testing.T) {
	input := selectionFixture()
	input.Bounds.MaxNewToolsPerTurn = 0
	input.Ranked = input.Ranked[:1]
	first := mergeOK(t, input) // Zero growth still allows cold seeding.
	require.Equal(t, []string{"a"}, selectedNames(first))
	input.Prior = &first.State
	input.Ranked = selectionFixture().Ranked
	input.Called = []string{"d"}
	second := mergeOK(t, input)
	require.Equal(t, []string{"a", "d"}, selectedNames(second))
	require.True(t, second.State.Tools[1].Pinned)
	input.Bounds.PinCalledTools = false
	input.Prior = &second.State
	input.Called = []string{"b"}
	third := mergeOK(t, input)
	require.Equal(t, []string{"a", "d"}, selectedNames(third))
	require.Zero(t, third.Receipt.Pinned)
}

func TestMergePinReplacementPreservesSurvivors(t *testing.T) {
	input := selectionFixture()
	first := mergeOK(t, input)
	// b is newer than c/d despite appearing before them. Equal age removes
	// the later retained position; no sort may disturb the surviving prefix.
	first.State.Turn = 4
	first.State.Tools[1].FirstSeenTurn = 4
	first.State.Tools[2].FirstSeenTurn = 4
	first.State.Tools[2].Pinned = true
	input.Prior, input.Called = &first.State, []string{"e"}
	second := mergeOK(t, input)
	require.Equal(t, []string{"a", "c", "d", "e"}, selectedNames(second))
	input.Prior, input.Called = &second.State, []string{"f"}
	third := mergeOK(t, input)
	require.Equal(t, []string{"a", "c", "e", "f"}, selectedNames(third))
}

func TestMergePinOverflowUsesFreshBoundedSelection(t *testing.T) {
	input := selectionFixture()
	input.Bounds.MaxTools = 2
	input.Called = []string{"a", "b"}
	first := mergeOK(t, input)
	input.Prior, input.Called = &first.State, []string{"c"}
	input.Ranked = []RankedTool{input.Ranked[4], input.Ranked[3], input.Ranked[2]}
	second := mergeOK(t, input)
	require.Equal(t, []string{"c", "d"}, selectedNames(second))
	require.Equal(t, ReasonPinCapacity, second.Receipt.Reason)
	require.Equal(t, OutcomeReset, second.Receipt.Outcome)
	require.Equal(t, uint64(1), second.State.Turn)
}

func TestMergePermutationsAndEqualScores(t *testing.T) {
	input := selectionFixture()
	input.Called, input.Required = []string{"f", "e", "e"}, []string{"d", "c"}
	for i := range input.Ranked {
		input.Ranked[i].Score = 1
	}
	expected := mergeOK(t, input)
	require.Equal(t, []string{"c", "d", "e", "f"}, selectedNames(expected))
	// Enumerate all rotations and reversals deterministically, including the
	// name tie-break before truncation in a separate ranking-only pass.
	for _, observations := range []bool{true, false} {
		if !observations {
			input.Called, input.Required = nil, nil
			expected = mergeOK(t, input)
			require.Equal(t, []string{"a", "b", "c", "d"}, selectedNames(expected))
		}
		for shift := 0; shift < len(input.Eligible); shift++ {
			input.Eligible = append(input.Eligible[1:], input.Eligible[0])
			input.Ranked = append(input.Ranked[1:], input.Ranked[0])
			slices.Reverse(input.Called)
			slices.Reverse(input.Required)
			require.Equal(t, expected, mergeOK(t, input))
		}
	}
}

func TestMergeEligibilityCannotBeExpandedByStateOrObservations(t *testing.T) {
	input := selectionFixture()
	first := mergeOK(t, input)
	input.Prior = &first.State
	input.Eligible = input.Eligible[1:] // a is no longer authorized.
	input.Eligible[0].DefinitionFingerprint = "new-b"
	input.Called = []string{"a", "b", "missing"}
	result := mergeOK(t, input)
	require.Equal(t, []string{"c", "d", "b", "e"}, selectedNames(result))
	require.Equal(t, "new-b", result.Tools[2].DefinitionFingerprint)
	require.NotContains(t, selectedNames(result), "a")
	result.Tools[0].Name = "mutated-output"
	result.State.Tools[0].Name = "mutated-state"
	require.Equal(t, "a", first.State.Tools[0].Name)
	require.Equal(t, "b", input.Eligible[0].Name)
}
