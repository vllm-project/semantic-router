package sessiontools

import (
	"reflect"
	"testing"
	"time"
)

func mergeCandidate(name, fingerprint string) ToolCandidate {
	return ToolCandidate{Name: name, DefinitionFingerprint: fingerprint}
}

func mergeState(names ...string) State {
	tools := make([]ToolState, 0, len(names))
	for index, name := range names {
		tools = append(tools, ToolState{
			Name:                  name,
			DefinitionFingerprint: "fp-" + name,
			FirstSeenTurn:         index,
		})
	}
	return State{
		SchemaVersion:         SchemaVersion,
		Revision:              4,
		PolicyFingerprint:     "policy",
		CatalogFingerprint:    "catalog",
		CapabilityFingerprint: "capability",
		Tools:                 tools,
		CreatedAt:             time.Unix(1, 0),
		LastSeenAt:            time.Unix(2, 0),
		ExpiresAt:             time.Unix(3, 0),
	}
}

func mergeToolNames(tools []ToolState) []string {
	names := make([]string, len(tools))
	for index, tool := range tools {
		names[index] = tool.Name
	}
	return names
}

func mergeAuthorized(names ...string) []ToolCandidate {
	result := make([]ToolCandidate, 0, len(names))
	for _, name := range names {
		result = append(result, mergeCandidate(name, "fp-"+name))
	}
	return result
}

func mergeSelected(names ...string) []ToolCandidate {
	return mergeAuthorized(names...)
}

func TestMergeToolSet_InitialSeedUsesStatelessSelectionOrder(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Authorized:         mergeAuthorized("search", "lookup", "math", "mail"),
		Selected:           mergeSelected("math", "search", "mail"),
		Turn:               0,
		MaxTools:           4,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"math", "search", "mail"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("initial order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.Added != 3 {
		t.Fatalf("initial added = %d, want 3", got.Receipt.Added)
	}
	if got.State.Tools[0].FirstSeenTurn != 0 {
		t.Fatalf("first-seen turn = %d, want 0", got.State.Tools[0].FirstSeenTurn)
	}
}

func TestMergeToolSet_FreshSeedAfterHistoricalInvalidationUsesFullSelection(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Previous: State{Tools: []ToolState{
			{Name: "revoked", DefinitionFingerprint: "fp-revoked", FirstSeenTurn: 1},
		}},
		Authorized:         mergeAuthorized("called", "math", "mail"),
		Selected:           mergeSelected("math", "mail"),
		CalledToolNames:    []string{"called"},
		Turn:               4,
		MaxTools:           3,
		MaxNewToolsPerTurn: 1,
		PinCalledTools:     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"called", "math", "mail"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("fresh-seed order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.Dropped != 1 || got.Receipt.Added != 3 {
		t.Fatalf("fresh-seed receipt = %+v, want dropped=1 added=3", got.Receipt)
	}
}

func TestMergeToolSet_RetainsAuthorizedHistoryAndBoundsGrowth(t *testing.T) {
	previous := mergeState("search", "lookup")
	got, err := MergeToolSet(MergeInput{
		Previous:           previous,
		Authorized:         mergeAuthorized("search", "lookup", "math", "mail"),
		Selected:           mergeSelected("mail", "math"),
		Turn:               2,
		MaxTools:           4,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup", "mail"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("merged order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.Reused != 2 || got.Receipt.Added != 1 || got.Receipt.SkippedGrowth != 1 {
		t.Fatalf("unexpected receipt: %+v", got.Receipt)
	}
	// Reordering the current request-time selection must not reorder retained
	// definitions on the next deterministic merge.
	second, err := MergeToolSet(MergeInput{
		Previous:           got.State,
		Authorized:         mergeAuthorized("search", "lookup", "math", "mail"),
		Selected:           mergeSelected("math", "mail"),
		Turn:               3,
		MaxTools:           4,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup", "mail", "math"}; !reflect.DeepEqual(mergeToolNames(second.State.Tools), want) {
		t.Fatalf("second merged order = %v, want %v", mergeToolNames(second.State.Tools), want)
	}
}

func TestMergeToolSet_ZeroGrowthStillReusesHistory(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Previous:           mergeState("search", "lookup"),
		Authorized:         mergeAuthorized("search", "lookup", "math"),
		Selected:           mergeSelected("math"),
		Turn:               2,
		MaxTools:           4,
		MaxNewToolsPerTurn: 0,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("zero-growth order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.SkippedGrowth != 1 {
		t.Fatalf("skipped growth = %d, want 1", got.Receipt.SkippedGrowth)
	}
}

func TestMergeToolSet_CalledToolsPinAndBypassGrowthQuota(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Previous:           mergeState("search"),
		Authorized:         mergeAuthorized("search", "lookup", "math"),
		Selected:           mergeSelected("math"),
		CalledToolNames:    []string{"lookup", "lookup", "not-authorized"},
		Turn:               3,
		MaxTools:           3,
		MaxNewToolsPerTurn: 0,
		PinCalledTools:     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("called-tool order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if !got.State.Tools[1].Pinned {
		t.Fatal("newly called tool must be pinned")
	}
	if got.Receipt.Added != 1 || got.Receipt.Pinned != 1 || got.Receipt.SkippedGrowth != 1 {
		t.Fatalf("unexpected called-tool receipt: %+v", got.Receipt)
	}
}

func TestMergeToolSet_FullSetReplacesOldestUnpinnedForCalledTool(t *testing.T) {
	previous := mergeState("old", "middle", "pinned")
	previous.Tools[2].Pinned = true
	got, err := MergeToolSet(MergeInput{
		Previous:        previous,
		Authorized:      mergeAuthorized("old", "middle", "pinned", "new"),
		CalledToolNames: []string{"new"},
		Turn:            5,
		MaxTools:        3,
		PinCalledTools:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"middle", "pinned", "new"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("replacement order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if !got.State.Tools[2].Pinned {
		t.Fatal("replacement called tool must be pinned")
	}
	if got.Receipt.Replaced != 1 || got.Receipt.Evicted != 1 {
		t.Fatalf("replacement receipt = %+v", got.Receipt)
	}
}

func TestMergeToolSet_FullSetReplacesOldestUnpinnedForSelectedTool(t *testing.T) {
	previous := mergeState("old", "middle", "pinned")
	previous.Tools[2].Pinned = true
	got, err := MergeToolSet(MergeInput{
		Previous:           previous,
		Authorized:         mergeAuthorized("old", "middle", "pinned", "new"),
		Selected:           mergeSelected("new"),
		Turn:               5,
		MaxTools:           3,
		MaxNewToolsPerTurn: 1,
		PinCalledTools:     true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"middle", "pinned", "new"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("selected replacement order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.State.Tools[2].Pinned {
		t.Fatal("selected replacement must not be pinned without a called tool")
	}
	if got.Receipt.Added != 1 || got.Receipt.Replaced != 1 || got.Receipt.Evicted != 1 {
		t.Fatalf("selected replacement receipt = %+v", got.Receipt)
	}
}

func TestMergeToolSet_DoesNotEvictPinnedForAnotherCalledTool(t *testing.T) {
	previous := mergeState("search", "lookup")
	previous.Tools[0].Pinned = true
	previous.Tools[1].Pinned = true
	got, err := MergeToolSet(MergeInput{
		Previous:        previous,
		Authorized:      mergeAuthorized("search", "lookup", "new"),
		CalledToolNames: []string{"new"},
		Turn:            5,
		MaxTools:        2,
		PinCalledTools:  true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("pinned overflow order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.PinnedOverflow != 1 {
		t.Fatalf("pinned overflow = %d, want 1", got.Receipt.PinnedOverflow)
	}
}

func TestMergeToolSet_DefinitionChangeMustBeReadmittedAsNew(t *testing.T) {
	previous := mergeState("search", "lookup")
	authorized := []ToolCandidate{
		mergeCandidate("search", "fp-search-v2"),
		mergeCandidate("lookup", "fp-lookup"),
	}
	got, err := MergeToolSet(MergeInput{
		Previous:           previous,
		Authorized:         authorized,
		Selected:           []ToolCandidate{mergeCandidate("search", "fp-search-v2")},
		Turn:               4,
		MaxTools:           2,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"lookup", "search"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("definition-change order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.State.Tools[1].DefinitionFingerprint != "fp-search-v2" {
		t.Fatalf("definition fingerprint = %q, want fp-search-v2", got.State.Tools[1].DefinitionFingerprint)
	}
	if got.State.Tools[1].FirstSeenTurn != 4 {
		t.Fatalf("first-seen turn = %d, want 4", got.State.Tools[1].FirstSeenTurn)
	}
	if got.Receipt.Dropped != 1 || got.Receipt.Added != 1 || got.Receipt.Replaced != 0 {
		t.Fatalf("definition-change receipt = %+v", got.Receipt)
	}
}

func TestMergeToolSet_PinDisabledClearsPinsAndIgnoresCalls(t *testing.T) {
	previous := mergeState("search")
	previous.Tools[0].Pinned = true
	got, err := MergeToolSet(MergeInput{
		Previous:           previous,
		Authorized:         mergeAuthorized("search", "lookup"),
		CalledToolNames:    []string{"lookup"},
		Turn:               2,
		MaxTools:           2,
		MaxNewToolsPerTurn: 0,
		PinCalledTools:     false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("pin-disabled order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.State.Tools[0].Pinned {
		t.Fatal("pin-disabled merge must clear stale pin metadata")
	}
}

func TestMergeToolSet_DropsUnauthorizedHistoryAndIgnoresUnauthorizedSelection(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Previous: State{Tools: []ToolState{
			{Name: "allowed", DefinitionFingerprint: "fp-allowed", FirstSeenTurn: 0},
			{Name: "revoked", DefinitionFingerprint: "fp-revoked", FirstSeenTurn: 1},
		}},
		Authorized:         mergeAuthorized("allowed", "new"),
		Selected:           mergeSelected("revoked", "new"),
		Turn:               2,
		MaxTools:           3,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"allowed", "new"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("authorization-filtered order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
	if got.Receipt.Dropped != 1 {
		t.Fatalf("dropped = %d, want 1", got.Receipt.Dropped)
	}
}

func TestMergeToolSet_SelectedIsAdvisoryAndFirstValidDuplicateWins(t *testing.T) {
	got, err := MergeToolSet(MergeInput{
		Authorized: mergeAuthorized("search", "lookup"),
		Selected: []ToolCandidate{
			{},
			mergeCandidate("not-authorized", "fp-not-authorized"),
			mergeCandidate("search", "wrong-fingerprint"),
			mergeCandidate("search", "fp-search"),
			mergeCandidate("lookup", "fp-lookup"),
			mergeCandidate("search", "fp-search"),
		},
		Turn:               0,
		MaxTools:           3,
		MaxNewToolsPerTurn: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []string{"search", "lookup"}; !reflect.DeepEqual(mergeToolNames(got.State.Tools), want) {
		t.Fatalf("advisory selected order = %v, want %v", mergeToolNames(got.State.Tools), want)
	}
}

func TestMergeToolSet_RejectsMalformedInputs(t *testing.T) {
	cases := []struct {
		name  string
		input MergeInput
	}{
		{
			name:  "zero max tools",
			input: MergeInput{MaxTools: 0},
		},
		{
			name:  "negative turn",
			input: MergeInput{Turn: -1, MaxTools: 1},
		},
		{
			name: "duplicate authorized names",
			input: MergeInput{
				Authorized: []ToolCandidate{mergeCandidate("same", "one"), mergeCandidate("same", "two")},
				MaxTools:   1,
			},
		},
		{
			name: "duplicate previous names",
			input: MergeInput{
				Previous: State{Tools: []ToolState{
					{Name: "same", DefinitionFingerprint: "one"},
					{Name: "same", DefinitionFingerprint: "two"},
				}},
				MaxTools: 1,
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := MergeToolSet(tc.input); err == nil {
				t.Fatal("expected malformed merge input to fail")
			}
		})
	}
}

func TestMergeToolSet_DoesNotMutateInputAndIsDeterministic(t *testing.T) {
	previous := mergeState("search", "lookup")
	original := previous.Clone()
	input := MergeInput{
		Previous:           previous,
		Authorized:         mergeAuthorized("search", "lookup", "math"),
		Selected:           mergeSelected("math"),
		CalledToolNames:    []string{"lookup"},
		Turn:               3,
		MaxTools:           4,
		MaxNewToolsPerTurn: 1,
		PinCalledTools:     true,
	}
	first, err := MergeToolSet(input)
	if err != nil {
		t.Fatal(err)
	}
	second, err := MergeToolSet(input)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("repeated merge differs:\nfirst=%+v\nsecond=%+v", first, second)
	}
	if !reflect.DeepEqual(previous, original) {
		t.Fatalf("merge mutated previous state: got=%+v want=%+v", previous, original)
	}
}
