package memory

import "testing"

func TestQdrantRetrieveQueryPolicies(t *testing.T) {
	limit, threshold := qdrantRetrieveQuery(5, 0.7, RetrieveOptions{})
	if limit != 5 || threshold == nil || *threshold != 0.7 {
		t.Fatalf("vector-only query = (%d, %v), want limit 5 and threshold 0.7", limit, threshold)
	}

	limit, threshold = qdrantRetrieveQuery(5, 0.7, RetrieveOptions{HybridSearch: true})
	// retrieveSearchTopK(5, true) == 40
	if limit != 40 || threshold != nil {
		t.Fatalf("hybrid query = (%d, %v), want widened window and no server threshold", limit, threshold)
	}

	limit, threshold = qdrantRetrieveQuery(5, 0.7, RetrieveOptions{AdaptiveThreshold: true})
	// retrieveSearchTopK(5, false) == 20
	if limit != 20 || threshold != nil {
		t.Fatalf("adaptive query = (%d, %v), want widened window and no server threshold", limit, threshold)
	}
}

func TestFinalizePolicyRetrieveAdaptiveElbowUsesBelowFloorNeighbor(t *testing.T) {
	// Floor 0.70. The largest gap is .71 → .60, so the elbow stays at the floor
	// and the first three survive. Dropping .60 before the elbow (a server-side
	// threshold) makes .90 → .80 the largest gap and keeps only .90.
	candidates := []*RetrieveResult{
		{Memory: &Memory{ID: "a"}, Score: 0.90},
		{Memory: &Memory{ID: "b"}, Score: 0.80},
		{Memory: &Memory{ID: "c"}, Score: 0.71},
		{Memory: &Memory{ID: "d"}, Score: 0.60},
	}
	got := finalizePolicyRetrieve(candidates, RetrieveOptions{AdaptiveThreshold: true}, 0.70, 10)
	if len(got) != 3 || got[0].Memory.ID != "a" || got[1].Memory.ID != "b" || got[2].Memory.ID != "c" {
		ids := make([]string, len(got))
		for i, result := range got {
			ids[i] = result.Memory.ID
		}
		t.Fatalf("kept %v, want [a b c]", ids)
	}
}

func TestFinalizePolicyRetrieveAdaptiveDropsTail(t *testing.T) {
	candidates := []*RetrieveResult{
		{Memory: &Memory{ID: "a", Content: "alpha"}, Score: 0.90},
		{Memory: &Memory{ID: "b", Content: "beta"}, Score: 0.85},
		{Memory: &Memory{ID: "c", Content: "gamma"}, Score: 0.20},
		{Memory: &Memory{ID: "d", Content: "delta"}, Score: 0.10},
	}
	got := finalizePolicyRetrieve(candidates, RetrieveOptions{AdaptiveThreshold: true}, 0.05, 10)
	if len(got) != 2 {
		t.Fatalf("kept %d memories, want 2 above the score elbow", len(got))
	}
	if got[0].Memory.ID != "a" || got[1].Memory.ID != "b" {
		t.Fatalf("kept ids %s, %s", got[0].Memory.ID, got[1].Memory.ID)
	}
}

func TestFinalizePolicyRetrieveHybridPromotesLexicalMatch(t *testing.T) {
	candidates := []*RetrieveResult{
		{Memory: &Memory{ID: "weather", Content: "tomorrow will be sunny and warm"}, Score: 0.55},
		{Memory: &Memory{ID: "budget", Content: "project budget is ten thousand dollars"}, Score: 0.50},
	}
	got := finalizePolicyRetrieve(candidates, RetrieveOptions{
		Query:        "project budget",
		HybridSearch: true,
	}, 0.1, 2)
	if len(got) == 0 || got[0].Memory.ID != "budget" {
		t.Fatalf("hybrid ranking = %#v, want budget first", got)
	}
}
