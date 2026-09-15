package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestSampleQueryWindowsKeepsBothEnds(t *testing.T) {
	windows := make([]embedding.Window, 47)
	for i := range windows {
		windows[i] = embedding.Window{Start: i * 10, End: i*10 + 10}
	}

	kept := sampleQueryWindows(windows, 8)
	if len(kept) != 8 {
		t.Fatalf("kept %d windows, want 8", len(kept))
	}
	if kept[0] != windows[0] {
		t.Errorf("first window is %v, want %v", kept[0], windows[0])
	}
	if kept[len(kept)-1] != windows[len(windows)-1] {
		t.Errorf("last window is %v, want %v", kept[len(kept)-1], windows[len(windows)-1])
	}
	for i := 1; i < len(kept); i++ {
		if kept[i].Start <= kept[i-1].Start {
			t.Fatalf("windows are not in order at %d: %v then %v", i, kept[i-1], kept[i])
		}
	}
}

func TestSampleQueryWindowsBelowTheLimit(t *testing.T) {
	windows := []embedding.Window{{Start: 0, End: 10}, {Start: 5, End: 15}}
	if kept := sampleQueryWindows(windows, 8); len(kept) != 2 {
		t.Fatalf("kept %d windows, want both", len(kept))
	}
	if kept := sampleQueryWindows(windows, 1); len(kept) != 1 || kept[0] != windows[0] {
		t.Fatalf("a limit of one keeps the first window, got %v", kept)
	}
}

func TestRagHitsKeepBestScorePerDocument(t *testing.T) {
	var hits ragHits
	hits.add([]string{"alpha", "beta"}, []float32{0.30, 0.90})
	hits.add([]string{"alpha", "gamma"}, []float32{0.80, 0.40})

	contents, scores := hits.top(0)
	if len(contents) != 3 {
		t.Fatalf("kept %d documents, want 3", len(contents))
	}
	want := []struct {
		content string
		score   float32
	}{{"beta", 0.90}, {"alpha", 0.80}, {"gamma", 0.40}}
	for i, expected := range want {
		if contents[i] != expected.content || scores[i] != expected.score {
			t.Fatalf("rank %d is %q at %.2f, want %q at %.2f", i, contents[i], scores[i], expected.content, expected.score)
		}
	}

	topContents, topScores := hits.top(2)
	if len(topContents) != 2 || len(topScores) != 2 || topContents[0] != "beta" {
		t.Fatalf("top 2 is %v at %v", topContents, topScores)
	}
}

func TestRagHitsToleratesMissingScores(t *testing.T) {
	var hits ragHits
	hits.add([]string{"alpha", "beta"}, nil)
	contents, scores := hits.top(0)
	if len(contents) != 2 || len(scores) != 2 {
		t.Fatalf("kept %v at %v", contents, scores)
	}
}
