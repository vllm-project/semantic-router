package promptcompression

import "testing"

func TestProfileConfigAppliesDomainWeights(t *testing.T) {
	cfg := ProfileConfig("security", 2048)
	if cfg.MaxTokens != 2048 {
		t.Fatalf("expected token budget to be preserved, got %d", cfg.MaxTokens)
	}
	if cfg.NoveltyWeight != 0.30 {
		t.Fatalf("expected security novelty weight 0.30, got %.2f", cfg.NoveltyWeight)
	}
	if cfg.PreserveFirstN != 2 || cfg.PreserveLastN != 3 {
		t.Fatalf("unexpected security preserve window: first=%d last=%d", cfg.PreserveFirstN, cfg.PreserveLastN)
	}
}

func TestProfileConfigNormalizesProfileName(t *testing.T) {
	cfg := ProfileConfig("multi-turn", 4096)
	if cfg.PreserveLastN != 5 {
		t.Fatalf("expected multi-turn alias to apply profile, got preserve_last_n=%d", cfg.PreserveLastN)
	}
}

func TestProfileConfigFallsBackToDefault(t *testing.T) {
	got := ProfileConfig("unknown", 1024)
	want := DefaultConfig(1024)
	if got != want {
		t.Fatalf("expected unknown profile to use default config, got %+v want %+v", got, want)
	}
}
