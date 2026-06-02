package promptcompression

import "strings"

// ProfileConfig returns the built-in prompt-compression profile for a common
// workload while preserving the caller's token budget.
func ProfileConfig(profile string, maxTokens int) Config {
	cfg := DefaultConfig(maxTokens)
	switch normalizeProfile(profile) {
	case "coding":
		cfg.TextRankWeight = 0.15
		cfg.PositionWeight = 0.30
		cfg.TFIDFWeight = 0.40
		cfg.NoveltyWeight = 0.15
		cfg.PositionDepth = 0.60
		cfg.PreserveFirstN = 2
		cfg.PreserveLastN = 4
	case "medical":
		cfg.TextRankWeight = 0.20
		cfg.PositionWeight = 0.35
		cfg.TFIDFWeight = 0.30
		cfg.NoveltyWeight = 0.15
		cfg.PreserveFirstN = 3
		cfg.PreserveLastN = 2
	case "security":
		cfg.TextRankWeight = 0.15
		cfg.PositionWeight = 0.30
		cfg.TFIDFWeight = 0.25
		cfg.NoveltyWeight = 0.30
		cfg.PositionDepth = 0.60
		cfg.PreserveFirstN = 2
		cfg.PreserveLastN = 3
	case "multi_turn":
		cfg.TextRankWeight = 0.15
		cfg.PositionWeight = 0.25
		cfg.TFIDFWeight = 0.35
		cfg.NoveltyWeight = 0.10
		cfg.PreserveFirstN = 2
		cfg.PreserveLastN = 5
	}
	return cfg
}

func normalizeProfile(profile string) string {
	return strings.ReplaceAll(strings.ToLower(strings.TrimSpace(profile)), "-", "_")
}
