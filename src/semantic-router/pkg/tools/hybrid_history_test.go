package tools_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

var _ = Describe("FilterAndRankToolsWithConversation hybrid_history", func() {
	It("falls back to semantic-only ranking when history is too short", func() {
		minSteps := 2
		th := float32(0.05)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
			HybridHistory: &config.HybridHistoryToolRetrievalConfig{
				MinHistorySteps:            &minSteps,
				HistoryConfidenceThreshold: &th,
			},
		}
		candidates := []tools.ToolSimilarity{
			candidate("alpha", "desc", "c", nil, 0.9),
			candidate("beta", "desc", "c", nil, 0.5),
		}
		selected := tools.FilterAndRankToolsWithConversation("q", candidates, 2, advanced, "", []string{"alpha"}, 0.9)
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("alpha"))
	})

	It("ranks by transition when last tool predicts the next", func() {
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
		}
		// History: tool_a -> tool_b, then tool_b -> tool_b (last call is tool_b)
		hist := []string{"tool_a", "tool_b", "tool_b"}
		candidates := []tools.ToolSimilarity{
			candidate("tool_b", "next after b", "c", nil, 0.5),
			candidate("tool_c", "other", "c", nil, 0.5),
		}
		selected := tools.FilterAndRankToolsWithConversation("query", candidates, 2, advanced, "", hist, 0.5)
		Expect(selected[0].Function.Name).To(Equal("tool_b"))
	})

	It("applies repetition penalty when the same tool appears often in the window", func() {
		rep := float32(0.5)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
			HybridHistory: &config.HybridHistoryToolRetrievalConfig{
				RepetitionPenaltyStrength: &rep,
			},
		}
		hist := []string{"spam", "spam", "spam", "pivot_step"}
		candidates := []tools.ToolSimilarity{
			candidate("spam", "does spam", "c", nil, 0.99),
			candidate("other", "does other", "c", nil, 0.4),
		}
		selected := tools.FilterAndRankToolsWithConversation("q", candidates, 2, advanced, "", hist, 0.9)
		Expect(selected[0].Function.Name).To(Equal("other"))
	})
})

// hybrid_history signal strength: integration-style checks that fallback matches pure semantic order,
// while full hybrid can reorder when transition strongly favors a lower-similarity tool (multi-step plan).
var _ = Describe("hybrid_history fallback vs active hybrid (realistic scenarios)", func() {
	// Shared scenario: query matches both tools lexically; semantic-only would prefer high_rank.
	// History mimics "already called next_tool twice after open_session" — transition favors next_tool again.
	historyFavoringNextTool := []string{"open_session", "next_tool", "next_tool"}
	highThenLow := []tools.ToolSimilarity{
		candidate("high_rank", "generic assistant description user query", "misc", nil, 0.96),
		candidate("next_tool", "specialized follow-up chain step", "misc", nil, 0.38),
	}

	It("falls back to semantic-only when history_signal_strength is below history_confidence_threshold", func() {
		// len=3, horizon=8 -> strength = 3/8 * diversity(1) = 0.375; threshold 0.99 forces fallback.
		th := float32(0.99)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
			HybridHistory: &config.HybridHistoryToolRetrievalConfig{
				HistoryConfidenceThreshold: &th,
			},
		}
		selected := tools.FilterAndRankToolsWithConversation(
			"assistant query user description",
			highThenLow,
			2,
			advanced,
			"",
			historyFavoringNextTool,
			0.85,
		)
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("high_rank"), "pure semantic: highest similarity first")
		Expect(selected[1].Function.Name).To(Equal("next_tool"))
	})

	It("does not fall back when history_signal_strength meets history_confidence_threshold — hybrid can outrank by transition", func() {
		// Same history and candidates; threshold 0.2 so 0.375 > 0.2 — run hybrid. Transition from last
		// `next_tool` strongly supports another `next_tool` call; that beats high_rank on embedding alone.
		th := float32(0.2)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
			HybridHistory: &config.HybridHistoryToolRetrievalConfig{
				HistoryConfidenceThreshold: &th,
			},
		}
		selected := tools.FilterAndRankToolsWithConversation(
			"assistant query user description",
			highThenLow,
			2,
			advanced,
			"",
			historyFavoringNextTool,
			0.85,
		)
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("next_tool"), "transition + equal weights should promote the plan-consistent tool")
		Expect(selected[1].Function.Name).To(Equal("high_rank"))
	})

	It("falls back to semantic-only when transcript has fewer tool steps than min_history_steps (cold start)", func() {
		minSteps := 3
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			RetrievalStrategy: config.ToolRetrievalStrategyHybridHistory,
			HybridHistory: &config.HybridHistoryToolRetrievalConfig{
				MinHistorySteps: &minSteps,
			},
		}
		// Only one prior tool in window — not enough for hybrid per policy.
		hist := []string{"only_prior_tool"}
		candidates := []tools.ToolSimilarity{
			candidate("strong_match", "matches query vocabulary", "misc", nil, 0.91),
			candidate("weak_match", "looser text match", "misc", nil, 0.42),
		}
		selected := tools.FilterAndRankToolsWithConversation("query words strong vocabulary", candidates, 2, advanced, "", hist, 0.7)
		Expect(selected[0].Function.Name).To(Equal("strong_match"))
		Expect(selected[1].Function.Name).To(Equal("weak_match"))
	})
})
