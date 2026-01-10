package tools_test

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

var _ = Describe("FilterAndRankTools", func() {
	It("should return empty slice when candidates are empty", func() {
		selected := tools.FilterAndRankTools("query", []tools.ToolSimilarity{}, 5, nil, "")
		Expect(selected).To(BeEmpty())
	})

	It("should honor allow/block lists", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "weather report", "weather", []string{"weather"}, 0.9),
			candidate("search_web", "search the web", "search", []string{"search"}, 0.8),
			candidate("send_email", "send email", "communication", []string{"email"}, 0.7),
		}

		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:    true,
			AllowTools: []string{"send_email"},
		}

		selected := tools.FilterAndRankTools("weather", candidates, 3, advanced, "")
		Expect(selected).To(HaveLen(1))
		Expect(selected[0].Function.Name).To(Equal("send_email"))
	})

	It("should filter by category when enabled", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "weather report", "weather", []string{"weather"}, 0.6),
			candidate("search_web", "search the web", "search", []string{"search"}, 0.9),
		}

		useCategory := true
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			UseCategoryFilter: &useCategory,
		}

		selected := tools.FilterAndRankTools("weather", candidates, 3, advanced, "weather")
		Expect(selected).To(HaveLen(1))
		Expect(selected[0].Function.Name).To(Equal("get_weather"))
	})

	It("should fall back when category filter has no matches", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "weather report", "weather", []string{"weather"}, 0.6),
			candidate("search_web", "search the web", "search", []string{"search"}, 0.9),
		}

		useCategory := true
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			UseCategoryFilter: &useCategory,
		}

		selected := tools.FilterAndRankTools("weather", candidates, 3, advanced, "finance")
		Expect(selected).To(HaveLen(2))
	})

	It("should apply lexical overlap filtering", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "weather report", "weather", []string{"weather"}, 0.6),
			candidate("send_email", "send email", "communication", []string{"email"}, 0.9),
		}

		minOverlap := 1
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			MinLexicalOverlap: &minOverlap,
		}

		selected := tools.FilterAndRankTools("weather", candidates, 3, advanced, "")
		Expect(selected).To(HaveLen(1))
		Expect(selected[0].Function.Name).To(Equal("get_weather"))
	})

	It("should only count lexical overlap (not tags) for min_lexical_overlap", func() {
		candidates := []tools.ToolSimilarity{
			candidate("tag_only", "sunny skies", "misc", []string{"weather"}, 0.8),
			candidate("lexical_hit", "weather report", "weather", []string{"forecast"}, 0.5),
		}

		minOverlap := 1
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:           true,
			MinLexicalOverlap: &minOverlap,
		}

		selected := tools.FilterAndRankTools("weather", candidates, 2, advanced, "")
		Expect(selected).To(HaveLen(1))
		Expect(selected[0].Function.Name).To(Equal("lexical_hit"))
	})

	It("should rank by combined score", func() {
		candidates := []tools.ToolSimilarity{
			candidate("tool_a", "unrelated content", "misc", []string{}, 0.9),
			candidate("tool_b", "weather report", "weather", []string{"weather"}, 0.6),
		}

		weightEmbed := float32(0.4)
		weightLexical := float32(0.6)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled: true,
			Weights: config.ToolFilteringWeights{
				Embed:   &weightEmbed,
				Lexical: &weightLexical,
			},
		}

		selected := tools.FilterAndRankTools("weather", candidates, 2, advanced, "")
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("tool_b"))
	})

	It("should not force embed weight when all weights are explicitly zero", func() {
		candidates := []tools.ToolSimilarity{
			candidate("tool_a", "weather report", "weather", []string{}, 0.9),
			candidate("tool_b", "weather report", "weather", []string{}, 0.8),
		}

		zero := float32(0)
		minCombined := float32(0.1)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:          true,
			MinCombinedScore: &minCombined,
			Weights: config.ToolFilteringWeights{
				Embed:    &zero,
				Lexical:  &zero,
				Tag:      &zero,
				Name:     &zero,
				Category: &zero,
			},
		}

		selected := tools.FilterAndRankTools("weather", candidates, 2, advanced, "")
		Expect(selected).To(BeEmpty())
	})

	It("should normalize combined score when weights sum exceeds 1", func() {
		candidates := []tools.ToolSimilarity{
			candidate("tool_a", "weather report forecast", "weather", []string{}, 0.8),
		}

		one := float32(1)
		minCombined := float32(0.9)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:          true,
			MinCombinedScore: &minCombined,
			Weights: config.ToolFilteringWeights{
				Embed:   &one,
				Lexical: &one,
			},
		}

		selected := tools.FilterAndRankTools("weather report forecast today", candidates, 1, advanced, "")
		Expect(selected).To(BeEmpty())
	})

	It("should honor block list", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "weather report", "weather", []string{"weather"}, 0.9),
			candidate("send_email", "send email", "communication", []string{"email"}, 0.8),
		}

		advanced := &config.AdvancedToolFilteringConfig{
			Enabled:    true,
			BlockTools: []string{"send_email"},
		}

		selected := tools.FilterAndRankTools("weather", candidates, 3, advanced, "")
		Expect(selected).To(HaveLen(1))
		Expect(selected[0].Function.Name).To(Equal("get_weather"))
	})

	It("should rank higher when tag weight is set", func() {
		candidates := []tools.ToolSimilarity{
			candidate("tag_match", "general info", "misc", []string{"weather"}, 0.2),
			candidate("no_tag", "general info", "misc", []string{"other"}, 0.9),
		}

		weightTag := float32(1)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled: true,
			Weights: config.ToolFilteringWeights{
				Tag: &weightTag,
			},
		}

		selected := tools.FilterAndRankTools("weather", candidates, 2, advanced, "")
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("tag_match"))
	})

	It("should rank higher when name weight is set", func() {
		candidates := []tools.ToolSimilarity{
			candidate("get_weather", "general info", "misc", []string{}, 0.2),
			candidate("search_web", "general info", "misc", []string{}, 0.9),
		}

		weightName := float32(1)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled: true,
			Weights: config.ToolFilteringWeights{
				Name: &weightName,
			},
		}

		selected := tools.FilterAndRankTools("get weather", candidates, 2, advanced, "")
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("get_weather"))
	})

	It("should rank higher when category weight is set", func() {
		candidates := []tools.ToolSimilarity{
			candidate("weather_tool", "general info", "weather", []string{}, 0.2),
			candidate("search_tool", "general info", "search", []string{}, 0.9),
		}

		weightCategory := float32(1)
		advanced := &config.AdvancedToolFilteringConfig{
			Enabled: true,
			Weights: config.ToolFilteringWeights{
				Category: &weightCategory,
			},
		}

		selected := tools.FilterAndRankTools("anything", candidates, 2, advanced, "weather")
		Expect(selected).To(HaveLen(2))
		Expect(selected[0].Function.Name).To(Equal("weather_tool"))
	})
})

func candidate(name string, description string, category string, tags []string, similarity float32) tools.ToolSimilarity {
	return tools.ToolSimilarity{
		Entry: tools.ToolEntry{
			Tool: openai.ChatCompletionToolParam{
				Type: "function",
				Function: openai.FunctionDefinitionParam{
					Name:        name,
					Description: param.NewOpt(description),
				},
			},
			Description: description,
			Category:    category,
			Tags:        tags,
		},
		Similarity: similarity,
	}
}
