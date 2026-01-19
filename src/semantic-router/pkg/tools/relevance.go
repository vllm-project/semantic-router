// Package tools provides tool selection and filtering capabilities
// for the semantic router.
package tools

import (
	"sort"
	"strings"
	"unicode"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// FilterAndRankTools applies advanced filtering and ranking to tool candidates.
func FilterAndRankTools(query string, candidates []ToolSimilarity, topK int, advanced *config.AdvancedToolFilteringConfig, selectedCategory string) []openai.ChatCompletionToolParam {
	if len(candidates) == 0 {
		return []openai.ChatCompletionToolParam{}
	}
	if advanced == nil || !advanced.Enabled {
		return selectTopKBySimilarity(candidates, topK)
	}

	filtered := applyAllowBlockFilters(candidates, advanced.AllowTools, advanced.BlockTools)
	filtered = applyCategoryFilter(filtered, advanced.UseCategoryFilter, selectedCategory)

	queryTokens := tokenize(query)
	querySet := tokenSet(queryTokens)
	queryTokenCount := len(queryTokens)

	weights := resolveWeights(advanced.Weights)
	weightSum := weights.embed + weights.lexical + weights.tag + weights.name + weights.category
	minOverlap := 0
	if advanced.MinLexicalOverlap != nil {
		minOverlap = *advanced.MinLexicalOverlap
	}

	minCombined := float32(0)
	if advanced.MinCombinedScore != nil {
		minCombined = *advanced.MinCombinedScore
	}

	scored := make([]scoredCandidate, 0, len(filtered))
	for _, candidate := range filtered {
		nameTokens := tokenize(candidate.Entry.Tool.Function.Name)
		descriptionTokens := tokenize(candidate.Entry.Description)
		categoryTokens := tokenize(candidate.Entry.Category)
		tagTokens := collectTagTokens(candidate.Entry.Tags)

		lexicalTokens := make([]string, 0, len(nameTokens)+len(descriptionTokens)+len(categoryTokens))
		lexicalTokens = append(lexicalTokens, nameTokens...)
		lexicalTokens = append(lexicalTokens, descriptionTokens...)
		lexicalTokens = append(lexicalTokens, categoryTokens...)

		lexicalOverlap := countOverlap(querySet, lexicalTokens)
		if minOverlap > 0 && lexicalOverlap < minOverlap {
			continue
		}

		lexicalScore := scoreOverlap(queryTokenCount, lexicalOverlap)
		tagScore := scoreOverlap(len(tagTokens), countOverlap(querySet, tagTokens))
		nameScore := nameMatchScore(nameTokens, querySet)
		categoryScore := categoryMatchScore(selectedCategory, candidate.Entry.Category)
		similarityScore := clamp01(candidate.Similarity)

		combinedScore := float32(0)
		if weightSum > 0 {
			combinedScore = (weights.embed*similarityScore +
				weights.lexical*lexicalScore +
				weights.tag*tagScore +
				weights.name*nameScore +
				weights.category*categoryScore) / weightSum
		}

		if combinedScore < minCombined {
			continue
		}

		scored = append(scored, scoredCandidate{
			ToolSimilarity: candidate,
			CombinedScore:  combinedScore,
		})
	}

	if len(scored) == 0 {
		return []openai.ChatCompletionToolParam{}
	}

	sort.Slice(scored, func(i, j int) bool {
		if scored[i].CombinedScore == scored[j].CombinedScore {
			if scored[i].Similarity == scored[j].Similarity {
				return scored[i].Entry.Tool.Function.Name < scored[j].Entry.Tool.Function.Name
			}
			return scored[i].Similarity > scored[j].Similarity
		}
		return scored[i].CombinedScore > scored[j].CombinedScore
	})

	limit := topK
	if limit <= 0 || limit > len(scored) {
		limit = len(scored)
	}

	selected := make([]openai.ChatCompletionToolParam, 0, limit)
	for i := 0; i < limit; i++ {
		selected = append(selected, scored[i].Entry.Tool)
	}

	return selected
}

type scoredCandidate struct {
	ToolSimilarity
	CombinedScore float32
}

type resolvedWeights struct {
	embed    float32
	lexical  float32
	tag      float32
	name     float32
	category float32
}

func resolveWeights(weights config.ToolFilteringWeights) resolvedWeights {
	resolved := resolvedWeights{}
	anySet := false
	if weights.Embed != nil {
		resolved.embed = *weights.Embed
		anySet = true
	}
	if weights.Lexical != nil {
		resolved.lexical = *weights.Lexical
		anySet = true
	}
	if weights.Tag != nil {
		resolved.tag = *weights.Tag
		anySet = true
	}
	if weights.Name != nil {
		resolved.name = *weights.Name
		anySet = true
	}
	if weights.Category != nil {
		resolved.category = *weights.Category
		anySet = true
	}

	if !anySet {
		resolved.embed = 1
	}

	return resolved
}

func selectTopKBySimilarity(candidates []ToolSimilarity, topK int) []openai.ChatCompletionToolParam {
	if len(candidates) == 0 {
		return []openai.ChatCompletionToolParam{}
	}

	sorted := make([]ToolSimilarity, len(candidates))
	copy(sorted, candidates)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Similarity > sorted[j].Similarity
	})

	limit := topK
	if limit <= 0 || limit > len(sorted) {
		limit = len(sorted)
	}

	selected := make([]openai.ChatCompletionToolParam, 0, limit)
	for i := 0; i < limit; i++ {
		selected = append(selected, sorted[i].Entry.Tool)
	}

	return selected
}

func applyAllowBlockFilters(candidates []ToolSimilarity, allowList []string, blockList []string) []ToolSimilarity {
	allowSet := normalizeNameSet(allowList)
	blockSet := normalizeNameSet(blockList)

	filtered := make([]ToolSimilarity, 0, len(candidates))
	for _, candidate := range candidates {
		name := strings.ToLower(candidate.Entry.Tool.Function.Name)
		if len(allowSet) > 0 {
			if _, ok := allowSet[name]; !ok {
				continue
			}
		}
		if _, blocked := blockSet[name]; blocked {
			continue
		}
		filtered = append(filtered, candidate)
	}

	return filtered
}

func applyCategoryFilter(candidates []ToolSimilarity, useCategoryFilter *bool, selectedCategory string) []ToolSimilarity {
	if useCategoryFilter == nil || !*useCategoryFilter {
		return candidates
	}
	if strings.TrimSpace(selectedCategory) == "" {
		return candidates
	}

	selectedCategory = strings.ToLower(strings.TrimSpace(selectedCategory))
	filtered := make([]ToolSimilarity, 0, len(candidates))
	for _, candidate := range candidates {
		candidateCategory := strings.ToLower(strings.TrimSpace(candidate.Entry.Category))
		if candidateCategory == selectedCategory {
			filtered = append(filtered, candidate)
		}
	}

	if len(filtered) == 0 {
		return candidates
	}
	return filtered
}

func normalizeNameSet(names []string) map[string]struct{} {
	if len(names) == 0 {
		return map[string]struct{}{}
	}

	set := make(map[string]struct{}, len(names))
	for _, name := range names {
		trimmed := strings.TrimSpace(strings.ToLower(name))
		if trimmed == "" {
			continue
		}
		set[trimmed] = struct{}{}
	}
	return set
}

func tokenize(input string) []string {
	if input == "" {
		return []string{}
	}

	lower := strings.ToLower(input)
	tokens := make([]string, 0)
	var builder strings.Builder

	flush := func() {
		if builder.Len() == 0 {
			return
		}
		tokens = append(tokens, builder.String())
		builder.Reset()
	}

	for _, r := range lower {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			builder.WriteRune(r)
			continue
		}
		flush()
	}
	flush()

	return tokens
}

func tokenSet(tokens []string) map[string]struct{} {
	set := make(map[string]struct{}, len(tokens))
	for _, token := range tokens {
		if token == "" {
			continue
		}
		set[token] = struct{}{}
	}
	return set
}

func collectTagTokens(tags []string) []string {
	if len(tags) == 0 {
		return []string{}
	}

	collected := make([]string, 0, len(tags))
	for _, tag := range tags {
		collected = append(collected, tokenize(tag)...)
	}
	return collected
}

func countOverlap(querySet map[string]struct{}, tokens []string) int {
	if len(querySet) == 0 || len(tokens) == 0 {
		return 0
	}

	seen := make(map[string]struct{})
	count := 0
	for _, token := range tokens {
		if token == "" {
			continue
		}
		if _, ok := querySet[token]; !ok {
			continue
		}
		if _, already := seen[token]; already {
			continue
		}
		seen[token] = struct{}{}
		count++
	}
	return count
}

func scoreOverlap(denominator int, overlap int) float32 {
	if denominator <= 0 || overlap <= 0 {
		return 0
	}
	return float32(overlap) / float32(denominator)
}

func nameMatchScore(nameTokens []string, querySet map[string]struct{}) float32 {
	if len(nameTokens) == 0 || len(querySet) == 0 {
		return 0
	}
	for _, token := range nameTokens {
		if _, ok := querySet[token]; !ok {
			return 0
		}
	}
	return 1
}

func categoryMatchScore(selectedCategory string, toolCategory string) float32 {
	if strings.TrimSpace(selectedCategory) == "" || strings.TrimSpace(toolCategory) == "" {
		return 0
	}
	if strings.EqualFold(strings.TrimSpace(selectedCategory), strings.TrimSpace(toolCategory)) {
		return 1
	}
	return 0
}

func clamp01(value float32) float32 {
	return max(float32(0), min(float32(1), value))
}
