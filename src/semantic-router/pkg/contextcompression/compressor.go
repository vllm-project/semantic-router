package contextcompression

import (
	"sort"
	"strings"
	"unicode"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

const omissionMarker = "\n[... context omitted by route compression ...]\n"

// Result describes one extractive tool-output compression.
type Result struct {
	Content          string
	OriginalTokens   int
	CompressedTokens int
	Applied          bool
}

type scoredChunk struct {
	index int
	text  string
	score float64
	token int
}

// CompressToolOutput keeps query-relevant, leading, and trailing chunks within
// a target token budget.
func CompressToolOutput(content string, query string, minTokens int, targetTokens int) Result {
	originalTokens := promptcompression.CountTokensApprox(content)
	minTokens, targetTokens = normalizeTokenBudget(minTokens, targetTokens)
	if originalTokens < minTokens || targetTokens <= 0 || targetTokens >= originalTokens {
		return unchangedResult(content, originalTokens)
	}

	chunks := chunkText(content)
	if len(chunks) < 2 {
		return unchangedResult(content, originalTokens)
	}
	scored := scoreChunks(chunks, query)
	sort.SliceStable(scored, func(left, right int) bool {
		return scored[left].score > scored[right].score
	})
	selected := selectChunks(scored, targetTokens)
	if len(selected) == 0 {
		return unchangedResult(content, originalTokens)
	}
	compressed := joinSelectedChunks(selected)
	compressedTokens := promptcompression.CountTokensApprox(compressed)
	return Result{
		Content:          compressed,
		OriginalTokens:   originalTokens,
		CompressedTokens: compressedTokens,
		Applied:          compressedTokens < originalTokens,
	}
}

func normalizeTokenBudget(minTokens int, targetTokens int) (int, int) {
	if minTokens <= 0 {
		minTokens = 2000
	}
	if targetTokens <= 0 {
		targetTokens = minTokens / 2
	}
	return minTokens, targetTokens
}

func unchangedResult(content string, originalTokens int) Result {
	return Result{
		Content:          content,
		OriginalTokens:   originalTokens,
		CompressedTokens: originalTokens,
	}
}

func scoreChunks(chunks []string, query string) []scoredChunk {
	queryTerms := termSet(query)
	scored := make([]scoredChunk, 0, len(chunks))
	for index, chunk := range chunks {
		score := overlapScore(queryTerms, termSet(chunk))
		if index == 0 || index == len(chunks)-1 {
			score += 0.25
		}
		scored = append(scored, scoredChunk{
			index: index,
			text:  chunk,
			score: score,
			token: promptcompression.CountTokensApprox(chunk),
		})
	}
	return scored
}

func selectChunks(scored []scoredChunk, targetTokens int) []scoredChunk {
	selected := make([]scoredChunk, 0, len(scored))
	used := 0
	for _, candidate := range scored {
		if candidate.token <= 0 || used+candidate.token > targetTokens {
			continue
		}
		selected = append(selected, candidate)
		used += candidate.token
	}
	return selected
}

func joinSelectedChunks(selected []scoredChunk) string {
	sort.Slice(selected, func(left, right int) bool {
		return selected[left].index < selected[right].index
	})
	parts := make([]string, 0, len(selected)*2)
	previous := -1
	for _, candidate := range selected {
		if previous >= 0 && candidate.index > previous+1 {
			parts = append(parts, omissionMarker)
		}
		parts = append(parts, candidate.text)
		previous = candidate.index
	}
	return strings.Join(parts, "\n")
}

func chunkText(content string) []string {
	lines := strings.Split(content, "\n")
	chunks := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			chunks = append(chunks, trimmed)
		}
	}
	if len(chunks) > 1 {
		return chunks
	}

	words := strings.Fields(content)
	const wordsPerChunk = 80
	chunks = chunks[:0]
	for start := 0; start < len(words); start += wordsPerChunk {
		stop := min(start+wordsPerChunk, len(words))
		chunks = append(chunks, strings.Join(words[start:stop], " "))
	}
	return chunks
}

func termSet(text string) map[string]struct{} {
	terms := make(map[string]struct{})
	for _, field := range strings.FieldsFunc(strings.ToLower(text), func(value rune) bool {
		return !unicode.IsLetter(value) && !unicode.IsDigit(value) && value != '_'
	}) {
		if len(field) > 1 {
			terms[field] = struct{}{}
		}
	}
	return terms
}

func overlapScore(queryTerms map[string]struct{}, chunkTerms map[string]struct{}) float64 {
	if len(queryTerms) == 0 || len(chunkTerms) == 0 {
		return 0
	}
	matched := 0
	for term := range queryTerms {
		if _, ok := chunkTerms[term]; ok {
			matched++
		}
	}
	return float64(matched) / float64(len(queryTerms))
}
