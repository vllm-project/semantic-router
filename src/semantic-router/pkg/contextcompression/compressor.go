package contextcompression

import (
	"encoding/json"
	"io"
	"math"
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
	Format           string
	OmittedChunks    int
}

type scoredChunk struct {
	index int
	text  string
	score float64
	token int
}

// CompressToolOutput keeps query-relevant, leading, and trailing chunks within
// a target token budget. JSON objects and arrays are compressed through string
// leaves so the resulting payload remains valid JSON with the same structure.
func CompressToolOutput(content string, query string, minTokens int, targetTokens int) Result {
	originalTokens := estimateTokens(content)
	minTokens, targetTokens = normalizeTokenBudget(minTokens, targetTokens)
	if originalTokens < minTokens || targetTokens <= 0 || targetTokens >= originalTokens {
		return unchangedResult(content, originalTokens)
	}

	if isJSONObjectOrArray(content) {
		if result, ok := compressJSON(content, query, originalTokens, targetTokens); ok {
			return result
		}
		return unchangedResult(content, originalTokens)
	}

	return compressText(content, query, originalTokens, targetTokens)
}

func compressText(content string, query string, originalTokens int, targetTokens int) Result {
	chunks := chunkText(content, targetTokens)
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
	compressed, omitted := joinSelectedChunks(selected, len(chunks))
	compressedTokens := estimateTokens(compressed)
	if compressedTokens > targetTokens || compressedTokens >= originalTokens {
		return unchangedResult(content, originalTokens)
	}
	return Result{
		Content:          compressed,
		OriginalTokens:   originalTokens,
		CompressedTokens: compressedTokens,
		Applied:          true,
		Format:           "text",
		OmittedChunks:    omitted,
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
		Format:           "text",
	}
}

func scoreChunks(chunks []string, query string) []scoredChunk {
	queryTerms := tokenizeTerms(query)
	documentTerms := make([][]string, len(chunks))
	documentFrequency := make(map[string]int)
	totalDocumentLength := 0
	for index, chunk := range chunks {
		terms := tokenizeTerms(chunk)
		documentTerms[index] = terms
		totalDocumentLength += len(terms)
		seen := make(map[string]struct{}, len(terms))
		for _, term := range terms {
			if _, ok := seen[term]; ok {
				continue
			}
			seen[term] = struct{}{}
			documentFrequency[term]++
		}
	}
	averageDocumentLength := float64(totalDocumentLength) / float64(max(1, len(chunks)))

	scored := make([]scoredChunk, 0, len(chunks))
	for index, chunk := range chunks {
		score := bm25Score(
			queryTerms,
			documentTerms[index],
			documentFrequency,
			len(chunks),
			averageDocumentLength,
		)
		if index == 0 || index == len(chunks)-1 {
			score += 0.35
		}
		scored = append(scored, scoredChunk{
			index: index,
			text:  chunk,
			score: score,
			token: estimateTokens(chunk),
		})
	}
	return scored
}

func selectChunks(scored []scoredChunk, targetTokens int) []scoredChunk {
	selected := make([]scoredChunk, 0, len(scored))
	selectedIndexes := make(map[int]struct{}, len(scored))

	// Reserve the strongest relevance match before adding boundaries. Adding
	// both boundaries first can consume the budget with two omission markers
	// and starve the only chunk that answers the user's query.
	if len(scored) > 0 {
		selected = appendIfFits(
			selected,
			selectedIndexes,
			scored[0],
			len(scored),
			targetTokens,
		)
	}

	// Preserve boundaries when they still fit. The final relevance pass fills
	// the remaining budget without allowing omission markers to exceed it.
	for _, boundary := range []int{0, max(0, len(scored)-1)} {
		for _, candidate := range scored {
			if candidate.index != boundary {
				continue
			}
			selected = appendIfFits(selected, selectedIndexes, candidate, len(scored), targetTokens)
			break
		}
	}
	for _, candidate := range scored {
		if candidate.token <= 0 {
			continue
		}
		selected = appendIfFits(selected, selectedIndexes, candidate, len(scored), targetTokens)
	}
	return selected
}

func appendIfFits(
	selected []scoredChunk,
	selectedIndexes map[int]struct{},
	candidate scoredChunk,
	totalChunks int,
	targetTokens int,
) []scoredChunk {
	if _, exists := selectedIndexes[candidate.index]; exists {
		return selected
	}
	trial := append(append([]scoredChunk(nil), selected...), candidate)
	body, _ := joinSelectedChunks(trial, totalChunks)
	if estimateTokens(body) > targetTokens {
		return selected
	}
	selectedIndexes[candidate.index] = struct{}{}
	return append(selected, candidate)
}

func joinSelectedChunks(selected []scoredChunk, totalChunks int) (string, int) {
	ordered := append([]scoredChunk(nil), selected...)
	sort.Slice(ordered, func(left, right int) bool {
		return ordered[left].index < ordered[right].index
	})
	parts := make([]string, 0, len(ordered)*2)
	previous := -1
	omitted := 0
	for _, candidate := range ordered {
		if previous >= 0 && candidate.index > previous+1 {
			parts = append(parts, omissionMarker)
			omitted += candidate.index - previous - 1
		}
		parts = append(parts, candidate.text)
		previous = candidate.index
	}
	if len(ordered) > 0 {
		omitted += ordered[0].index
		omitted += totalChunks - ordered[len(ordered)-1].index - 1
	}
	return strings.Join(parts, "\n"), omitted
}

func chunkText(content string, targetTokens int) []string {
	lines := strings.Split(content, "\n")
	chunks := make([]string, 0, len(lines))
	maxChunkTokens := max(8, targetTokens/4)
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			chunks = append(chunks, splitBoundedChunk(trimmed, maxChunkTokens)...)
		}
	}
	if len(chunks) == 0 && strings.TrimSpace(content) != "" {
		return splitBoundedChunk(strings.TrimSpace(content), maxChunkTokens)
	}
	return chunks
}

func splitBoundedChunk(text string, maxTokens int) []string {
	if estimateTokens(text) <= maxTokens {
		return []string{text}
	}
	words := strings.Fields(text)
	if len(words) > 1 {
		return splitWords(words, maxTokens)
	}
	return splitRunes([]rune(text), maxTokens)
}

func splitWords(words []string, maxTokens int) []string {
	chunks := make([]string, 0, len(words))
	current := make([]string, 0)
	for _, word := range words {
		trial := strings.Join(current, " ")
		if trial != "" {
			trial += " "
		}
		trial += word
		if len(current) > 0 && estimateTokens(trial) > maxTokens {
			chunks = append(chunks, strings.Join(current, " "))
			current = current[:0]
		}
		current = append(current, word)
	}
	if len(current) > 0 {
		chunks = append(chunks, strings.Join(current, " "))
	}
	return chunks
}

func splitRunes(runes []rune, maxTokens int) []string {
	chunks := make([]string, 0)
	start := 0
	for start < len(runes) {
		stop := start + 1
		for stop <= len(runes) && estimateTokens(string(runes[start:stop])) <= maxTokens {
			stop++
		}
		stop--
		if stop <= start {
			stop = start + 1
		}
		chunks = append(chunks, string(runes[start:stop]))
		start = stop
	}
	return chunks
}

func tokenizeTerms(text string) []string {
	lower := []rune(strings.ToLower(text))
	terms := make([]string, 0, len(lower))
	word := make([]rune, 0)
	flushWord := func() {
		if len(word) > 1 {
			terms = append(terms, string(word))
		}
		word = word[:0]
	}
	var previousCJK rune
	for _, value := range lower {
		switch {
		case isCJK(value):
			flushWord()
			terms = append(terms, string(value))
			if previousCJK != 0 {
				terms = append(terms, string([]rune{previousCJK, value}))
			}
			previousCJK = value
		case unicode.IsLetter(value) || unicode.IsDigit(value) || value == '_':
			previousCJK = 0
			word = append(word, value)
		case unicode.IsSymbol(value):
			flushWord()
			previousCJK = 0
			terms = append(terms, string(value))
		default:
			flushWord()
			previousCJK = 0
		}
	}
	flushWord()
	return terms
}

func isCJK(value rune) bool {
	return unicode.In(value, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul)
}

func bm25Score(
	queryTerms []string,
	documentTerms []string,
	documentFrequency map[string]int,
	documentCount int,
	averageDocumentLength float64,
) float64 {
	if len(queryTerms) == 0 || len(documentTerms) == 0 {
		return 0
	}
	termFrequency := make(map[string]int, len(documentTerms))
	for _, term := range documentTerms {
		termFrequency[term]++
	}
	uniqueQuery := make(map[string]struct{}, len(queryTerms))
	score := 0.0
	const k1 = 1.2
	const b = 0.75
	for _, term := range queryTerms {
		if _, seen := uniqueQuery[term]; seen {
			continue
		}
		uniqueQuery[term] = struct{}{}
		tf := float64(termFrequency[term])
		if tf == 0 {
			continue
		}
		df := float64(documentFrequency[term])
		idf := math.Log(1 + (float64(documentCount)-df+0.5)/(df+0.5))
		lengthRatio := float64(len(documentTerms)) / maxFloat(1, averageDocumentLength)
		score += idf * (tf * (k1 + 1)) / (tf + k1*(1-b+b*lengthRatio))
	}
	return score
}

func estimateTokens(text string) int {
	if text == "" {
		return 0
	}
	heuristic := promptcompression.CountTokensApprox(text)
	byteEstimate := (len([]byte(text)) + 3) / 4
	return max(heuristic, byteEstimate)
}

// EstimateTokens returns the conservative local token estimate used when no
// model/provider counter is available.
func EstimateTokens(text string) int {
	return estimateTokens(text)
}

func isJSONObjectOrArray(content string) bool {
	trimmed := strings.TrimSpace(content)
	return strings.HasPrefix(trimmed, "{") || strings.HasPrefix(trimmed, "[")
}

func compressJSON(
	content string,
	query string,
	originalTokens int,
	targetTokens int,
) (Result, bool) {
	decoder := json.NewDecoder(strings.NewReader(content))
	decoder.UseNumber()
	var value interface{}
	if err := decoder.Decode(&value); err != nil {
		return Result{}, false
	}
	var trailing interface{}
	if err := decoder.Decode(&trailing); err != io.EOF {
		return Result{}, false
	}
	switch value.(type) {
	case map[string]interface{}, []interface{}:
	default:
		return Result{}, false
	}

	skeletonBytes, err := json.Marshal(stripJSONStringLeaves(value))
	if err != nil {
		return Result{}, false
	}
	skeletonTokens := estimateTokens(string(skeletonBytes))
	if skeletonTokens >= targetTokens {
		return Result{}, false
	}
	totalStringTokens := countJSONStringTokens(value)
	if totalStringTokens == 0 {
		return Result{}, false
	}

	available := targetTokens - skeletonTokens
	var compressedValue interface{}
	var omitted int
	for range 3 {
		compressedValue, omitted = compressJSONStringLeaves(value, query, available, totalStringTokens)
		compressedBytes, marshalErr := json.Marshal(compressedValue)
		if marshalErr != nil {
			return Result{}, false
		}
		compressedTokens := estimateTokens(string(compressedBytes))
		if compressedTokens <= targetTokens && compressedTokens < originalTokens {
			return Result{
				Content:          string(compressedBytes),
				OriginalTokens:   originalTokens,
				CompressedTokens: compressedTokens,
				Applied:          true,
				Format:           "json",
				OmittedChunks:    omitted,
			}, true
		}
		available = max(1, int(float64(available)*float64(targetTokens)/float64(max(1, compressedTokens))))
	}
	return Result{}, false
}

func stripJSONStringLeaves(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(typed))
		for key, child := range typed {
			result[key] = stripJSONStringLeaves(child)
		}
		return result
	case []interface{}:
		result := make([]interface{}, len(typed))
		for index, child := range typed {
			result[index] = stripJSONStringLeaves(child)
		}
		return result
	case string:
		return ""
	default:
		return typed
	}
}

func countJSONStringTokens(value interface{}) int {
	switch typed := value.(type) {
	case map[string]interface{}:
		total := 0
		for _, child := range typed {
			total += countJSONStringTokens(child)
		}
		return total
	case []interface{}:
		total := 0
		for _, child := range typed {
			total += countJSONStringTokens(child)
		}
		return total
	case string:
		return estimateTokens(typed)
	default:
		return 0
	}
}

func compressJSONStringLeaves(
	value interface{},
	query string,
	availableTokens int,
	totalStringTokens int,
) (interface{}, int) {
	switch typed := value.(type) {
	case map[string]interface{}:
		result := make(map[string]interface{}, len(typed))
		omitted := 0
		for key, child := range typed {
			compressedChild, childOmitted := compressJSONStringLeaves(
				child,
				query+" "+key,
				availableTokens,
				totalStringTokens,
			)
			result[key] = compressedChild
			omitted += childOmitted
		}
		return result, omitted
	case []interface{}:
		result := make([]interface{}, len(typed))
		omitted := 0
		for index, child := range typed {
			compressedChild, childOmitted := compressJSONStringLeaves(
				child,
				query,
				availableTokens,
				totalStringTokens,
			)
			result[index] = compressedChild
			omitted += childOmitted
		}
		return result, omitted
	case string:
		originalTokens := estimateTokens(typed)
		if originalTokens == 0 {
			return typed, 0
		}
		budget := max(1, availableTokens*originalTokens/max(1, totalStringTokens))
		if originalTokens <= budget {
			return typed, 0
		}
		result := compressText(typed, query, originalTokens, budget)
		if !result.Applied {
			return typed, 0
		}
		return result.Content, result.OmittedChunks
	default:
		return typed, 0
	}
}

func maxFloat(left float64, right float64) float64 {
	if left > right {
		return left
	}
	return right
}
