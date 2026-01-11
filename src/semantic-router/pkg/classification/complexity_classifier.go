package classification

import (
	"math"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ComplexityClassifier implements language-agnostic query complexity classification
// Uses structural features (length, word count, punctuation, etc.) that work for any language
// rather than language-specific keywords, making it suitable for multilingual queries
type ComplexityClassifier struct {
	rules []config.ComplexityRule
}

// ComplexityResult represents the result of complexity classification
type ComplexityResult struct {
	ComplexityType string  // "simple", "medium", "complex"
	Confidence     float64 // Confidence score (0.0-1.0)
}

// NewComplexityClassifier creates a new complexity classifier
func NewComplexityClassifier(cfgRules []config.ComplexityRule) (*ComplexityClassifier, error) {
	return &ComplexityClassifier{
		rules: cfgRules,
	}, nil
}

// Classify determines query complexity using heuristics
func (c *ComplexityClassifier) Classify(text string) (*ComplexityResult, error) {
	if text == "" {
		return &ComplexityResult{
			ComplexityType: "simple",
			Confidence:     1.0,
		}, nil
	}

	// Calculate complexity score using heuristics
	score := c.calculateComplexityScore(text)

	// Get configurable thresholds (default: 0.30, 0.60)
	// Adjusted from 0.35/0.65 to better capture complex queries scoring 0.70-0.72
	simpleThreshold := 0.30
	complexThreshold := 0.60
	if len(c.rules) > 0 {
		if c.rules[0].SimpleThreshold != nil {
			simpleThreshold = *c.rules[0].SimpleThreshold
		}
		if c.rules[0].ComplexThreshold != nil {
			complexThreshold = *c.rules[0].ComplexThreshold
		}
	}

	// Determine complexity level with smooth confidence transitions
	var complexityType string
	var confidence float64

	if score <= simpleThreshold {
		complexityType = "simple"
		// Confidence: 0.9 at score=0, 0.75 at threshold
		if simpleThreshold > 0 {
			confidence = 0.9 - (score/simpleThreshold)*0.15
		} else {
			confidence = 0.9
		}
	} else if score <= complexThreshold {
		complexityType = "medium"
		// Confidence: 0.75 at simple threshold, 0.70 at complex threshold
		thresholdRange := complexThreshold - simpleThreshold
		if thresholdRange > 0 {
			normalizedScore := (score - simpleThreshold) / thresholdRange // 0 to 1 in medium range
			confidence = 0.75 - normalizedScore*0.05
		} else {
			confidence = 0.75
		}
	} else {
		complexityType = "complex"
		// Confidence: 0.70 at threshold, 0.85 at score=1.0
		complexRange := 1.0 - complexThreshold
		if complexRange > 0 {
			normalizedScore := (score - complexThreshold) / complexRange // 0 to 1 in complex range
			confidence = 0.70 + normalizedScore*0.15
		} else {
			confidence = 0.70
		}
	}

	// Ensure confidence is in valid range
	if confidence < 0.5 {
		confidence = 0.5
	}
	if confidence > 1.0 {
		confidence = 1.0
	}

	logging.Infof("Complexity classification: type=%s, score=%.2f, confidence=%.2f", complexityType, score, confidence)

	return &ComplexityResult{
		ComplexityType: complexityType,
		Confidence:     confidence,
	}, nil
}

// calculateComplexityScore calculates a complexity score (0.0-1.0) based on language-agnostic heuristics
// This approach works for any language by focusing on structural features rather than language-specific keywords
// Uses weighted sum (not average) to better capture complexity
func (c *ComplexityClassifier) calculateComplexityScore(text string) float64 {
	trimmedText := strings.TrimSpace(text)
	if trimmedText == "" {
		return 0.0
	}

	// Use rune count (Unicode characters) instead of byte length for accurate multilingual support
	textLength := utf8.RuneCountInString(trimmedText)
	if textLength == 0 {
		return 0.0
	}

	// Detect if text uses space-separated languages (most languages) or character-based (CJK)
	// For CJK languages, use character count instead of word count
	words := strings.Fields(trimmedText)
	wordCount := len(words)

	// Check if text contains CJK characters (Chinese, Japanese, Korean)
	hasCJK := false
	for _, r := range trimmedText {
		if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
			hasCJK = true
			break
		}
	}

	// For CJK languages, use character count as "word" count since they don't use spaces
	effectiveWordCount := wordCount
	if hasCJK && wordCount == 1 && textLength > 1 {
		// Likely CJK text without spaces - use character count
		effectiveWordCount = textLength
	}

	// Calculate all complexity factors
	lengthScore := c.calculateLengthScore(textLength)
	wordScore := c.calculateWordScore(effectiveWordCount)
	sentenceScore := c.calculateSentenceScore(trimmedText)
	punctuationScore := c.calculatePunctuationScore(trimmedText)
	longTokenScore := c.calculateLongTokenScore(trimmedText, words, hasCJK, wordCount)
	diversityScore := c.calculateDiversityScore(trimmedText, textLength)
	tokenScore, hasTokenScore := c.calculateTokenScore(trimmedText, textLength, hasCJK)
	structureScore := c.calculateStructureScore(trimmedText)
	constraintScore := c.calculateConstraintScore(trimmedText)
	reasoningScore := c.calculateReasoningScore(trimmedText)

	// Weighted sum allows scores to accumulate
	// Weights are heuristic-based; token count (20%) correlates best with inference cost
	// If BERT unavailable, token count is excluded and weights redistributed
	var (
		lengthWeight      = 0.15
		tokenWeight       = 0.20
		wordWeight        = 0.15
		sentenceWeight    = 0.10
		punctuationWeight = 0.10
		longTokenWeight   = 0.08
		diversityWeight   = 0.07
		structureWeight   = 0.10
		constraintWeight  = 0.03
		reasoningWeight   = 0.02
	)

	if !hasTokenScore {
		// Redistribute token weight when BERT unavailable (avoids inaccurate estimation)
		lengthWeight = 0.19
		wordWeight = 0.19
		sentenceWeight = 0.12
		punctuationWeight = 0.12
		longTokenWeight = 0.10
		diversityWeight = 0.09
		structureWeight = 0.12
		constraintWeight = 0.04
		reasoningWeight = 0.03
		tokenWeight = 0.0 // Exclude token count
	}

	score := lengthScore*lengthWeight +
		tokenScore*tokenWeight +
		wordScore*wordWeight +
		sentenceScore*sentenceWeight +
		punctuationScore*punctuationWeight +
		longTokenScore*longTokenWeight +
		diversityScore*diversityWeight +
		structureScore*structureWeight +
		constraintScore*constraintWeight +
		reasoningScore*reasoningWeight

	// Ensure score is in valid range
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}

	return score
}

// calculateLengthScore calculates a normalized length score based on character count
// Normalization: 100 chars = 0.5, 200+ chars = 1.0
func (c *ComplexityClassifier) calculateLengthScore(textLength int) float64 {
	return math.Min(float64(textLength)/200.0, 1.0)
}

// calculateWordScore calculates a normalized word/token count score
// Normalization: 15 words = 0.5, 30+ words = 1.0
// For CJK languages, this uses character count instead
func (c *ComplexityClassifier) calculateWordScore(effectiveWordCount int) float64 {
	return math.Min(float64(effectiveWordCount)/30.0, 1.0)
}

// calculateSentenceScore calculates a normalized sentence count score
// Normalization: 2 sentences = 0.5, 4+ sentences = 1.0
func (c *ComplexityClassifier) calculateSentenceScore(text string) float64 {
	sentenceEnders := []rune{'.', '!', '?', '。', '！', '？', '。', '！', '؟'}
	sentenceCount := 0
	for _, char := range text {
		for _, ender := range sentenceEnders {
			if char == ender {
				sentenceCount++
				break
			}
		}
	}
	if sentenceCount == 0 {
		sentenceCount = 1
	}
	return math.Min(float64(sentenceCount)/4.0, 1.0)
}

// calculatePunctuationScore calculates a normalized punctuation score
// Normalization: 3 punctuation marks = 0.5, 6+ = 1.0
func (c *ComplexityClassifier) calculatePunctuationScore(text string) float64 {
	complexPunctuation := []rune{',', ';', ':', '(', ')', '（', '）', '，', '；', '：'}
	punctuationCount := 0
	for _, char := range text {
		for _, punct := range complexPunctuation {
			if char == punct {
				punctuationCount++
				break
			}
		}
	}
	return math.Min(float64(punctuationCount)/6.0, 1.0)
}

// calculateLongTokenScore calculates a normalized long token score
// Normalization: 2 long tokens = 0.5, 4+ = 1.0
func (c *ComplexityClassifier) calculateLongTokenScore(text string, words []string, hasCJK bool, wordCount int) float64 {
	longTokenCount := 0
	if hasCJK && wordCount == 1 {
		// For CJK languages, count long character sequences
		// A sequence of 4+ CJK characters is considered a "long token"
		cjkSequenceLength := 0
		for _, r := range text {
			if unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) {
				cjkSequenceLength++
			} else {
				if cjkSequenceLength >= 4 {
					longTokenCount++
				}
				cjkSequenceLength = 0
			}
		}
		if cjkSequenceLength >= 4 {
			longTokenCount++
		}
	} else {
		// For space-separated languages, use word-based detection
		for _, word := range words {
			cleanWord := strings.TrimFunc(word, func(r rune) bool {
				return !unicode.IsLetter(r) && !unicode.IsDigit(r) && !unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana)
			})
			// Use rune count for accurate length measurement
			if utf8.RuneCountInString(cleanWord) > 8 || (utf8.RuneCountInString(cleanWord) > 4 && unicode.In([]rune(cleanWord)[0], unicode.Han, unicode.Hiragana, unicode.Katakana)) {
				longTokenCount++
			}
		}
	}
	return math.Min(float64(longTokenCount)/4.0, 1.0)
}

// calculateDiversityScore calculates a normalized character diversity score
// Higher diversity ratio = more complex
func (c *ComplexityClassifier) calculateDiversityScore(text string, textLength int) float64 {
	uniqueChars := make(map[rune]bool)
	for _, char := range text {
		if unicode.IsLetter(char) || unicode.IsDigit(char) {
			uniqueChars[char] = true
		}
	}
	uniqueCount := len(uniqueChars)
	diversityRatio := float64(uniqueCount) / float64(textLength)
	return math.Min(diversityRatio*2.0, 1.0)
}

// calculateTokenScore calculates a normalized token count score
// Only uses actual tokenization when BERT is available (no fallback to estimation)
// Returns (score, hasTokenScore) where hasTokenScore=false if BERT unavailable
// Normalization: 75 tokens = 0.5, 150+ tokens = 1.0
func (c *ComplexityClassifier) calculateTokenScore(text string, textLength int, hasCJK bool) (float64, bool) {
	tokenCount, hasTokenCount := c.getTokenCount(text, textLength, hasCJK)
	if !hasTokenCount {
		return 0.0, false
	}
	return math.Min(float64(tokenCount)/150.0, 1.0), true
}

// getTokenCount gets the actual token count using tokenization when available
// Returns (tokenCount, hasTokenCount) where hasTokenCount=false if BERT unavailable
// We do NOT fall back to estimation to avoid inaccurate scores that could mislead routing
func (c *ComplexityClassifier) getTokenCount(text string, charCount int, hasCJK bool) (int, bool) {
	if charCount == 0 {
		return 0, false
	}

	// Use BERT tokenization if available (WordPiece, reasonable approximation for cost estimation)
	result, err := candle_binding.TokenizeTextDefault(text)
	if err == nil && result.TokenIDs != nil {
		// Count non-padding tokens (exclude [PAD] = 0)
		tokenCount := 0
		for _, tokenID := range result.TokenIDs {
			if tokenID != 0 {
				tokenCount++
			}
		}
		// If all tokens are non-zero, use full length
		if tokenCount == 0 {
			tokenCount = len(result.TokenIDs)
		}
		logging.Debugf("Complexity: Using BERT tokenization, count=%d (text length=%d)", tokenCount, charCount)
		return tokenCount, true
	}

	// BERT unavailable: exclude token count (matches codebase pattern of skipping when deps unavailable)
	logging.Debugf("Complexity: BERT tokenization unavailable, excluding token count from scoring (text length=%d)", charCount)
	return 0, false
}

// calculateStructureScore detects structural complexity: lists, JSON, code blocks, tables
// Returns a score (0.0-1.0) based on structural features
func (c *ComplexityClassifier) calculateStructureScore(text string) float64 {
	score := 0.0

	// Detect lists (numbered, bulleted, dash-based)
	listPattern := regexp.MustCompile(`(?m)^\s*(\d+[.)]|[*\-+])\s+`)
	listMatches := listPattern.FindAllString(text, -1)
	listCount := len(listMatches)
	if listCount > 0 {
		// 2 list items = 0.3, 5+ = 1.0
		listScore := math.Min(float64(listCount)/5.0, 1.0)
		score += listScore * 0.4
	}

	// Detect JSON-like structures (basic detection)
	jsonPattern := regexp.MustCompile(`\{[^{}]*"[^{}]*"[^{}]*\}`)
	jsonMatches := jsonPattern.FindAllString(text, -1)
	if len(jsonMatches) > 0 {
		score += 0.3
	}

	// Detect code blocks (backticks, code-like patterns)
	codeBlockPattern := regexp.MustCompile("```[\\s\\S]*?```|`[^`]+`")
	codeMatches := codeBlockPattern.FindAllString(text, -1)
	if len(codeMatches) > 0 {
		score += 0.3
	}

	// Detect table-like patterns (multiple pipes or tabs in lines)
	tablePattern := regexp.MustCompile(`(?m)^[^\n]*[|\t].*[|\t]`)
	tableMatches := tablePattern.FindAllString(text, -1)
	if len(tableMatches) >= 2 {
		score += 0.2
	}

	return math.Min(score, 1.0)
}

// calculateConstraintScore detects constraint markers: "must", "should", format requirements
// Returns a score (0.0-1.0) based on constraint indicators
func (c *ComplexityClassifier) calculateConstraintScore(text string) float64 {
	score := 0.0
	lowerText := strings.ToLower(text)

	// Constraint markers (language-agnostic structural patterns)
	constraintPatterns := []string{
		"must", "should", "need to", "required", "require",
		"format", "output", "limit", "constraint", "specify",
		"exactly", "precisely", "only", "not", "cannot",
	}

	constraintCount := 0
	for _, pattern := range constraintPatterns {
		if strings.Contains(lowerText, pattern) {
			constraintCount++
		}
	}

	// 2 constraints = 0.5, 4+ = 1.0
	if constraintCount > 0 {
		score = math.Min(float64(constraintCount)/4.0, 1.0)
	}

	// Detect bullet points or numbered requirements (structural constraint indicators)
	bulletPattern := regexp.MustCompile(`(?m)^\s*[\d\w][.)]\s+`)
	bulletMatches := bulletPattern.FindAllString(text, -1)
	if len(bulletMatches) >= 3 {
		score = math.Max(score, 0.5) // At least 0.5 if multiple requirements
	}

	return math.Min(score, 1.0)
}

// calculateReasoningScore detects reasoning indicators: "why", "how", "prove", multi-step instructions
// Returns a score (0.0-1.0) based on reasoning complexity
func (c *ComplexityClassifier) calculateReasoningScore(text string) float64 {
	score := 0.0
	lowerText := strings.ToLower(text)

	// Reasoning question markers
	reasoningQuestions := []string{
		"why", "how", "prove", "explain", "analyze", "compare",
		"contrast", "evaluate", "justify", "demonstrate", "derive",
	}

	reasoningCount := 0
	for _, marker := range reasoningQuestions {
		if strings.Contains(lowerText, marker) {
			reasoningCount++
		}
	}

	// 1 reasoning marker = 0.5, 2+ = 1.0
	if reasoningCount > 0 {
		score = math.Min(float64(reasoningCount)/2.0, 1.0)
	}

	// Detect multi-step instructions (connectors suggesting sequential steps)
	multiStepMarkers := []string{
		"then", "next", "after", "before", "first", "second", "finally",
		"step 1", "step 2", "step 3", "step one", "step two",
		"and then", "followed by", "subsequently",
	}

	stepCount := 0
	for _, marker := range multiStepMarkers {
		if strings.Contains(lowerText, marker) {
			stepCount++
		}
	}

	// Multi-step instructions increase reasoning complexity
	if stepCount >= 2 {
		score = math.Max(score, 0.7)
	}

	return math.Min(score, 1.0)
}
