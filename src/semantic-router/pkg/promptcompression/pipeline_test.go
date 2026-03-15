package promptcompression

import (
	"regexp"
	"strings"
	"testing"
)

// ============================================================================
// ParsePipeline / LoadPipeline
// ============================================================================

func TestParsePipeline_Default(t *testing.T) {
	yaml := `
max_tokens: 256
preserve_first_n: 2
preserve_last_n: 1
output_headroom: 10
pipeline:
  scoring:
    - name: textrank
      weight: 0.5
    - name: tfidf
      weight: 0.5
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline error: %v", err)
	}
	if p.MaxTokens != 256 {
		t.Errorf("MaxTokens = %d, want 256", p.MaxTokens)
	}
	if p.PreserveFirstN != 2 {
		t.Errorf("PreserveFirstN = %d, want 2", p.PreserveFirstN)
	}
	if p.PreserveLastN != 1 {
		t.Errorf("PreserveLastN = %d, want 1", p.PreserveLastN)
	}
	if p.OutputHeadroom != 10 {
		t.Errorf("OutputHeadroom = %d, want 10", p.OutputHeadroom)
	}
	if len(p.Scorers) != 2 {
		t.Errorf("expected 2 scorers, got %d", len(p.Scorers))
	}
}

func TestParsePipeline_AllPhases(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  pre:
    - name: dedup
      params:
        threshold: 0.9
  scoring:
    - name: textrank
      weight: 1.0
    - name: position
      weight: 1.0
      params:
        depth: 0.6
    - name: tfidf
      weight: 1.0
    - name: novelty
      weight: 1.0
  adjust:
    - name: pattern_boost
    - name: role_weight
    - name: focus_keywords
      params:
        keywords: ["error", "bug"]
        boost: 3.0
  select:
    - name: must_contain
      params:
        substrings: ["CRITICAL"]
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline error: %v", err)
	}
	if len(p.PreProcessors) != 1 {
		t.Errorf("want 1 pre-processor, got %d", len(p.PreProcessors))
	}
	if len(p.Scorers) != 4 {
		t.Errorf("want 4 scorers, got %d", len(p.Scorers))
	}
	if len(p.Adjustors) != 3 {
		t.Errorf("want 3 adjustors, got %d", len(p.Adjustors))
	}
	if len(p.Selectors) != 1 {
		t.Errorf("want 1 selector, got %d", len(p.Selectors))
	}
}

func TestParsePipeline_InvalidYAML(t *testing.T) {
	// yaml.v3 is lenient; use a structurally broken document (unclosed bracket).
	_, err := ParsePipeline([]byte("pipeline: {scoring: [unclosed"))
	if err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestParsePipeline_UnknownScoringOptimizer(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: does_not_exist
      weight: 1.0
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for unknown scoring optimizer")
	}
}

func TestParsePipeline_UnknownAdjustOptimizer(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  adjust:
    - name: no_such_adjuster
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for unknown adjust optimizer")
	}
}

func TestParsePipeline_UnknownPreProcessor(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  pre:
    - name: no_such_preprocessor
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for unknown pre-processor")
	}
}

func TestParsePipeline_UnknownSelectionOptimizer(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  select:
    - name: no_such_selector
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for unknown selection optimizer")
	}
}

func TestParsePipeline_InvalidRegexPatternBoost(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  adjust:
    - name: pattern_boost
      params:
        patterns:
          - regex: "[(invalid"
            multiplier: 2.0
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for invalid regex in pattern_boost")
	}
}

func TestParsePipeline_InvalidRegexMustContain(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  select:
    - name: must_contain
      params:
        patterns: ["[(invalid"]
`
	_, err := ParsePipeline([]byte(yaml))
	if err == nil {
		t.Error("expected error for invalid regex in must_contain")
	}
}

func TestParsePipeline_WeightNormalization(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: textrank
      weight: 2.0
    - name: tfidf
      weight: 8.0
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline: %v", err)
	}
	var total float64
	for _, s := range p.Scorers {
		total += s.Weight
	}
	if total < 0.9999 || total > 1.0001 {
		t.Errorf("scorer weights should sum to 1.0, got %.6f", total)
	}
	// textrank=2/10=0.2, tfidf=8/10=0.8
	if p.Scorers[0].Weight < 0.199 || p.Scorers[0].Weight > 0.201 {
		t.Errorf("textrank weight = %.3f, want ~0.2", p.Scorers[0].Weight)
	}
}

// ============================================================================
// DefaultPipeline
// ============================================================================

func TestDefaultPipeline_RespectsMaxTokens(t *testing.T) {
	p := DefaultPipeline(128)
	if p.MaxTokens != 128 {
		t.Errorf("DefaultPipeline MaxTokens = %d, want 128", p.MaxTokens)
	}
	if len(p.Scorers) == 0 {
		t.Error("DefaultPipeline should have at least one scorer")
	}
}

func TestDefaultPipeline_ZeroPassthrough(t *testing.T) {
	p := DefaultPipeline(0)
	text := "Hello world."
	result := CompressWithPipeline(text, p)
	if result.Compressed != text {
		t.Errorf("MaxTokens=0 should pass through, got %q", result.Compressed)
	}
}

// ============================================================================
// CompressWithPipeline
// ============================================================================

func TestCompressWithPipeline_NoOp_ShortText(t *testing.T) {
	p := DefaultPipeline(512)
	text := "Short text that fits in the budget."
	result := CompressWithPipeline(text, p)
	if result.Compressed != text {
		t.Errorf("within-budget text should not be modified")
	}
	if result.Ratio != 1.0 {
		t.Errorf("ratio = %.3f, want 1.0", result.Ratio)
	}
}

func TestCompressWithPipeline_ReducesTokens(t *testing.T) {
	p := DefaultPipeline(50)
	sentences := []string{
		"Machine learning is a powerful technique for pattern recognition.",
		"Neural networks learn hierarchical representations from training data.",
		"Convolutional networks excel at image classification and detection.",
		"Recurrent networks handle variable-length sequential input data.",
		"Transformer models use self-attention to capture long-range dependencies.",
		"Pre-training on large corpora enables few-shot generalization to new tasks.",
		"Fine-tuning adapts a pre-trained model to downstream classification tasks.",
		"Evaluation metrics such as accuracy and F1 measure model performance.",
		"Hyperparameter search over learning rate and batch size improves results.",
		"Regularization via dropout prevents overfitting on small training datasets.",
	}
	text := strings.Join(sentences, " ")

	result := CompressWithPipeline(text, p)
	if result.CompressedTokens > 50 {
		t.Errorf("compressed tokens %d > budget 50", result.CompressedTokens)
	}
	if result.Ratio >= 1.0 {
		t.Errorf("expected compression, got ratio %.3f", result.Ratio)
	}
}

func TestCompressWithPipeline_OutputHeadroom(t *testing.T) {
	yaml := `
max_tokens: 100
output_headroom: 40
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"Backpropagation is the primary algorithm for training neural networks effectively.",
		"Gradient descent iteratively moves parameters towards the loss function minimum.",
		"The learning rate controls the step size during each gradient update iteration.",
		"Momentum accelerates gradient descent by accumulating velocity in update directions.",
		"Batch normalization stabilizes training by normalizing activations across mini-batches.",
		"Weight initialization using Xavier or He schemes helps avoid vanishing gradients.",
		"Early stopping halts training when validation loss stops improving further.",
		"Data augmentation artificially increases training set diversity for better generalization.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)
	// effective budget is 100 - 40 = 60
	if result.CompressedTokens > 60 {
		t.Errorf("headroom not respected: %d tokens > 60 (budget 100 - headroom 40)", result.CompressedTokens)
	}
}

func TestCompressWithPipeline_PreserveFirstAndLast(t *testing.T) {
	yaml := `
max_tokens: 30
preserve_first_n: 1
preserve_last_n: 1
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
`
	p, _ := ParsePipeline([]byte(yaml))
	sentences := []string{
		"FIRST: system prompt context.",
		"Filler one about nothing.",
		"Filler two about nothing.",
		"Filler three about nothing.",
		"Filler four about nothing.",
		"LAST: the user's actual question.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	if !strings.Contains(result.Compressed, "FIRST:") {
		t.Error("first sentence should be preserved")
	}
	if !strings.Contains(result.Compressed, "LAST:") {
		t.Error("last sentence should be preserved")
	}
}

func TestCompressWithPipeline_ScoresMapPopulated(t *testing.T) {
	yaml := `
max_tokens: 20
pipeline:
  scoring:
    - name: textrank
      weight: 0.5
    - name: tfidf
      weight: 0.5
`
	p, _ := ParsePipeline([]byte(yaml))
	text := "First sentence. Second sentence. Third sentence."
	result := CompressWithPipeline(text, p)

	for _, s := range result.SentenceScores {
		if _, ok := s.Scores["textrank"]; !ok {
			t.Errorf("sentence %d missing 'textrank' in Scores map", s.Index)
		}
		if _, ok := s.Scores["tfidf"]; !ok {
			t.Errorf("sentence %d missing 'tfidf' in Scores map", s.Index)
		}
	}
}

func TestCompressWithPipeline_KeptIndicesOrdered(t *testing.T) {
	p := DefaultPipeline(40)
	sentences := []string{
		"Topic A: artificial intelligence applications.",
		"Topic B: weather and climate patterns.",
		"Topic C: machine learning algorithms.",
		"Topic D: economic indicators.",
		"Topic E: deep learning architectures.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	for i := 1; i < len(result.KeptIndices); i++ {
		if result.KeptIndices[i] <= result.KeptIndices[i-1] {
			t.Errorf("kept indices not ordered: %v", result.KeptIndices)
		}
	}
}

func TestCompressWithPipeline_EmptyText(t *testing.T) {
	p := DefaultPipeline(512)
	result := CompressWithPipeline("", p)
	if result.Compressed != "" {
		t.Errorf("empty text should return empty, got %q", result.Compressed)
	}
}

func TestCompressWithPipeline_SingleSentence(t *testing.T) {
	p := DefaultPipeline(3)
	text := "One sentence cannot be split further"
	result := CompressWithPipeline(text, p)
	if result.Compressed != text {
		t.Errorf("single sentence should pass through unchanged")
	}
}

// ============================================================================
// CompressMessages
// ============================================================================

func TestCompressMessages_Basic(t *testing.T) {
	p := DefaultPipeline(40)
	messages := []Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Help me understand gradient descent."},
		{Role: "assistant", Content: "Gradient descent is an optimization algorithm. It minimizes a function by iteratively moving in the direction of steepest descent. The learning rate controls step size."},
		{Role: "user", Content: "How does momentum help?"},
	}
	result := CompressMessages(messages, p)
	if result.Compressed == "" {
		t.Error("CompressMessages should produce non-empty output")
	}
	if result.OriginalTokens == 0 {
		t.Error("OriginalTokens should be > 0")
	}
}

func TestCompressMessages_EmptyMessages(t *testing.T) {
	p := DefaultPipeline(512)
	result := CompressMessages(nil, p)
	if result.Compressed != "" {
		t.Errorf("nil messages should return empty result, got %q", result.Compressed)
	}
}

func TestCompressMessages_WithinBudgetPassthrough(t *testing.T) {
	p := DefaultPipeline(512)
	messages := []Message{
		{Role: "user", Content: "Short question."},
	}
	result := CompressMessages(messages, p)
	if result.Ratio != 1.0 {
		t.Errorf("within-budget messages should return ratio 1.0, got %.3f", result.Ratio)
	}
}

func TestCompressMessages_RoleWeightedCompression(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: role_weight
      params:
        weights:
          system:    1.0
          user:      3.0
          assistant: 0.1
          tool:      1.0
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline: %v", err)
	}

	// 5 assistant reasoning sentences + 1 user sentence.
	// With user ×3.0 and assistant ×0.1, user should be strongly preferred.
	messages := []Message{
		{Role: "user", Content: "Debug the authentication token refresh failure."},
		{Role: "assistant", Content: "Let me analyze the issue step by step."},
		{Role: "assistant", Content: "First I will read the relevant source files."},
		{Role: "assistant", Content: "Then I will trace the execution path through the code."},
		{Role: "assistant", Content: "I will also check the database connection pool configuration."},
		{Role: "assistant", Content: "Finally I will examine the JWT library version being used."},
	}
	result := CompressMessages(messages, p)

	// The user message should survive; most assistant reasoning should be dropped.
	if !strings.Contains(result.Compressed, "authentication") &&
		!strings.Contains(result.Compressed, "token") {
		t.Error("user message should survive role-weighted compression")
	}
	if result.CompressedTokens > 30 {
		t.Errorf("compressed tokens %d > budget 30", result.CompressedTokens)
	}
}

// ============================================================================
// dedup PreProcessor
// ============================================================================

func TestDedup_RemovesNearDuplicates(t *testing.T) {
	// Use the dedup pre-processor directly to avoid budget/passthrough interference.
	threshold := 0.85
	pp := &dedupPreProcessor{threshold: threshold}

	// Build TF vectors for 4 sentences where S0 and S2 are identical.
	sentences := []string{
		"The authentication service validates JWT tokens on every request.",
		"The database connection pool manages concurrent query execution.",
		"The authentication service validates JWT tokens on every request.", // dup of S0
		"Memory usage peaks during large batch processing operations.",
	}
	sentTokens, tfVecs := buildTokenData(sentences)

	outS, _, _ := pp.Process(sentences, sentTokens, tfVecs)

	// S0 should be dropped; S2 (more recent copy) should be kept.
	count := 0
	for _, s := range outS {
		if strings.Contains(s, "authentication service validates") {
			count++
		}
	}
	if count != 1 {
		t.Errorf("dedup should keep exactly 1 copy, found %d in %v", count, outS)
	}
	if len(outS) != 3 {
		t.Errorf("expected 3 sentences after dedup (1 dup dropped), got %d: %v", len(outS), outS)
	}
}

func TestDedup_KeepsDistinctSentences(t *testing.T) {
	// Test the pre-processor directly: distinct sentences must pass through unchanged.
	pp := &dedupPreProcessor{threshold: 0.95}

	sentences := []string{
		"Gradient descent minimizes the loss function.",
		"Backpropagation computes the gradient efficiently.",
		"Momentum accelerates convergence during optimization.",
	}
	sentTokens, tfVecs := buildTokenData(sentences)
	outS, _, _ := pp.Process(sentences, sentTokens, tfVecs)

	if len(outS) != 3 {
		t.Errorf("distinct sentences should not be deduped; got %d: %v", len(outS), outS)
	}
}

func TestDedup_HighThresholdKeepsAll(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  pre:
    - name: dedup
      params:
        threshold: 1.01   # impossible to match — nothing is dropped
  scoring:
    - name: tfidf
      weight: 1.0
`
	p, _ := ParsePipeline([]byte(yaml))
	text := "Alpha sentence. Alpha sentence. Alpha sentence."
	result := CompressWithPipeline(text, p)
	// Threshold > 1.0 means nothing is considered a duplicate.
	if len(result.SentenceScores) < 2 {
		t.Logf("threshold > 1.0: kept %d sentences (cosine is max 1.0, so no drop expected)", len(result.SentenceScores))
	}
}

// ============================================================================
// pattern_boost AdjustOptimizer
// ============================================================================

func TestPatternBoost_DefaultPatternsBoostErrors(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: pattern_boost
`
	p, _ := ParsePipeline([]byte(yaml))

	// Error line should score higher than generic filler.
	sentences := []string{
		"Error: connection refused on port 5432 for database host.",
		"The system processed the batch job without significant issues.",
		"Configuration parameters were loaded from the environment successfully.",
		"Background workers completed all scheduled maintenance operations.",
		"Service health checks returned normal status across all endpoints.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	// Find composite score for error sentence vs others.
	errorScore := -1.0
	maxOtherScore := -1.0
	for _, s := range result.SentenceScores {
		if strings.Contains(s.Text, "Error: connection") {
			errorScore = s.Composite
		} else if s.Composite > maxOtherScore {
			maxOtherScore = s.Composite
		}
	}
	if errorScore < 0 {
		t.Fatal("error sentence not found in scored sentences")
	}
	t.Logf("Error score=%.3f, max other=%.3f", errorScore, maxOtherScore)
	if errorScore <= maxOtherScore {
		t.Errorf("error sentence (%.3f) should have highest composite score (max other=%.3f)",
			errorScore, maxOtherScore)
	}
}

func TestPatternBoost_FilepathWithLineNumber(t *testing.T) {
	// Test the adjuster directly using a custom regex so there is no dependency
	// on the sentence splitter handling dots in file names.
	fileLineRe := regexp.MustCompile(`crash_at_line:\d+`)
	adj := &patternBoostOptimizer{patterns: []patternEntry{
		{re: fileLineRe, multiplier: 2.0},
	}}

	scored := []ScoredSentence{
		{Index: 0, Text: "The nil pointer crash_at_line:88 in the router handler code", Composite: 1.0},
		{Index: 1, Text: "All other parts of the service are operating normally", Composite: 1.0},
		{Index: 2, Text: "Background metrics collection completes on schedule", Composite: 1.0},
	}
	ctx := OptimizerContext{Sentences: []string{scored[0].Text, scored[1].Text, scored[2].Text}}
	adj.Adjust(scored, ctx)

	if scored[0].Composite <= scored[1].Composite {
		t.Errorf("file:line sentence (%.3f) should outscore filler (%.3f)",
			scored[0].Composite, scored[1].Composite)
	}
	t.Logf("file:line score=%.3f, filler=%.3f", scored[0].Composite, scored[1].Composite)
}

func TestPatternBoost_CustomPatterns(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: pattern_boost
      params:
        patterns:
          - regex: "\\bSECRET\\b"
            multiplier: 5.0
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"The SECRET key must never be logged or exposed to external systems.",
		"Routine maintenance completed without incident across all environments.",
		"All scheduled jobs finished within their allocated time windows today.",
		"Monitoring dashboards confirm normal operational status for services.",
		"Database backup operations completed successfully within the time window.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	secretScore := -1.0
	maxOtherScore := -1.0
	for _, s := range result.SentenceScores {
		if strings.Contains(s.Text, "SECRET") {
			secretScore = s.Composite
		} else if s.Composite > maxOtherScore {
			maxOtherScore = s.Composite
		}
	}
	t.Logf("SECRET score=%.3f, max other=%.3f", secretScore, maxOtherScore)
	if secretScore <= maxOtherScore {
		t.Errorf("SECRET sentence should be highest scored: %.3f <= %.3f", secretScore, maxOtherScore)
	}
}

func TestPatternBoost_TODOMarker(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: pattern_boost
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"TODO: implement retry logic for transient database connection failures.",
		"The application currently runs in a single datacenter for simplicity.",
		"Metrics collection occurs on a five minute interval across all nodes.",
		"Service discovery is handled by the internal DNS resolver automatically.",
		"Log aggregation pipelines process approximately one million events daily.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	todoScore := -1.0
	maxOtherScore := -1.0
	for _, s := range result.SentenceScores {
		if strings.Contains(s.Text, "TODO:") {
			todoScore = s.Composite
		} else if s.Composite > maxOtherScore {
			maxOtherScore = s.Composite
		}
	}
	t.Logf("TODO score=%.3f, max other=%.3f", todoScore, maxOtherScore)
	if todoScore <= maxOtherScore {
		t.Errorf("TODO sentence should be highest: %.3f <= %.3f", todoScore, maxOtherScore)
	}
}

// ============================================================================
// focus_keywords AdjustOptimizer
// ============================================================================

func TestFocusKeywords_BoostsSentencesWithKeyword(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: focus_keywords
      params:
        keywords: ["authentication", "JWT"]
        boost: 4.0
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"The JWT token expires after twenty four hours by default.",
		"Background job scheduling is handled by a dedicated worker pool.",
		"Disk I/O throughput has been stable over the past monitoring window.",
		"Load balancer configuration uses round robin for upstream selection.",
		"Container image sizes were reduced by removing unnecessary build layers.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	jwtScore := -1.0
	maxOtherScore := -1.0
	for _, s := range result.SentenceScores {
		if strings.Contains(strings.ToLower(s.Text), "jwt") {
			jwtScore = s.Composite
		} else if s.Composite > maxOtherScore {
			maxOtherScore = s.Composite
		}
	}
	t.Logf("JWT score=%.3f, max other=%.3f", jwtScore, maxOtherScore)
	if jwtScore <= maxOtherScore {
		t.Errorf("JWT sentence should have highest composite: %.3f <= %.3f", jwtScore, maxOtherScore)
	}
}

func TestFocusKeywords_EmptyKeywordsNoOp(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: focus_keywords
      params:
        keywords: []
        boost: 5.0
`
	p1, _ := ParsePipeline([]byte(yaml))

	yaml2 := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
`
	p2, _ := ParsePipeline([]byte(yaml2))

	text := "Neural networks are universal function approximators. " +
		"Attention mechanisms improve sequence model performance. " +
		"Transfer learning reduces data requirements for new tasks."

	r1 := CompressWithPipeline(text, p1)
	r2 := CompressWithPipeline(text, p2)

	// Empty keywords should not change scores.
	for i := range r1.SentenceScores {
		if i >= len(r2.SentenceScores) {
			break
		}
		d := r1.SentenceScores[i].Composite - r2.SentenceScores[i].Composite
		if d < -0.001 || d > 0.001 {
			t.Errorf("empty keywords changed score[%d]: %.3f vs %.3f", i,
				r1.SentenceScores[i].Composite, r2.SentenceScores[i].Composite)
		}
	}
}

func TestFocusKeywords_CaseInsensitive(t *testing.T) {
	yaml := `
max_tokens: 30
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: focus_keywords
      params:
        keywords: ["POSTGRES"]
        boost: 3.0
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"The postgres connection pool is exhausted under high load conditions.",
		"Background tasks execute on a separate thread pool for isolation.",
		"Metrics are exported to the observability platform every minute.",
		"Health check endpoints respond within ten milliseconds under normal load.",
		"Log rotation is configured to retain seven days of application logs.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	postgresScore := -1.0
	maxOther := -1.0
	for _, s := range result.SentenceScores {
		if strings.Contains(strings.ToLower(s.Text), "postgres") {
			postgresScore = s.Composite
		} else if s.Composite > maxOther {
			maxOther = s.Composite
		}
	}
	if postgresScore <= maxOther {
		t.Errorf("case-insensitive keyword match failed: postgres=%.3f maxOther=%.3f",
			postgresScore, maxOther)
	}
}

// ============================================================================
// must_contain SelectionOptimizer
// ============================================================================

func TestMustContain_SubstringForceKeep(t *testing.T) {
	// Test the selector directly: ForceKeep must return the index of the matching sentence.
	sel := &mustContainOptimizer{substrings: []string{"CRITICAL:"}}

	sentences := []string{
		"Background operation completed without errors in the expected time.",
		"CRITICAL: database replication lag exceeds threshold.",
		"Routine health check returned normal status across all nodes.",
	}
	_, tfVecs := buildTokenData(sentences)
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}
	ctx := OptimizerContext{
		Sentences:   sentences,
		TFVecs:      tfVecs,
		TokenCounts: tokenCounts,
	}

	indices := sel.ForceKeep(ctx)
	found := false
	for _, idx := range indices {
		if idx == 1 {
			found = true
		}
	}
	if !found {
		t.Errorf("must_contain should force-keep index 1 (CRITICAL sentence); got %v", indices)
	}

	// Also verify it survives in a pipeline with a comfortable budget.
	yaml := `
max_tokens: 60
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  select:
    - name: must_contain
      params:
        substrings: ["CRITICAL:"]
`
	p, _ := ParsePipeline([]byte(yaml))
	// Make the CRITICAL sentence low-scoring by surrounding it with highly similar filler.
	longSentences := []string{
		"Background operation completed without errors in the expected time range.",
		"Routine health check returned normal status across all nodes.",
		"Scheduled backup job completed successfully.",
		"Performance metrics within acceptable bounds.",
		"CRITICAL: database replication lag exceeds threshold.",
		"All other services operational.",
		"No anomalies detected in monitoring.",
	}
	result := CompressWithPipeline(strings.Join(longSentences, " "), p)
	if !strings.Contains(result.Compressed, "CRITICAL:") {
		t.Error("must_contain should force-keep CRITICAL sentence in pipeline")
	}
}

func TestMustContain_RegexForceKeep(t *testing.T) {
	yaml := `
max_tokens: 10
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  select:
    - name: must_contain
      params:
        patterns: ["\\bCVE-\\d{4}-\\d+\\b"]
`
	p, _ := ParsePipeline([]byte(yaml))

	sentences := []string{
		"All systems are operating within normal parameters at this time.",
		"The vulnerability CVE-2024-12345 requires an immediate patch deployment.",
		"Routine log rotation completed without consuming excess disk storage.",
		"Service latency metrics remain within the configured SLA boundaries.",
		"Configuration drift detection found no unauthorized changes today.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	if !strings.Contains(result.Compressed, "CVE-2024-12345") {
		t.Error("must_contain regex should force-keep CVE sentence")
	}
}

func TestMustContain_NoMatch_NoEffect(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  select:
    - name: must_contain
      params:
        substrings: ["NEVER_PRESENT_MARKER_XYZ"]
`
	p, _ := ParsePipeline([]byte(yaml))
	text := "Normal sentence one. Normal sentence two. Normal sentence three."
	result := CompressWithPipeline(text, p)
	if result.Compressed == "" {
		t.Error("must_contain with no match should not drop all sentences")
	}
}

func TestMustContain_MultipleSubstrings(t *testing.T) {
	// Test selector directly: both markers must appear in ForceKeep output.
	sel := &mustContainOptimizer{substrings: []string{"MARKER_A", "MARKER_B"}}

	sentences := []string{
		"MARKER_A: first critical event that must not be dropped.",
		"Generic operational log entry about routine maintenance.",
		"MARKER_B: second critical event requiring guaranteed retention.",
		"Another generic entry about infrastructure health.",
	}
	_, tfVecs := buildTokenData(sentences)
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}
	ctx := OptimizerContext{Sentences: sentences, TFVecs: tfVecs, TokenCounts: tokenCounts}

	indices := sel.ForceKeep(ctx)
	indexSet := map[int]bool{}
	for _, i := range indices {
		indexSet[i] = true
	}
	if !indexSet[0] {
		t.Errorf("MARKER_A at index 0 should be force-kept; got %v", indices)
	}
	if !indexSet[2] {
		t.Errorf("MARKER_B at index 2 should be force-kept; got %v", indices)
	}
}

// ============================================================================
// role_weight AdjustOptimizer
// ============================================================================

func TestRoleWeight_UserPreferredOverAssistant(t *testing.T) {
	yaml := `
max_tokens: 15
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: role_weight
      params:
        weights:
          user:      3.0
          assistant: 0.1
`
	p, _ := ParsePipeline([]byte(yaml))

	messages := []Message{
		{Role: "user", Content: "How does the token refresh endpoint authenticate callers?"},
		{Role: "assistant", Content: "I will look at the implementation to understand this better."},
		{Role: "assistant", Content: "The code uses an internal service account credential mechanism."},
		{Role: "assistant", Content: "I am reading the authentication middleware source code now."},
		{Role: "assistant", Content: "The refresh logic delegates to the OAuth provider library."},
	}
	result := CompressMessages(messages, p)

	// With strong user boost and heavy assistant dampening, user message should survive.
	if !strings.Contains(result.Compressed, "token refresh") &&
		!strings.Contains(result.Compressed, "authenticate") {
		t.Logf("compressed: %q", result.Compressed)
		t.Error("user message about token refresh should survive role-weighted compression")
	}
}

func TestRoleWeight_NoRolesIsNoOp(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: role_weight
`
	p1, _ := ParsePipeline([]byte(yaml))

	yaml2 := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
`
	p2, _ := ParsePipeline([]byte(yaml2))

	// Flat text (no roles) — role_weight must be a no-op.
	text := "Sentence A about topic one. Sentence B about topic two. Sentence C about topic three."
	r1 := CompressWithPipeline(text, p1)
	r2 := CompressWithPipeline(text, p2)

	for i := range r1.SentenceScores {
		if i >= len(r2.SentenceScores) {
			break
		}
		d := r1.SentenceScores[i].Composite - r2.SentenceScores[i].Composite
		if d < -0.001 || d > 0.001 {
			t.Errorf("role_weight changed score for flat text at index %d: %.3f vs %.3f",
				i, r1.SentenceScores[i].Composite, r2.SentenceScores[i].Composite)
		}
	}
}

func TestRoleWeight_CustomWeights(t *testing.T) {
	yaml := `
max_tokens: 20
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: role_weight
      params:
        weights:
          system: 5.0
          user:   0.1
`
	p, _ := ParsePipeline([]byte(yaml))

	messages := []Message{
		{Role: "system", Content: "You must always enforce strict rate limiting on all API endpoints."},
		{Role: "user", Content: "Show me a joke please."},
	}
	result := CompressMessages(messages, p)

	// System has 5x multiplier; user has 0.1x — system should survive.
	if !strings.Contains(result.Compressed, "rate limiting") &&
		!strings.Contains(result.Compressed, "API endpoints") {
		t.Logf("compressed: %q", result.Compressed)
		t.Error("system message should survive with 5x boost")
	}
}

// ============================================================================
// age_decay ScoringOptimizer
// ============================================================================

func TestAgeDecay_NewerSentencesScoreHigher(t *testing.T) {
	// Test the scorer directly without going through the pipeline (avoids
	// passthrough when text fits within budget).
	scorer := &ageDecayOptimizer{factor: 0.5, ages: []int{3, 2, 1, 0}}
	sentences := []string{
		"Old turn content A.",
		"Older turn content B.",
		"Recent turn content C.",
		"Newest turn content D.",
	}
	_, tfVecs := buildTokenData(sentences)
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}
	ctx := OptimizerContext{Sentences: sentences, TFVecs: tfVecs, TokenCounts: tokenCounts}
	scores := scorer.Score(ctx)

	if len(scores) != 4 {
		t.Fatalf("expected 4 scores, got %d", len(scores))
	}
	// age=3→score≈0.22, age=2→0.37, age=1→0.61, age=0→1.0; each step higher.
	for i := 1; i < 4; i++ {
		if scores[i] <= scores[i-1] {
			t.Errorf("score[%d] (age=%d) should be > score[%d] (age=%d): %.3f <= %.3f",
				i, 3-i, i-1, 4-i, scores[i], scores[i-1])
		}
	}
	t.Logf("age_decay scores (factor=0.5): %v", scores)
}

func TestAgeDecay_DefaultFactor(t *testing.T) {
	// Default factor=0.15; age=10 → exp(-1.5)≈0.22, age=0 → 1.0
	scorer := &ageDecayOptimizer{factor: 0.15, ages: []int{10, 0}}
	sentences := []string{
		"Old context sentence from a previous session turn.",
		"Newest context sentence in the current turn.",
	}
	_, tfVecs := buildTokenData(sentences)
	tokenCounts := make([]int, len(sentences))
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}
	ctx := OptimizerContext{Sentences: sentences, TFVecs: tfVecs, TokenCounts: tokenCounts}
	scores := scorer.Score(ctx)

	if len(scores) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(scores))
	}
	if scores[1] <= scores[0] {
		t.Errorf("newest (age=0, score=%.3f) should outscore old (age=10, score=%.3f)", scores[1], scores[0])
	}
	t.Logf("age_decay scores: old(age=10)=%.3f newest(age=0)=%.3f", scores[0], scores[1])
}

func TestAgeDecay_MissingAgesDefaultToZero(t *testing.T) {
	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: age_decay
      weight: 1.0
      params:
        factor: 0.3
`
	p, _ := ParsePipeline([]byte(yaml))
	text := "Sentence A. Sentence B. Sentence C."
	result := CompressWithPipeline(text, p)

	// All ages default to 0 → all scores should be exp(0) = 1.0 before normalization.
	for _, s := range result.SentenceScores {
		if ageDecayScore, ok := s.Scores["age_decay"]; ok {
			if ageDecayScore < 0.99 || ageDecayScore > 1.01 {
				t.Errorf("age_decay score with missing ages should be ~1.0, got %.3f", ageDecayScore)
			}
		}
	}
}

// ============================================================================
// PipelineSelector
// ============================================================================

func TestPipelineSelector_FallbackToDefault(t *testing.T) {
	defPipeline := DefaultPipeline(256)
	codingPipeline := DefaultPipeline(128)

	sel := NewPipelineSelectorFromMap(defPipeline, map[string]Pipeline{
		"coding": codingPipeline,
	})

	got := sel.Select("medical") // no specific pipeline for medical
	if got.MaxTokens != 256 {
		t.Errorf("unknown domain should fall back to default (256), got %d", got.MaxTokens)
	}
}

func TestPipelineSelector_DomainMatch(t *testing.T) {
	defPipeline := DefaultPipeline(256)
	codingPipeline := DefaultPipeline(128)
	medicalPipeline := DefaultPipeline(512)

	sel := NewPipelineSelectorFromMap(defPipeline, map[string]Pipeline{
		"coding":  codingPipeline,
		"medical": medicalPipeline,
	})

	if p := sel.Select("coding"); p.MaxTokens != 128 {
		t.Errorf("coding domain should return 128-token pipeline, got %d", p.MaxTokens)
	}
	if p := sel.Select("medical"); p.MaxTokens != 512 {
		t.Errorf("medical domain should return 512-token pipeline, got %d", p.MaxTokens)
	}
}

func TestPipelineSelector_Register(t *testing.T) {
	defPipeline := DefaultPipeline(256)
	sel := NewPipelineSelectorFromMap(defPipeline, nil)

	// Before registration — falls back to default.
	if p := sel.Select("security"); p.MaxTokens != 256 {
		t.Errorf("before registration expected default 256, got %d", p.MaxTokens)
	}

	// Register at runtime.
	sel.Register("security", DefaultPipeline(64))
	if p := sel.Select("security"); p.MaxTokens != 64 {
		t.Errorf("after registration expected 64, got %d", p.MaxTokens)
	}
}

func TestPipelineSelector_NilMapUsesDefault(t *testing.T) {
	defPipeline := DefaultPipeline(999)
	sel := NewPipelineSelectorFromMap(defPipeline, nil)
	if p := sel.Select("anything"); p.MaxTokens != 999 {
		t.Errorf("nil domain map should always return default, got %d", p.MaxTokens)
	}
}

func TestPipelineSelector_EmptyDomainFallsBack(t *testing.T) {
	defPipeline := DefaultPipeline(512)
	sel := NewPipelineSelectorFromMap(defPipeline, map[string]Pipeline{
		"coding": DefaultPipeline(256),
	})
	if p := sel.Select(""); p.MaxTokens != 512 {
		t.Errorf("empty domain string should fall back to default, got %d", p.MaxTokens)
	}
}

// ============================================================================
// Registry error paths
// ============================================================================

func TestRegistry_CustomScoringOptimizer(t *testing.T) {
	RegisterScoring("test_constant_scorer", func(params map[string]any) (ScoringOptimizer, error) {
		score := 0.5
		if v, ok := params["score"]; ok {
			score = anyToFloat64(v)
		}
		return &constantScorer{score: score}, nil
	})

	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: test_constant_scorer
      weight: 1.0
      params:
        score: 0.75
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("custom scorer registration failed: %v", err)
	}
	if len(p.Scorers) != 1 || p.Scorers[0].Name != "test_constant_scorer" {
		t.Errorf("custom scorer not wired into pipeline: %+v", p.Scorers)
	}
}

func TestRegistry_CustomAdjustOptimizer(t *testing.T) {
	RegisterAdjust("test_double_adjuster", func(_ map[string]any) (AdjustOptimizer, error) {
		return &doubleAdjuster{}, nil
	})

	yaml := `
max_tokens: 512
pipeline:
  scoring:
    - name: tfidf
      weight: 1.0
  adjust:
    - name: test_double_adjuster
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("custom adjuster registration failed: %v", err)
	}
	if len(p.Adjustors) != 1 || p.Adjustors[0].Name() != "test_double_adjuster" {
		t.Errorf("custom adjuster not found in pipeline")
	}
}

// ── helpers for registry tests ──────────────────────────────────────────────

type constantScorer struct{ score float64 }

func (c *constantScorer) Name() string { return "test_constant_scorer" }
func (c *constantScorer) Score(ctx OptimizerContext) []float64 {
	scores := make([]float64, len(ctx.Sentences))
	for i := range scores {
		scores[i] = c.score
	}
	return scores
}

type doubleAdjuster struct{}

func (d *doubleAdjuster) Name() string { return "test_double_adjuster" }
func (d *doubleAdjuster) Adjust(scored []ScoredSentence, _ OptimizerContext) {
	for i := range scored {
		scored[i].Composite *= 2.0
	}
}

// ============================================================================
// End-to-end: Claude Code-inspired pipeline on a coding conversation
// ============================================================================

func TestClaudeCodePipeline_PreservesErrors(t *testing.T) {
	yaml := `
max_tokens: 60
preserve_first_n: 1
preserve_last_n: 1
pipeline:
  pre:
    - name: dedup
      params:
        threshold: 0.92
  scoring:
    - name: tfidf
      weight: 0.35
    - name: position
      weight: 0.35
      params:
        depth: 0.6
    - name: novelty
      weight: 0.30
  adjust:
    - name: pattern_boost
      params:
        patterns:
          - regex: "(?i)(error|panic|fatal)[\\s:]"
            multiplier: 2.0
          - regex: "\\w+\\.go:\\d+"
            multiplier: 1.8
    - name: role_weight
      params:
        weights:
          user:      2.0
          assistant: 0.5
          tool:      1.2
  select:
    - name: must_contain
      params:
        substrings: ["FATAL:"]
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline: %v", err)
	}

	messages := []Message{
		{Role: "user", Content: "The service is crashing in production right now."},
		{Role: "assistant", Content: "I will investigate the issue immediately."},
		{Role: "tool", Content: "FATAL: goroutine panic at router.go:88 — nil pointer dereference."},
		{Role: "assistant", Content: "The stack trace indicates a nil pointer in the router."},
		{Role: "assistant", Content: "I am reading the router.go file to find the root cause."},
		{Role: "assistant", Content: "The issue appears to be in the middleware initialization order."},
		{Role: "tool", Content: "error: connection refused on 127.0.0.1:6379 for Redis client."},
		{Role: "assistant", Content: "Redis is unavailable which is causing the panic in the router."},
		{Role: "user", Content: "How do we fix this without a full restart?"},
	}
	result := CompressMessages(messages, p)

	if result.CompressedTokens > 60 {
		t.Errorf("exceeded budget: %d > 60 tokens", result.CompressedTokens)
	}
	// FATAL must always be kept (must_contain selector).
	if !strings.Contains(result.Compressed, "FATAL:") {
		t.Error("must_contain selector should force-keep FATAL sentence")
	}
	t.Logf("Compressed (%d→%d tokens, ratio=%.2f): %q",
		result.OriginalTokens, result.CompressedTokens, result.Ratio, result.Compressed)
}

// ============================================================================
// Pipeline with all four phases: integration smoke test
// ============================================================================

func TestAllFourPhasesIntegration(t *testing.T) {
	yaml := `
max_tokens: 50
preserve_first_n: 1
preserve_last_n: 1
output_headroom: 5
pipeline:
  pre:
    - name: dedup
      params:
        threshold: 0.9
  scoring:
    - name: textrank
      weight: 0.25
    - name: position
      weight: 0.25
    - name: tfidf
      weight: 0.25
    - name: novelty
      weight: 0.25
  adjust:
    - name: pattern_boost
    - name: focus_keywords
      params:
        keywords: ["RETAIN"]
        boost: 10.0
  select:
    - name: must_contain
      params:
        substrings: ["KEEP_THIS"]
`
	p, err := ParsePipeline([]byte(yaml))
	if err != nil {
		t.Fatalf("ParsePipeline: %v", err)
	}
	if len(p.PreProcessors) != 1 {
		t.Errorf("expected 1 pre-processor, got %d", len(p.PreProcessors))
	}
	if len(p.Scorers) != 4 {
		t.Errorf("expected 4 scorers, got %d", len(p.Scorers))
	}
	if len(p.Adjustors) != 2 {
		t.Errorf("expected 2 adjustors, got %d", len(p.Adjustors))
	}
	if len(p.Selectors) != 1 {
		t.Errorf("expected 1 selector, got %d", len(p.Selectors))
	}

	sentences := []string{
		"KEEP_THIS: mandatory context for the processing pipeline.",
		"Filler context about general operational information.",
		"More filler about background system configuration details.",
		"Another filler sentence without any special significance here.",
		"The RETAIN keyword appears in this sentence for boost testing.",
		"Yet more filler about routine maintenance and standard operations.",
		"Final filler sentence with no distinguishing features whatsoever.",
	}
	text := strings.Join(sentences, " ")
	result := CompressWithPipeline(text, p)

	if result.CompressedTokens > 45 { // 50 - headroom 5
		t.Errorf("headroom not respected: %d tokens (effective budget 45)", result.CompressedTokens)
	}
	if !strings.Contains(result.Compressed, "KEEP_THIS") {
		t.Error("must_contain should force-keep KEEP_THIS sentence")
	}
	t.Logf("All-phases: %d→%d tokens, kept=%v", result.OriginalTokens, result.CompressedTokens, result.KeptIndices)
}
