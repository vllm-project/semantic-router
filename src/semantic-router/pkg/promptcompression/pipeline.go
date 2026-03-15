package promptcompression

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// WeightedScorer pairs a ScoringOptimizer with its pipeline weight.
type WeightedScorer struct {
	Name      string
	Optimizer ScoringOptimizer
	Weight    float64 // normalized so all weights in the pipeline sum to 1.0
}

// Pipeline is the fully assembled, runnable compression pipeline.
// Build one via ParsePipeline / LoadPipeline, or DefaultPipeline.
type Pipeline struct {
	MaxTokens      int
	PreserveFirstN int
	PreserveLastN  int
	OutputHeadroom int // subtracted from MaxTokens before selection

	PreProcessors []PreProcessor
	Scorers       []WeightedScorer
	Adjustors     []AdjustOptimizer
	Selectors     []SelectionOptimizer
}

// Message is a single turn in a multi-turn conversation.
// Used by CompressMessages to tag sentences with their source role.
type Message struct {
	Role    string // "system", "user", "assistant", "tool"
	Content string
}

// ── YAML schema ──────────────────────────────────────────────────────────────

type pipelineYAML struct {
	MaxTokens      int    `yaml:"max_tokens"`
	PreserveFirstN int    `yaml:"preserve_first_n"`
	PreserveLastN  int    `yaml:"preserve_last_n"`
	OutputHeadroom int    `yaml:"output_headroom"`
	Pipeline       struct {
		Pre     []optimizerEntryYAML `yaml:"pre"`
		Scoring []optimizerEntryYAML `yaml:"scoring"`
		Adjust  []optimizerEntryYAML `yaml:"adjust"`
		Select  []optimizerEntryYAML `yaml:"select"`
	} `yaml:"pipeline"`
}

type optimizerEntryYAML struct {
	Name   string         `yaml:"name"`
	Weight float64        `yaml:"weight"`
	Params map[string]any `yaml:"params"`
}

// ── Constructors ──────────────────────────────────────────────────────────────

// LoadPipeline reads a YAML file from disk and assembles a Pipeline.
func LoadPipeline(path string) (Pipeline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Pipeline{}, fmt.Errorf("promptcompression: reading %q: %w", path, err)
	}
	return ParsePipeline(data)
}

// ParsePipeline unmarshals YAML bytes and assembles a Pipeline from the
// registered optimizer factories.
func ParsePipeline(data []byte) (Pipeline, error) {
	var cfg pipelineYAML
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Pipeline{}, fmt.Errorf("promptcompression: parsing pipeline YAML: %w", err)
	}
	return buildPipeline(cfg)
}

func buildPipeline(cfg pipelineYAML) (Pipeline, error) {
	p := Pipeline{
		MaxTokens:      cfg.MaxTokens,
		PreserveFirstN: cfg.PreserveFirstN,
		PreserveLastN:  cfg.PreserveLastN,
		OutputHeadroom: cfg.OutputHeadroom,
	}

	for _, y := range cfg.Pipeline.Pre {
		pp, err := lookupPreProcessor(y.Name, y.Params)
		if err != nil {
			return Pipeline{}, err
		}
		p.PreProcessors = append(p.PreProcessors, pp)
	}

	var totalWeight float64
	for _, y := range cfg.Pipeline.Scoring {
		opt, err := lookupScoring(y.Name, y.Params)
		if err != nil {
			return Pipeline{}, err
		}
		p.Scorers = append(p.Scorers, WeightedScorer{Name: y.Name, Optimizer: opt, Weight: y.Weight})
		totalWeight += y.Weight
	}
	// Normalize so weights sum to 1.0 — caller can use arbitrary positive values.
	if totalWeight > 0 {
		for i := range p.Scorers {
			p.Scorers[i].Weight /= totalWeight
		}
	}

	for _, y := range cfg.Pipeline.Adjust {
		opt, err := lookupAdjust(y.Name, y.Params)
		if err != nil {
			return Pipeline{}, err
		}
		p.Adjustors = append(p.Adjustors, opt)
	}

	for _, y := range cfg.Pipeline.Select {
		opt, err := lookupSelection(y.Name, y.Params)
		if err != nil {
			return Pipeline{}, err
		}
		p.Selectors = append(p.Selectors, opt)
	}

	return p, nil
}

// DefaultPipeline returns a Pipeline equivalent to DefaultConfig.
// Useful when no external YAML is present.
func DefaultPipeline(maxTokens int) Pipeline {
	p, _ := ParsePipeline([]byte(fmt.Sprintf(`
max_tokens: %d
preserve_first_n: 3
preserve_last_n: 2
pipeline:
  scoring:
    - name: textrank
      weight: 0.20
    - name: position
      weight: 0.40
      params:
        depth: 0.5
    - name: tfidf
      weight: 0.35
    - name: novelty
      weight: 0.05
`, maxTokens)))
	return p
}

// ── Compression entry points ──────────────────────────────────────────────────

// CompressWithPipeline compresses text using the given Pipeline.
// It is the pipeline-driven equivalent of Compress.
func CompressWithPipeline(text string, p Pipeline) Result {
	originalTokens := CountTokensApprox(text)
	effectiveBudget := p.MaxTokens - p.OutputHeadroom
	if effectiveBudget <= 0 || originalTokens <= effectiveBudget {
		return Result{
			Compressed:       text,
			OriginalTokens:   originalTokens,
			CompressedTokens: originalTokens,
			Ratio:            1.0,
		}
	}

	sentences := SplitSentences(text)
	if len(sentences) <= 1 {
		return Result{
			Compressed:       text,
			OriginalTokens:   originalTokens,
			CompressedTokens: originalTokens,
			Ratio:            1.0,
		}
	}

	sentTokens, tfVecs := buildTokenData(sentences)

	// Pre-processing phase (e.g. deduplication).
	for _, pp := range p.PreProcessors {
		sentences, sentTokens, tfVecs = pp.Process(sentences, sentTokens, tfVecs)
	}

	// Cap sentence count to keep TextRank O(n²) manageable.
	if len(sentences) > maxSentences {
		sentences = sampleSentences(sentences, maxSentences)
		sentTokens, tfVecs = buildTokenData(sentences)
	}

	n := len(sentences)
	tokenCounts := make([]int, n)
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}

	ctx := OptimizerContext{
		Sentences:   sentences,
		SentTokens:  sentTokens,
		TFVecs:      tfVecs,
		TokenCounts: tokenCounts,
	}

	scored := runScoringPhase(ctx, p.Scorers, n)

	// Adjust phase: each adjuster modifies scored[i].Composite in-place.
	for _, adj := range p.Adjustors {
		adj.Adjust(scored, ctx)
	}

	// Selection phase: collect force-kept indices.
	forceKeep := make(map[int]bool)
	for _, sel := range p.Selectors {
		for _, idx := range sel.ForceKeep(ctx) {
			if idx >= 0 && idx < n {
				forceKeep[idx] = true
			}
		}
	}

	cfg := Config{
		MaxTokens:      effectiveBudget,
		PreserveFirstN: p.PreserveFirstN,
		PreserveLastN:  p.PreserveLastN,
	}
	kept := selectSentencesWithForce(scored, tokenCounts, cfg, forceKeep)
	sort.Ints(kept)

	parts := make([]string, 0, len(kept))
	compressedTokens := 0
	for _, idx := range kept {
		parts = append(parts, sentences[idx])
		compressedTokens += tokenCounts[idx]
	}

	ratio := 0.0
	if originalTokens > 0 {
		ratio = float64(compressedTokens) / float64(originalTokens)
	}
	return Result{
		Compressed:       strings.Join(parts, " "),
		OriginalTokens:   originalTokens,
		CompressedTokens: compressedTokens,
		Ratio:            ratio,
		SentenceScores:   scored,
		KeptIndices:      kept,
	}
}

// CompressMessages flattens a multi-turn conversation into text, tags each
// sentence with its source role, and compresses using the pipeline.
// Role metadata is passed to AdjustOptimizers via OptimizerContext.SentenceRoles,
// so a role_weight adjuster can apply per-role score multipliers without
// needing to modify the sentence text.
func CompressMessages(messages []Message, p Pipeline) Result {
	var parts []string
	var roleMap []string // role for each sentence after splitting

	for _, msg := range messages {
		sents := SplitSentences(msg.Content)
		for _, s := range sents {
			parts = append(parts, s)
			roleMap = append(roleMap, msg.Role)
		}
	}

	if len(parts) == 0 {
		return Result{}
	}

	// Build a synthetic text and run the standard pipeline, then inject roles
	// into the context just before the adjust phase.
	text := strings.Join(parts, " ")
	originalTokens := CountTokensApprox(text)
	effectiveBudget := p.MaxTokens - p.OutputHeadroom

	if effectiveBudget <= 0 || originalTokens <= effectiveBudget {
		return Result{
			Compressed:       text,
			OriginalTokens:   originalTokens,
			CompressedTokens: originalTokens,
			Ratio:            1.0,
		}
	}

	sentences := parts
	sentTokens, tfVecs := buildTokenData(sentences)

	for _, pp := range p.PreProcessors {
		sentences, sentTokens, tfVecs = pp.Process(sentences, sentTokens, tfVecs)
		// Trim roleMap to match after dedup
		if len(sentences) < len(roleMap) {
			roleMap = roleMap[:len(sentences)]
		}
	}

	if len(sentences) > maxSentences {
		sentences = sampleSentences(sentences, maxSentences)
		sentTokens, tfVecs = buildTokenData(sentences)
		if len(sentences) < len(roleMap) {
			roleMap = roleMap[:len(sentences)]
		}
	}

	n := len(sentences)
	tokenCounts := make([]int, n)
	for i, s := range sentences {
		tokenCounts[i] = CountTokensApprox(s)
	}

	ctx := OptimizerContext{
		Sentences:     sentences,
		SentTokens:    sentTokens,
		TFVecs:        tfVecs,
		TokenCounts:   tokenCounts,
		SentenceRoles: roleMap,
	}

	scored := runScoringPhase(ctx, p.Scorers, n)

	for _, adj := range p.Adjustors {
		adj.Adjust(scored, ctx)
	}

	forceKeep := make(map[int]bool)
	for _, sel := range p.Selectors {
		for _, idx := range sel.ForceKeep(ctx) {
			if idx >= 0 && idx < n {
				forceKeep[idx] = true
			}
		}
	}

	cfg := Config{
		MaxTokens:      effectiveBudget,
		PreserveFirstN: p.PreserveFirstN,
		PreserveLastN:  p.PreserveLastN,
	}
	kept := selectSentencesWithForce(scored, tokenCounts, cfg, forceKeep)
	sort.Ints(kept)

	out := make([]string, 0, len(kept))
	compressedTokens := 0
	for _, idx := range kept {
		out = append(out, sentences[idx])
		compressedTokens += tokenCounts[idx]
	}

	ratio := 0.0
	if originalTokens > 0 {
		ratio = float64(compressedTokens) / float64(originalTokens)
	}
	return Result{
		Compressed:       strings.Join(out, " "),
		OriginalTokens:   originalTokens,
		CompressedTokens: compressedTokens,
		Ratio:            ratio,
		SentenceScores:   scored,
		KeptIndices:      kept,
	}
}

// ── Shared helpers ────────────────────────────────────────────────────────────

// buildTokenData computes sentTokens and tfVecs from sentences.
// Extracted to avoid code duplication after pre-processing and re-sampling.
func buildTokenData(sentences []string) ([][]string, []map[string]float64) {
	n := len(sentences)
	sentTokens := make([][]string, n)
	tfVecs := make([]map[string]float64, n)
	for i, s := range sentences {
		toks := TokenizeWords(s)
		sentTokens[i] = toks
		tf := make(map[string]float64, len(toks))
		for _, t := range toks {
			tf[t]++
		}
		if cnt := float64(len(toks)); cnt > 0 {
			for k := range tf {
				tf[k] /= cnt
			}
		}
		tfVecs[i] = tf
	}
	return sentTokens, tfVecs
}

// runScoringPhase executes all weighted scorers and returns a ScoredSentence
// slice with Composite pre-filled and per-scorer scores stored in Scores.
func runScoringPhase(ctx OptimizerContext, scorers []WeightedScorer, n int) []ScoredSentence {
	scored := make([]ScoredSentence, n)
	for i := range ctx.Sentences {
		scored[i] = ScoredSentence{
			Index:  i,
			Text:   ctx.Sentences[i],
			Tokens: ctx.TokenCounts[i],
			Scores: make(map[string]float64, len(scorers)),
		}
	}

	for _, ws := range scorers {
		raw := ws.Optimizer.Score(ctx)
		if len(raw) != n {
			continue
		}
		normalizeSlice(raw)
		for i, v := range raw {
			scored[i].Scores[ws.Name] = v
			scored[i].Composite += ws.Weight * v
		}
	}
	return scored
}

// selectSentencesWithForce extends selectSentences with a forceKeep set.
// Forced sentences are reserved first (budget-permitting), then PreserveFirstN
// and PreserveLastN, then the greedy highest-scoring remainder.
func selectSentencesWithForce(
	scored []ScoredSentence,
	tokenCounts []int,
	cfg Config,
	forceKeep map[int]bool,
) []int {
	n := len(scored)
	kept := make([]bool, n)
	budget := cfg.MaxTokens
	usedTokens := 0
	keptCount := 0

	tryKeep := func(i int) {
		if i >= 0 && i < n && !kept[i] && usedTokens+tokenCounts[i] <= budget {
			kept[i] = true
			usedTokens += tokenCounts[i]
			keptCount++
		}
	}

	for i := range forceKeep {
		tryKeep(i)
	}
	for i := 0; i < cfg.PreserveFirstN && i < n; i++ {
		tryKeep(i)
	}
	for i := n - cfg.PreserveLastN; i < n; i++ {
		tryKeep(i)
	}

	type cand struct {
		index int
		score float64
	}
	candidates := make([]cand, 0, n-keptCount)
	for i := range scored {
		if !kept[i] {
			candidates = append(candidates, cand{i, scored[i].Composite})
		}
	}
	sort.Slice(candidates, func(a, b int) bool {
		return candidates[a].score > candidates[b].score
	})
	for _, c := range candidates {
		tryKeep(c.index)
	}

	result := make([]int, 0, keptCount)
	for i, ok := range kept {
		if ok {
			result = append(result, i)
		}
	}
	return result
}
