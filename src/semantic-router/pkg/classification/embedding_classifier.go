package classification

import (
	"encoding/base64"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// getEmbeddingWithModelType is a package-level variable for computing single embeddings.
// It exists so tests can override it.
var getEmbeddingWithModelType = candle_binding.GetEmbeddingWithModelType

// getMultiModalTextEmbedding computes a text embedding via the multimodal model.
// Package-level var so tests can override it.
var getMultiModalTextEmbedding = func(text string, targetDim int) ([]float32, error) {
	output, err := candle_binding.MultiModalEncodeText(text, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// getMultiModalImageEmbedding computes an image embedding from a base64-encoded
// image (raw base64 or data-URI) via the multimodal model.
// Also supports local file paths for preloading knowledge-base image candidates.
// Package-level var so tests can override it.
var getMultiModalImageEmbedding = func(imageRef string, targetDim int) ([]float32, error) {
	if imageRef == "" {
		return nil, fmt.Errorf("imageRef cannot be empty")
	}

	payload := imageRef

	// If imageRef is a local file path, read and base64-encode it
	if strings.HasPrefix(imageRef, "/") || strings.HasPrefix(imageRef, "./") {
		data, err := os.ReadFile(imageRef)
		if err != nil {
			return nil, fmt.Errorf("failed to read image file %q: %w", imageRef, err)
		}
		payload = base64.StdEncoding.EncodeToString(data)
	} else if idx := strings.Index(imageRef, ";base64,"); idx >= 0 {
		// Strip data-URI prefix if present (e.g. "data:image/png;base64,...")
		payload = imageRef[idx+len(";base64,"):]
	}

	output, err := candle_binding.MultiModalEncodeImageFromBase64(payload, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// initMultiModalModel is a package-level var for initializing the multimodal model.
var initMultiModalModel = candle_binding.InitMultiModalEmbeddingModel

// EmbeddingClassifierInitializer initializes KeywordEmbeddingClassifier for embedding based classification
type EmbeddingClassifierInitializer interface {
	Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error
}

type ExternalModelBasedEmbeddingInitializer struct{}

func (c *ExternalModelBasedEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool) error {
	// Resolve model paths using registry (supports aliases like "qwen3", "gemma", "mmbert")
	qwen3ModelPath = config.ResolveModelPath(qwen3ModelPath)
	gemmaModelPath = config.ResolveModelPath(gemmaModelPath)
	mmBertModelPath = config.ResolveModelPath(mmBertModelPath)

	err := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath, useCPU)
	if err != nil {
		return err
	}

	backend := "embedding_models"
	if mmBertModelPath != "" {
		backend = "mmbert_2d_matryoshka"
	}
	logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
		"backend":           backend,
		"qwen3_model_ref":   qwen3ModelPath,
		"gemma_model_ref":   gemmaModelPath,
		"mmbert_model_ref":  mmBertModelPath,
		"use_cpu":           useCPU,
		"mmbert_2d_enabled": mmBertModelPath != "",
	})
	return nil
}

// createEmbeddingInitializer creates the appropriate keyword embedding initializer based on configuration
func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &ExternalModelBasedEmbeddingInitializer{}
}

// EmbeddingClassifier performs embedding-based similarity classification.
// When preloading is enabled, candidate embeddings are computed once at initialization
// and reused for all classification requests, significantly improving performance.
type EmbeddingClassifier struct {
	rules []config.EmbeddingRule

	// rulesByModality is a precomputed lookup keyed by EffectiveQueryModality.
	// Rules do not mutate at runtime so this is populated once in
	// NewEmbeddingClassifier and shared by every classify call. Empty entries
	// for unconfigured modalities let the multimodal path early-return without
	// paying the FFI cost when no rule will fire.
	rulesByModality map[config.QueryModality][]config.EmbeddingRule

	// Optimization: preloaded candidate embeddings
	candidateEmbeddings map[string][]float32 // candidate text -> embedding vector
	rulePrototypeBanks  map[string]*prototypeBank

	// Configuration
	optimizationConfig config.HNSWConfig
	preloadEnabled     bool
	modelType          string // Model type to use for embeddings ("qwen3" or "gemma")
}

// NewEmbeddingClassifier creates a new EmbeddingClassifier.
// If optimization config has PreloadEmbeddings enabled, candidate embeddings
// will be precomputed at initialization time for better runtime performance.
func NewEmbeddingClassifier(cfgRules []config.EmbeddingRule, optConfig config.HNSWConfig) (*EmbeddingClassifier, error) {
	// Apply defaults
	optConfig = optConfig.WithDefaults()

	c := &EmbeddingClassifier{
		rules:               cfgRules,
		rulesByModality:     buildRulesByModality(cfgRules),
		candidateEmbeddings: make(map[string][]float32),
		rulePrototypeBanks:  make(map[string]*prototypeBank),
		optimizationConfig:  optConfig,
		preloadEnabled:      optConfig.PreloadEmbeddings,
		modelType:           optConfig.ModelType, // Use configured model type
	}

	logging.ComponentEvent("classifier", "embedding_classifier_initialized", map[string]interface{}{
		"model_type":          c.modelType,
		"rules":               len(cfgRules),
		"preload_embeddings":  optConfig.PreloadEmbeddings,
		"target_dimension":    optConfig.TargetDimension,
		"prototype_scoring":   optConfig.PrototypeScoring.IsEnabled(),
		"multimodal_prepared": optConfig.ModelType == "multimodal",
	})

	// If preloading is enabled, compute all candidate embeddings at startup
	if optConfig.PreloadEmbeddings {
		if err := c.preloadCandidateEmbeddings(); err != nil {
			// Log warning but don't fail - fall back to runtime computation
			logging.ComponentWarnEvent("classifier", "embedding_candidates_preload_failed", map[string]interface{}{
				"model_type":       c.modelType,
				"target_dimension": c.optimizationConfig.TargetDimension,
				"error":            err.Error(),
				"fallback":         "runtime_computation",
			})
			c.preloadEnabled = false
		}
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all unique candidates across all rules
// Uses concurrent processing for better performance
func (c *EmbeddingClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()
	candidates := c.collectUniqueCandidates()
	if len(candidates) == 0 {
		logging.ComponentDebugEvent("classifier", "embedding_candidates_preload_skipped", map[string]interface{}{
			"reason": "no_candidates",
		})
		return nil
	}

	modelType := c.getModelType()
	logging.ComponentDebugEvent("classifier", "embedding_candidates_preload_started", map[string]interface{}{
		"candidates":       len(candidates),
		"model_type":       modelType,
		"target_dimension": c.optimizationConfig.TargetDimension,
	})

	numWorkers := c.preloadWorkerCount(len(candidates))
	successCount, firstError := c.collectCandidateEmbeddingResults(
		c.startCandidateEmbeddingWorkers(candidates, modelType, numWorkers),
	)

	elapsed := time.Since(startTime)
	logging.ComponentEvent("classifier", "embedding_candidates_preloaded", map[string]interface{}{
		"candidates":       successCount,
		"total_candidates": len(candidates),
		"model_type":       modelType,
		"target_dimension": c.optimizationConfig.TargetDimension,
		"workers":          numWorkers,
		"elapsed_ms":       elapsed.Milliseconds(),
	})

	if firstError != nil {
		return firstError
	}

	c.rebuildRulePrototypeBanks()

	return nil
}

// getModelType returns the model type to use for embeddings
func (c *EmbeddingClassifier) getModelType() string {
	// Check for test override via environment variable
	if model := os.Getenv("EMBEDDING_MODEL_OVERRIDE"); model != "" {
		logging.ComponentDebugEvent("classifier", "embedding_model_override_enabled", map[string]interface{}{
			"model_type": model,
		})
		return model
	}
	// Use the configured model type from config
	// This ensures consistency between preload and runtime
	return c.modelType
}

// IsKeywordEmbeddingClassifierEnabled checks if Keyword embedding classification rules are properly configured
func (c *Classifier) IsKeywordEmbeddingClassifierEnabled() bool {
	return len(c.Config.EmbeddingRules) > 0
}

// initializeKeywordEmbeddingClassifier initializes the KeywordEmbedding classification model
func (c *Classifier) initializeKeywordEmbeddingClassifier() error {
	if !c.IsKeywordEmbeddingClassifierEnabled() || c.keywordEmbeddingInitializer == nil {
		return fmt.Errorf("keyword embedding similarity match is not properly configured")
	}

	modelType := strings.ToLower(strings.TrimSpace(c.Config.EmbeddingConfig.ModelType))
	if modelType == "multimodal" {
		mmPath := config.ResolveModelPath(c.Config.MultiModalModelPath)
		if mmPath == "" {
			return fmt.Errorf("embedding_rules with model_type=multimodal requires embedding_models.multimodal_model_path")
		}
		if err := initMultiModalModel(mmPath, c.Config.UseCPU); err != nil {
			return fmt.Errorf("failed to initialize multimodal model for embedding_rules: %w", err)
		}
		logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
			"backend":   "multimodal",
			"model_ref": mmPath,
			"use_cpu":   c.Config.UseCPU,
		})
		return nil
	}

	// Initialize with all three model paths (qwen3, gemma, mmbert)
	// The Init method will handle path resolution and choose the appropriate FFI function
	return c.keywordEmbeddingInitializer.Init(
		c.Config.Qwen3ModelPath,
		c.Config.GemmaModelPath,
		c.Config.MmBertModelPath,
		c.Config.UseCPU,
	)
}

// Classify performs Embedding similarity classification on the given text.
// Returns the single best matching rule. Wraps ClassifyAll internally.
func (c *EmbeddingClassifier) Classify(text string) (string, float64, error) {
	matched, err := c.ClassifyAll(text)
	if err != nil {
		return "", 0.0, err
	}
	if len(matched) == 0 {
		return "", 0.0, nil
	}
	best := matched[0]
	for _, m := range matched[1:] {
		if m.Score > best.Score {
			best = m
		}
	}
	return best.RuleName, best.Score, nil
}

// ClassifyAll performs embedding similarity classification on the given text.
// Returns the highest-ranking matched rules, limited by embedding_config.top_k
// (default 1, 0 disables truncation). When top_k is increased, the decision
// engine can compose multiple embedding matches together.
func (c *EmbeddingClassifier) ClassifyAll(text string) ([]MatchedRule, error) {
	result, err := c.ClassifyDetailed(text)
	if err != nil {
		return nil, err
	}
	return c.sortAndLimitMatches(result.Matches), nil
}

// ClassifyDetailed performs full label scoring on a TEXT query and returns
// the complete score distribution plus all accepted matches before top-k
// output shaping. Only rules whose effective QueryModality is "text"
// participate. For image/audio queries, use ClassifyDetailedMultimodal.
func (c *EmbeddingClassifier) ClassifyDetailed(text string) (*EmbeddingClassificationResult, error) {
	if len(c.rules) == 0 {
		return &EmbeddingClassificationResult{}, nil
	}

	// Validate input
	if text == "" {
		return nil, fmt.Errorf("embedding similarity classification: query must be provided")
	}

	startTime := time.Now()

	// Step 1: Filter to text-modality rules. Rules that target image or
	// audio queries do not participate in text classification: including
	// them would surface false positives where a text query happens to land
	// near an image-anchor set in the shared multimodal space. Done before
	// computing the embedding so a classifier with only image rules does
	// not pay the embedding cost when called on the text path.
	textRules := c.rulesByModality[config.QueryModalityText]
	if len(textRules) == 0 {
		logging.Infof("No embedding rules configured for text-modality queries (text rules: %d / total: %d)",
			0, len(c.rules))
		return &EmbeddingClassificationResult{}, nil
	}

	// Step 2: Compute query embedding once
	modelType := c.getModelType()
	queryOutput, err := getEmbeddingWithModelType(text, modelType, c.optimizationConfig.TargetDimension)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}
	queryEmbedding := queryOutput.Embedding

	logging.Infof("Computed query embedding (model: %s, dimension: %d)", modelType, len(queryEmbedding))

	// Step 3: Ensure candidate embeddings and prototype banks exist
	if ensureErr := c.ensureCandidateEmbeddings(); ensureErr != nil {
		return nil, ensureErr
	}

	// Step 4: Score the matching rules against their prototype banks
	scoredRules, err := c.scoreRulesSlice(queryEmbedding, textRules)
	if err != nil {
		return nil, err
	}
	matched := c.findAllMatchedRules(scoredRules)

	elapsed := time.Since(startTime)
	logging.Infof("ClassifyDetailed completed in %v: %d rules matched out of %d (modality=text)",
		elapsed, len(matched), len(textRules))

	return &EmbeddingClassificationResult{
		Scores:  scoredRules,
		Matches: matched,
	}, nil
}

// ClassifyDetailedMultimodal performs full label scoring on a non-text query
// (image or audio) against rules whose effective QueryModality matches.
// payload is a base64 string, a data-URI ("data:image/png;base64,..."), or
// a local file path; the same accepted forms as the underlying multimodal
// FFI helpers.
//
// Use this when the request carries an image or audio attachment that
// should be classified against text anchors in the shared multimodal space.
// For text queries, use ClassifyDetailed.
func (c *EmbeddingClassifier) ClassifyDetailedMultimodal(modality config.QueryModality, payload string) (*EmbeddingClassificationResult, error) {
	if len(c.rules) == 0 {
		return &EmbeddingClassificationResult{}, nil
	}
	if payload == "" {
		return nil, fmt.Errorf("embedding similarity classification: query payload must be provided")
	}

	effective := config.QueryModality(strings.ToLower(strings.TrimSpace(string(modality))))
	if effective == "" || effective == config.QueryModalityText {
		return nil, fmt.Errorf("ClassifyDetailedMultimodal: modality must be %q or %q (got %q); use ClassifyDetailed for text",
			config.QueryModalityImage, config.QueryModalityAudio, modality)
	}

	// Defense in depth: configs declaring query_modality=audio are rejected
	// at validateEmbeddingContracts time, so a classifier that successfully
	// constructed cannot have audio rules. This branch only fires when a Go
	// caller passes config.QueryModalityAudio programmatically. Returning a
	// clear error keeps that footgun explicit; the candle-binding
	// MultiModalEncodeAudioFromBase64 FFI is not yet exposed and this branch
	// will compute the audio embedding once it lands.
	if effective == config.QueryModalityAudio {
		return nil, fmt.Errorf("audio modality is not yet supported by ClassifyDetailedMultimodal; pass %q instead",
			config.QueryModalityImage)
	}
	if effective != config.QueryModalityImage {
		return nil, fmt.Errorf("unsupported query modality %q (supported: %q, %q)",
			modality, config.QueryModalityImage, config.QueryModalityAudio)
	}

	startTime := time.Now()

	// Step 1: Filter to rules whose query_modality matches the incoming
	// query. Done before computing the embedding so a request that no rule
	// would have matched does not pay the FFI cost. For high-volume image
	// gateways this is the difference between zero work and full embedding
	// for every "no rule will fire" request.
	matchingRules := c.rulesByModality[effective]
	if len(matchingRules) == 0 {
		logging.Infof("No embedding rules configured for modality=%s (total rules: %d)",
			effective, len(c.rules))
		return &EmbeddingClassificationResult{}, nil
	}

	// Step 2: Compute query embedding via the multimodal image path.
	queryEmbedding, err := getMultiModalImageEmbedding(payload, c.optimizationConfig.TargetDimension)
	if err != nil {
		return nil, fmt.Errorf("failed to compute multimodal query embedding (modality=%s): %w", effective, err)
	}

	logging.Infof("Computed multimodal query embedding (modality: %s, dimension: %d)",
		effective, len(queryEmbedding))

	// Step 3: Ensure candidate embeddings and prototype banks exist.
	if ensureErr := c.ensureCandidateEmbeddings(); ensureErr != nil {
		return nil, ensureErr
	}

	// Step 4: Score the matching rules.
	scoredRules, err := c.scoreRulesSlice(queryEmbedding, matchingRules)
	if err != nil {
		return nil, err
	}
	matched := c.findAllMatchedRules(scoredRules)

	elapsed := time.Since(startTime)
	logging.Infof("ClassifyDetailedMultimodal(%s) completed in %v: %d rules matched out of %d",
		effective, elapsed, len(matched), len(matchingRules))

	return &EmbeddingClassificationResult{
		Scores:  scoredRules,
		Matches: matched,
	}, nil
}

// buildRulesByModality groups rules by their effective query modality so
// the classifier can dispatch each request to the correct subset without
// per-call allocation. Rules with an unset modality are bucketed under
// QueryModalityText to preserve backward-compatible behavior.
func buildRulesByModality(rules []config.EmbeddingRule) map[config.QueryModality][]config.EmbeddingRule {
	byModality := make(map[config.QueryModality][]config.EmbeddingRule, 3)
	for _, rule := range rules {
		modality := rule.EffectiveQueryModality()
		byModality[modality] = append(byModality[modality], rule)
	}
	return byModality
}

// MatchedRule holds the result for a matched embedding rule
type MatchedRule struct {
	RuleName string
	Score    float64
	Method   string // "hard" or "soft"
}

type EmbeddingRuleScore struct {
	Name           string
	Score          float64
	Best           float64
	Support        float64
	Threshold      float64
	PrototypeCount int
}

type EmbeddingClassificationResult struct {
	Scores  []EmbeddingRuleScore
	Matches []MatchedRule
}

// findAllMatchedRules aggregates candidate similarities per rule and returns all
// accepted matches before final top-k output shaping.
func (c *EmbeddingClassifier) findAllMatchedRules(scoredRules []EmbeddingRuleScore) []MatchedRule {
	hardMatches := make([]MatchedRule, 0, len(scoredRules))

	// Phase 1: collect all hard matches (score >= rule threshold).
	for _, rule := range scoredRules {
		if rule.Score >= rule.Threshold {
			logging.Infof("Hard match found: rule=%q, score=%.4f", rule.Name, rule.Score)
			hardMatches = append(hardMatches, MatchedRule{
				RuleName: rule.Name,
				Score:    rule.Score,
				Method:   "hard",
			})
		}
	}

	if len(hardMatches) > 0 {
		return c.sortMatches(hardMatches)
	}

	// Phase 2: No hard matches — check if soft matching is enabled
	if c.optimizationConfig.EnableSoftMatching == nil || !*c.optimizationConfig.EnableSoftMatching {
		logging.Infof("No hard match found and soft matching is disabled")
		return nil
	}

	softMatches := make([]MatchedRule, 0, len(scoredRules))
	for _, rule := range scoredRules {
		if rule.Score >= float64(c.optimizationConfig.MinScoreThreshold) {
			logging.Infof("Soft match found: rule=%q, score=%.4f (min_threshold=%.3f)",
				rule.Name, rule.Score, c.optimizationConfig.MinScoreThreshold)
			softMatches = append(softMatches, MatchedRule{
				RuleName: rule.Name,
				Score:    rule.Score,
				Method:   "soft",
			})
		}
	}

	if len(softMatches) == 0 {
		logging.Infof("No match found (best score below min_threshold=%.3f)", c.optimizationConfig.MinScoreThreshold)
		return nil
	}

	return c.sortMatches(softMatches)
}

// scoreRulesSlice scores an explicit subset of rules against a query
// embedding. Used by both the text and multimodal classification paths
// after each filters c.rules down to the rules eligible for its modality.
//
// Precondition: callers must have already invoked ensureCandidateEmbeddings.
// The two public entry points (ClassifyDetailed, ClassifyDetailedMultimodal)
// own that contract, so this internal helper does not re-check on every call.
func (c *EmbeddingClassifier) scoreRulesSlice(queryEmbedding []float32, rules []config.EmbeddingRule) ([]EmbeddingRuleScore, error) {
	scoredRules := make([]EmbeddingRuleScore, 0, len(rules))
	for _, rule := range rules {
		bank, ok := c.rulePrototypeBanks[rule.Name]
		if !ok || bank == nil || len(bank.prototypes) == 0 {
			continue
		}

		bankScore := bank.score(queryEmbedding, c.embeddingAggregationOptions(rule))
		logging.Infof("Rule %q: score=%.4f best=%.4f support=%.4f threshold=%.3f matched=%v (prototypes=%d)",
			rule.Name, bankScore.Score, bankScore.Best, bankScore.Support, rule.SimilarityThreshold,
			bankScore.Score >= float64(rule.SimilarityThreshold), bankScore.PrototypeCount)

		scoredRules = append(scoredRules, EmbeddingRuleScore{
			Name:           rule.Name,
			Score:          bankScore.Score,
			Best:           bankScore.Best,
			Support:        bankScore.Support,
			Threshold:      float64(rule.SimilarityThreshold),
			PrototypeCount: bankScore.PrototypeCount,
		})
	}
	return scoredRules, nil
}

func (c *EmbeddingClassifier) sortMatches(matches []MatchedRule) []MatchedRule {
	sort.Slice(matches, func(i, j int) bool {
		if matches[i].Score == matches[j].Score {
			return matches[i].RuleName < matches[j].RuleName
		}
		return matches[i].Score > matches[j].Score
	})
	return matches
}

func (c *EmbeddingClassifier) sortAndLimitMatches(matches []MatchedRule) []MatchedRule {
	matches = c.sortMatches(matches)
	topK := 1
	if c.optimizationConfig.TopK != nil {
		topK = *c.optimizationConfig.TopK
	}
	if topK == 0 || len(matches) <= topK {
		return matches
	}

	logging.Infof("Embedding matches limited to top_k=%d (available=%d)", topK, len(matches))
	return matches[:topK]
}

func (c *EmbeddingClassifier) embeddingAggregationOptions(rule config.EmbeddingRule) prototypeScoreOptions {
	switch rule.AggregationMethodConfiged {
	case config.AggregationMethodMean:
		return prototypeScoreOptions{BestWeight: 0, TopM: 0}
	default:
		// "max"/"any" keep their rule-level acceptance semantics, but the
		// underlying bank score now uses the shared prototype-aware defaults
		// instead of collapsing back to a single winning prototype.
		return defaultPrototypeScoreOptions(c.optimizationConfig.PrototypeScoring)
	}
}

// cosineSimilarity computes cosine similarity between two vectors.
// Assumes vectors are normalized (which they should be from BERT-style models).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var dotProduct float32
	for i := 0; i < minLen; i++ {
		dotProduct += a[i] * b[i]
	}

	return dotProduct
}

// GetPreloadStats returns statistics about preloaded embeddings
func (c *EmbeddingClassifier) GetPreloadStats() int {
	return len(c.candidateEmbeddings)
}

func (c *EmbeddingClassifier) rebuildRulePrototypeBanks() {
	c.rulePrototypeBanks = make(map[string]*prototypeBank, len(c.rules))
	prototypeCfg := c.optimizationConfig.PrototypeScoring.WithDefaults()
	for _, rule := range c.rules {
		examples := make([]prototypeExample, 0, len(rule.Candidates))
		for _, candidate := range rule.Candidates {
			embedding, ok := c.candidateEmbeddings[candidate]
			if !ok || len(embedding) == 0 {
				continue
			}
			examples = append(examples, prototypeExample{
				Key:       candidate,
				Text:      candidate,
				Embedding: embedding,
			})
		}
		bank := newPrototypeBank(examples, prototypeCfg)
		c.rulePrototypeBanks[rule.Name] = bank
		logPrototypeBankSummary("Embedding Signal", rule.Name, bank)
	}
}
