package classification

import (
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ComplexityClassifier performs complexity-based classification using embedding similarity.
// Each rule independently classifies difficulty level using hard/easy candidates.
// Supports both text candidates (via text embedding model) and image candidates
// (via the multimodal embedding model) for contrastive knowledge base comparison.
// Results are filtered by composer conditions in the classifier layer.
type ComplexityClassifier struct {
	rules []config.ComplexityRule

	// Precomputed text embeddings for hard and easy candidates
	hardEmbeddings     map[string]map[string][]float32 // ruleName -> candidate -> embedding
	easyEmbeddings     map[string]map[string][]float32 // ruleName -> candidate -> embedding
	hardPrototypeBanks map[string]*prototypeBank
	easyPrototypeBanks map[string]*prototypeBank

	// Precomputed image embeddings for hard and easy image candidates (multimodal)
	imageHardEmbeddings     map[string]map[string][]float32 // ruleName -> imageRef -> embedding
	imageEasyEmbeddings     map[string]map[string][]float32 // ruleName -> imageRef -> embedding
	imageHardPrototypeBanks map[string]*prototypeBank
	imageEasyPrototypeBanks map[string]*prototypeBank

	modelType          string // Model type for text embeddings ("qwen3" or "gemma")
	hasImageCandidates bool   // True if any rule uses image_candidates
	prototypeCfg       config.PrototypeScoringConfig
}

type ComplexityRuleResult struct {
	RuleName       string
	Difficulty     string
	TextHardScore  float64
	TextEasyScore  float64
	TextMargin     float64
	ImageHardScore float64
	ImageEasyScore float64
	ImageMargin    float64
	FusedMargin    float64
	Confidence     float64
	SignalSource   string
}

// NewComplexityClassifier creates a new ComplexityClassifier with precomputed candidate embeddings.
// When rules contain image_candidates, the multimodal model must be initialized beforehand.
func NewComplexityClassifier(
	rules []config.ComplexityRule,
	modelType string,
	prototypeCfg config.PrototypeScoringConfig,
) (*ComplexityClassifier, error) {
	if modelType == "" {
		modelType = "qwen3"
	}

	c := &ComplexityClassifier{
		rules:                   rules,
		hardEmbeddings:          make(map[string]map[string][]float32),
		easyEmbeddings:          make(map[string]map[string][]float32),
		hardPrototypeBanks:      make(map[string]*prototypeBank),
		easyPrototypeBanks:      make(map[string]*prototypeBank),
		imageHardEmbeddings:     make(map[string]map[string][]float32),
		imageEasyEmbeddings:     make(map[string]map[string][]float32),
		imageHardPrototypeBanks: make(map[string]*prototypeBank),
		imageEasyPrototypeBanks: make(map[string]*prototypeBank),
		modelType:               modelType,
		hasImageCandidates:      config.HasImageCandidatesInRules(rules),
		prototypeCfg:            prototypeCfg.WithDefaults(),
	}

	if c.hasImageCandidates {
		logging.Infof("ComplexityClassifier initialized with model type: %s + multimodal (image candidates detected)", c.modelType)
	} else {
		logging.Infof("ComplexityClassifier initialized with model type: %s", c.modelType)
	}

	if err := c.preloadCandidateEmbeddings(); err != nil {
		logging.Warnf("Failed to preload complexity candidate embeddings: %v", err)
		return nil, err
	}

	return c, nil
}

// preloadCandidateEmbeddings computes embeddings for all hard/easy candidates (text + image).
// Uses concurrent processing for better performance.
func (c *ComplexityClassifier) preloadCandidateEmbeddings() error {
	startTime := time.Now()
	logging.Infof("[Complexity Signal] Preloading embeddings for hard/easy candidates using model: %s with concurrent processing...", c.modelType)
	tasks := c.buildCandidateTasks()
	if len(tasks) == 0 {
		logging.Infof("[Complexity Signal] No candidates to preload")
		return nil
	}

	numWorkers := complexityWorkerCount(len(tasks))
	successCount, firstError := c.collectCandidateEmbeddingResults(c.startCandidateEmbeddingWorkers(tasks, numWorkers))

	elapsed := time.Since(startTime)
	logging.Infof("[Complexity Signal] Preloaded %d/%d complexity embeddings (text+image hard/easy candidates) using model %s in %v (workers: %d)",
		successCount, len(tasks), c.modelType, elapsed, numWorkers)

	if firstError != nil {
		return firstError
	}

	c.rebuildPrototypeBanks()

	return nil
}

// Classify evaluates the query against ALL complexity rules independently (text-only).
// For CUA requests with screenshots, use ClassifyWithImage instead.
func (c *ComplexityClassifier) Classify(query string) ([]string, error) {
	return c.ClassifyWithImage(query, "")
}

// ClassifyWithImage evaluates the query (and optionally a request image) against
// ALL complexity rules independently.
//
// When imageURL is provided (e.g. a base64 data-URI screenshot from a CUA request),
// SigLIP encodes the image and compares it against the image knowledge base.
// The text query is always compared against the text knowledge base.
// The difficulty score fuses both channels: d(t) = max(|d_vis|, |d_sem|).
//
// Returns: all matched rules in format "rulename:difficulty"
// (e.g., ["cua_difficulty:hard", "cua_difficulty:easy"])
func (c *ComplexityClassifier) ClassifyWithImage(query string, imageURL string) ([]string, error) {
	results, err := c.ClassifyDetailedWithImage(query, imageURL)
	if err != nil {
		return nil, err
	}
	matchedRules := make([]string, 0, len(results))
	for _, result := range results {
		matchedRules = append(matchedRules, fmt.Sprintf("%s:%s", result.RuleName, result.Difficulty))
	}
	return matchedRules, nil
}

func (c *ComplexityClassifier) ClassifyDetailedWithImage(query string, imageURL string) ([]ComplexityRuleResult, error) {
	if len(c.rules) == 0 {
		return nil, nil
	}

	queryEmbeddings, err := c.loadQueryEmbeddings(query, imageURL)
	if err != nil {
		return nil, err
	}
	scoreOptions := defaultPrototypeScoreOptions(c.prototypeCfg)
	results := make([]ComplexityRuleResult, 0, len(c.rules))
	for _, rule := range c.rules {
		result := c.classifyRuleWithEmbeddings(rule, queryEmbeddings, scoreOptions)
		logComplexityRuleResult(rule, result, queryEmbeddings.image != nil)
		results = append(results, result)
	}
	return results, nil
}

func (c *ComplexityClassifier) rebuildPrototypeBanks() {
	for _, rule := range c.rules {
		hardExamples := make([]prototypeExample, 0, len(c.hardEmbeddings[rule.Name]))
		for candidate, embedding := range c.hardEmbeddings[rule.Name] {
			hardExamples = append(hardExamples, prototypeExample{Key: rule.Name + ":hard:" + candidate, Text: candidate, Embedding: embedding})
		}
		easyExamples := make([]prototypeExample, 0, len(c.easyEmbeddings[rule.Name]))
		for candidate, embedding := range c.easyEmbeddings[rule.Name] {
			easyExamples = append(easyExamples, prototypeExample{Key: rule.Name + ":easy:" + candidate, Text: candidate, Embedding: embedding})
		}
		imageHardExamples := make([]prototypeExample, 0, len(c.imageHardEmbeddings[rule.Name]))
		for candidate, embedding := range c.imageHardEmbeddings[rule.Name] {
			imageHardExamples = append(imageHardExamples, prototypeExample{Key: rule.Name + ":image-hard:" + candidate, Text: candidate, Embedding: embedding})
		}
		imageEasyExamples := make([]prototypeExample, 0, len(c.imageEasyEmbeddings[rule.Name]))
		for candidate, embedding := range c.imageEasyEmbeddings[rule.Name] {
			imageEasyExamples = append(imageEasyExamples, prototypeExample{Key: rule.Name + ":image-easy:" + candidate, Text: candidate, Embedding: embedding})
		}
		hardBank := newPrototypeBank(hardExamples, c.prototypeCfg)
		easyBank := newPrototypeBank(easyExamples, c.prototypeCfg)
		imageHardBank := newPrototypeBank(imageHardExamples, c.prototypeCfg)
		imageEasyBank := newPrototypeBank(imageEasyExamples, c.prototypeCfg)
		c.hardPrototypeBanks[rule.Name] = hardBank
		c.easyPrototypeBanks[rule.Name] = easyBank
		c.imageHardPrototypeBanks[rule.Name] = imageHardBank
		c.imageEasyPrototypeBanks[rule.Name] = imageEasyBank
		logPrototypeBankSummary("Complexity hard", rule.Name, hardBank)
		logPrototypeBankSummary("Complexity easy", rule.Name, easyBank)
		if len(imageHardExamples) > 0 || len(imageEasyExamples) > 0 {
			logPrototypeBankSummary("Complexity image-hard", rule.Name, imageHardBank)
			logPrototypeBankSummary("Complexity image-easy", rule.Name, imageEasyBank)
		}
	}
}
