package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// categoryKBData holds the loaded exemplars for one category.
type categoryKBData struct {
	Exemplars  []string    `json:"exemplars"`
	Embeddings [][]float32 // pre-computed at init
}

// TaxonomyClassifyResult contains the output of taxonomy-backed classification.
type TaxonomyClassifyResult struct {
	BestCategory          string
	BestSimilarity        float64
	BestTier              string
	BestMatchedCategory   string
	BestMatchedSimilarity float64
	BestMatchedTier       string
	ContrastiveScore      float64
	MatchedCategories     []string
	MatchedTiers          []string
	CategoryTiers         map[string]string
	CategoryConfidences   map[string]float64
}

// CategoryKBClassifier performs contrastive per-category taxonomy
// classification. It loads JSON KB files from a directory (one per category),
// pre-embeds all exemplars at init, and at classify time computes max-cosine
// similarity per category plus a contrastive score between private and public
// tier groups.
type CategoryKBClassifier struct {
	rule      config.TaxonomyClassifierConfig
	taxonomy  config.TaxonomyDefinition
	kbs       map[string]*categoryKBData
	modelType string
	baseDir   string

	privateTiers map[string]bool // tiers treated as "private" for contrastive scoring
}

// NewCategoryKBClassifier creates a classifier from a taxonomy classifier config.
// It loads KB JSONs, taxonomy, and pre-embeds all exemplars.
func NewCategoryKBClassifier(rule config.TaxonomyClassifierConfig, modelType string, baseDir string) (*CategoryKBClassifier, error) {
	c := &CategoryKBClassifier{
		rule:         rule,
		kbs:          make(map[string]*categoryKBData),
		modelType:    modelType,
		baseDir:      baseDir,
		privateTiers: make(map[string]bool),
	}

	if err := c.loadTaxonomy(); err != nil {
		logging.Warnf("[TaxonomyClassifier:%s] No taxonomy loaded (%v), contrastive scoring will be disabled", rule.Name, err)
	} else {
		c.populatePrivateTiers()
	}

	if err := c.loadKBs(); err != nil {
		return nil, fmt.Errorf("failed to load taxonomy classifier assets from %s: %w", rule.Source.Path, err)
	}

	if err := c.preloadEmbeddings(); err != nil {
		return nil, fmt.Errorf("failed to preload taxonomy classifier embeddings: %w", err)
	}

	return c, nil
}

func (c *CategoryKBClassifier) loadTaxonomy() error {
	path := c.rule.Source.ResolveTaxonomyPath(c.baseDir)
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return config.UnmarshalTaxonomyDefinition(data, &c.taxonomy)
}

func (c *CategoryKBClassifier) loadKBs() error {
	root := c.rule.Source.ResolvePath(c.baseDir)
	entries, err := os.ReadDir(root)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if entry.Name() == c.rule.Source.ResolveTaxonomyBaseName() {
			continue
		}

		categoryName := strings.TrimSuffix(entry.Name(), ".json")
		data, err := os.ReadFile(filepath.Join(root, entry.Name()))
		if err != nil {
			logging.Warnf("[TaxonomyClassifier:%s] Failed to read %s: %v", c.rule.Name, entry.Name(), err)
			continue
		}

		var kb categoryKBData
		if err := config.UnmarshalTaxonomyExemplars(data, &kb); err != nil {
			logging.Warnf("[TaxonomyClassifier:%s] Failed to parse %s: %v", c.rule.Name, entry.Name(), err)
			continue
		}

		if len(kb.Exemplars) == 0 {
			logging.Warnf("[TaxonomyClassifier:%s] Skipping %s: no exemplars", c.rule.Name, categoryName)
			continue
		}

		c.kbs[categoryName] = &kb
	}

	if len(c.kbs) == 0 {
		return fmt.Errorf("no valid taxonomy classifier category files found in %s", root)
	}

	totalExemplars := 0
	for _, kb := range c.kbs {
		totalExemplars += len(kb.Exemplars)
	}
	logging.Infof("[TaxonomyClassifier:%s] Loaded %d categories with %d total exemplars from %s",
		c.rule.Name, len(c.kbs), totalExemplars, root)

	return nil
}

type exemplarRef struct {
	category string
	index    int
	text     string
}

type embeddingResult struct {
	ref       exemplarRef
	embedding []float32
	err       error
}

func (c *CategoryKBClassifier) collectExemplarRefs() []exemplarRef {
	var refs []exemplarRef
	for cat, kb := range c.kbs {
		kb.Embeddings = make([][]float32, len(kb.Exemplars))
		for i, text := range kb.Exemplars {
			refs = append(refs, exemplarRef{category: cat, index: i, text: text})
		}
	}
	return refs
}

func (c *CategoryKBClassifier) embedExemplarsParallel(refs []exemplarRef) <-chan embeddingResult {
	numWorkers := runtime.NumCPU() * 2
	if numWorkers > len(refs) {
		numWorkers = len(refs)
	}
	if numWorkers == 0 {
		numWorkers = 1
	}

	resultChan := make(chan embeddingResult, len(refs))
	refChan := make(chan exemplarRef, len(refs))
	for _, ref := range refs {
		refChan <- ref
	}
	close(refChan)

	modelType := c.modelType
	targetDim := 0 // use model default dimension for consistency with other classifiers

	var wg sync.WaitGroup
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ref := range refChan {
				output, err := getEmbeddingWithModelType(ref.text, modelType, targetDim)
				if err != nil {
					resultChan <- embeddingResult{ref: ref, err: err}
				} else {
					resultChan <- embeddingResult{ref: ref, embedding: output.Embedding}
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resultChan)
	}()

	return resultChan
}

func (c *CategoryKBClassifier) preloadEmbeddings() error {
	startTime := time.Now()
	refs := c.collectExemplarRefs()
	resultChan := c.embedExemplarsParallel(refs)

	var failCount int
	for res := range resultChan {
		if res.err != nil {
			failCount++
			logging.Warnf("[TaxonomyClassifier:%s] Failed to embed exemplar %q in %s: %v",
				c.rule.Name,
				res.ref.text, res.ref.category, res.err)
			continue
		}
		c.kbs[res.ref.category].Embeddings[res.ref.index] = res.embedding
	}

	if failCount > 0 {
		logging.Warnf("[TaxonomyClassifier:%s] %d/%d exemplar embeddings failed", c.rule.Name, failCount, len(refs))
	}

	logging.Infof("[TaxonomyClassifier:%s] Preloaded embeddings for %d exemplars across %d categories in %v",
		c.rule.Name, len(refs)-failCount, len(c.kbs), time.Since(startTime))

	return nil
}

func (c *CategoryKBClassifier) computeCategorySimilarities(queryEmb []float32) map[string]float64 {
	catMaxSim := make(map[string]float64, len(c.kbs))
	for catName, kb := range c.kbs {
		var maxSim float32
		for _, emb := range kb.Embeddings {
			if emb == nil {
				continue
			}
			if sim := cosineSimilarity(queryEmb, emb); sim > maxSim {
				maxSim = sim
			}
		}
		catMaxSim[catName] = float64(maxSim)
	}
	return catMaxSim
}

func (c *CategoryKBClassifier) buildMatchedRules(catMaxSim map[string]float64) ([]string, map[string]float64) {
	threshold := float64(c.rule.Threshold)
	secThreshold := float64(c.rule.SecurityThreshold)
	if secThreshold <= 0 {
		secThreshold = threshold
	}

	matchedRules := make([]string, 0)
	confidences := make(map[string]float64, len(catMaxSim))
	for cat, sim := range catMaxSim {
		confidences[cat] = sim
		appliedThreshold := threshold
		if c.isCategoryInTier(cat, "security_containment") {
			appliedThreshold = secThreshold
		}
		if sim >= appliedThreshold {
			matchedRules = append(matchedRules, cat)
		}
	}
	return matchedRules, confidences
}

func (c *CategoryKBClassifier) contrastiveScore(catMaxSim map[string]float64) float64 {
	var maxPrivate, maxPublic float64
	for cat, sim := range catMaxSim {
		if c.isCategoryPrivate(cat) {
			if sim > maxPrivate {
				maxPrivate = sim
			}
		} else if sim > maxPublic {
			maxPublic = sim
		}
	}
	return maxPrivate - maxPublic
}

// Classify computes per-category max-cosine-similarity, selects the best
// category, and produces a contrastive score (max private - max public).
func (c *CategoryKBClassifier) Classify(text string) (*TaxonomyClassifyResult, error) {
	if text == "" {
		return nil, fmt.Errorf("taxonomy classification: query must be provided")
	}

	startTime := time.Now()

	queryOutput, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute query embedding: %w", err)
	}

	catMaxSim := c.computeCategorySimilarities(queryOutput.Embedding)

	var bestCat string
	var bestSim float64
	for cat, sim := range catMaxSim {
		if sim > bestSim {
			bestSim = sim
			bestCat = cat
		}
	}

	matchedRules, confidences := c.buildMatchedRules(catMaxSim)
	contrastive := c.contrastiveScore(catMaxSim)
	categoryTiers := c.categoryTiers()

	bestTier := ""
	if entry, ok := c.taxonomy.Categories[bestCat]; ok {
		bestTier = entry.Tier
	}
	matchedTiers := c.collectMatchedTiers(matchedRules)
	bestMatchedCategory, bestMatchedSimilarity := bestMatchedCategory(matchedRules, confidences)
	bestMatchedTier := ""
	if entry, ok := c.taxonomy.Categories[bestMatchedCategory]; ok {
		bestMatchedTier = entry.Tier
	}

	elapsed := time.Since(startTime)
	logging.Infof("[TaxonomyClassifier:%s] Classified in %v: best=%s (%.3f) tier=%s, best_matched=%s (%.3f) tier=%s, contrastive=%.3f, matched=%d categories",
		c.rule.Name, elapsed, bestCat, bestSim, bestTier, bestMatchedCategory, bestMatchedSimilarity, bestMatchedTier, contrastive, len(matchedRules))

	return &TaxonomyClassifyResult{
		BestCategory:          bestCat,
		BestSimilarity:        bestSim,
		BestTier:              bestTier,
		BestMatchedCategory:   bestMatchedCategory,
		BestMatchedSimilarity: bestMatchedSimilarity,
		BestMatchedTier:       bestMatchedTier,
		ContrastiveScore:      contrastive,
		MatchedCategories:     matchedRules,
		MatchedTiers:          matchedTiers,
		CategoryTiers:         categoryTiers,
		CategoryConfidences:   confidences,
	}, nil
}

func (c *CategoryKBClassifier) isCategoryPrivate(category string) bool {
	entry, ok := c.taxonomy.Categories[category]
	if !ok {
		return false
	}
	return c.privateTiers[entry.Tier]
}

func (c *CategoryKBClassifier) isCategoryInTier(category, tier string) bool {
	entry, ok := c.taxonomy.Categories[category]
	if !ok {
		return false
	}
	return entry.Tier == tier
}

// CategoryCount returns the number of loaded categories.
func (c *CategoryKBClassifier) CategoryCount() int {
	return len(c.kbs)
}

func (c *CategoryKBClassifier) collectMatchedTiers(categories []string) []string {
	tierSet := make(map[string]struct{}, len(categories))
	matched := make([]string, 0, len(categories))
	for _, category := range categories {
		entry, ok := c.taxonomy.Categories[category]
		if !ok || entry.Tier == "" {
			continue
		}
		if _, exists := tierSet[entry.Tier]; exists {
			continue
		}
		tierSet[entry.Tier] = struct{}{}
		matched = append(matched, entry.Tier)
	}
	return matched
}

func bestMatchedCategory(categories []string, confidences map[string]float64) (string, float64) {
	bestCategory := ""
	bestConfidence := 0.0
	for _, category := range categories {
		confidence, ok := confidences[category]
		if !ok {
			continue
		}
		if confidence > bestConfidence {
			bestCategory = category
			bestConfidence = confidence
		}
	}
	return bestCategory, bestConfidence
}

func (c *CategoryKBClassifier) categoryTiers() map[string]string {
	tiers := make(map[string]string, len(c.taxonomy.Categories))
	for category, entry := range c.taxonomy.Categories {
		tiers[category] = entry.Tier
	}
	for category, tier := range c.taxonomy.CategoryToTier {
		if _, exists := tiers[category]; !exists {
			tiers[category] = tier
		}
	}
	return tiers
}

func (c *CategoryKBClassifier) populatePrivateTiers() {
	if c.taxonomy.TierGroups != nil {
		for _, groupName := range []string{"privacy_categories", "security_categories"} {
			for _, category := range c.taxonomy.TierGroups[groupName] {
				if entry, ok := c.taxonomy.Categories[category]; ok && entry.Tier != "" {
					c.privateTiers[entry.Tier] = true
				}
			}
		}
	}
	if len(c.privateTiers) > 0 {
		return
	}
	for _, tier := range []string{"security_containment", "privacy_policy"} {
		c.privateTiers[tier] = true
	}
}
