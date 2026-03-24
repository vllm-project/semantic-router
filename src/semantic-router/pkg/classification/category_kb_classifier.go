package classification

import (
	"encoding/json"
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

// CategoryKBTaxonomy maps category names to routing tier groups.
type CategoryKBTaxonomy struct {
	Categories map[string]CategoryKBTaxonomyEntry `json:"categories"`
	TierGroups map[string][]string                `json:"tier_groups"`
}

// CategoryKBTaxonomyEntry defines routing metadata for a single category.
type CategoryKBTaxonomyEntry struct {
	Tier        string `json:"tier"`
	Description string `json:"description,omitempty"`
}

// categoryKBData holds the loaded exemplars for one category.
type categoryKBData struct {
	Exemplars  []string    `json:"exemplars"`
	Embeddings [][]float32 // pre-computed at init
}

// CategoryKBClassifyResult contains the output of per-category KB classification.
type CategoryKBClassifyResult struct {
	BestCategory     string
	BestSimilarity   float64
	BestTier         string   // taxonomy tier of the best-matching category (e.g. "security_containment")
	ContrastiveScore float64  // max(private) - max(public)
	MatchedRules     []string // category names that exceeded the applicable threshold
	Confidences      map[string]float64
}

// CategoryKBClassifier performs contrastive per-category knowledge base
// classification. It loads JSON KB files from a directory (one per category),
// pre-embeds all exemplars at init, and at classify time computes max-cosine
// similarity per category plus a contrastive score between private and public
// tier groups.
type CategoryKBClassifier struct {
	rule      config.CategoryKBRule
	taxonomy  CategoryKBTaxonomy
	kbs       map[string]*categoryKBData
	modelType string

	privateTiers map[string]bool // tiers treated as "private" for contrastive scoring
}

// NewCategoryKBClassifier creates a classifier from a CategoryKBRule.
// It loads KB JSONs, taxonomy, and pre-embeds all exemplars.
func NewCategoryKBClassifier(rule config.CategoryKBRule, modelType string) (*CategoryKBClassifier, error) {
	c := &CategoryKBClassifier{
		rule:      rule,
		kbs:       make(map[string]*categoryKBData),
		modelType: modelType,
		privateTiers: map[string]bool{
			"security_containment": true,
			"privacy_policy":       true,
		},
	}

	if err := c.loadTaxonomy(); err != nil {
		logging.Warnf("[CategoryKB] No taxonomy loaded (%v), contrastive scoring will be disabled", err)
	}

	if err := c.loadKBs(); err != nil {
		return nil, fmt.Errorf("failed to load category KBs from %s: %w", rule.KBDir, err)
	}

	if err := c.preloadEmbeddings(); err != nil {
		return nil, fmt.Errorf("failed to preload category KB embeddings: %w", err)
	}

	return c, nil
}

func (c *CategoryKBClassifier) loadTaxonomy() error {
	path := c.rule.TaxonomyPath
	if path == "" {
		path = filepath.Join(c.rule.KBDir, "taxonomy.json")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, &c.taxonomy)
}

func (c *CategoryKBClassifier) loadKBs() error {
	entries, err := os.ReadDir(c.rule.KBDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if entry.Name() == "taxonomy.json" {
			continue
		}

		categoryName := strings.TrimSuffix(entry.Name(), ".json")
		data, err := os.ReadFile(filepath.Join(c.rule.KBDir, entry.Name()))
		if err != nil {
			logging.Warnf("[CategoryKB] Failed to read %s: %v", entry.Name(), err)
			continue
		}

		var kb categoryKBData
		if err := json.Unmarshal(data, &kb); err != nil {
			logging.Warnf("[CategoryKB] Failed to parse %s: %v", entry.Name(), err)
			continue
		}

		if len(kb.Exemplars) == 0 {
			logging.Warnf("[CategoryKB] Skipping %s: no exemplars", categoryName)
			continue
		}

		c.kbs[categoryName] = &kb
	}

	if len(c.kbs) == 0 {
		return fmt.Errorf("no valid category KBs found in %s", c.rule.KBDir)
	}

	totalExemplars := 0
	for _, kb := range c.kbs {
		totalExemplars += len(kb.Exemplars)
	}
	logging.Infof("[CategoryKB] Loaded %d categories with %d total exemplars from %s",
		len(c.kbs), totalExemplars, c.rule.KBDir)

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
			logging.Warnf("[CategoryKB] Failed to embed exemplar %q in %s: %v",
				res.ref.text, res.ref.category, res.err)
			continue
		}
		c.kbs[res.ref.category].Embeddings[res.ref.index] = res.embedding
	}

	if failCount > 0 {
		logging.Warnf("[CategoryKB] %d/%d exemplar embeddings failed", failCount, len(refs))
	}

	logging.Infof("[CategoryKB] Preloaded embeddings for %d exemplars across %d categories in %v",
		len(refs)-failCount, len(c.kbs), time.Since(startTime))

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
		if c.isCategoryInTierGroup(cat, "security_containment") {
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
func (c *CategoryKBClassifier) Classify(text string) (*CategoryKBClassifyResult, error) {
	if text == "" {
		return nil, fmt.Errorf("category_kb classification: query must be provided")
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

	bestTier := ""
	if entry, ok := c.taxonomy.Categories[bestCat]; ok {
		bestTier = entry.Tier
	}

	elapsed := time.Since(startTime)
	logging.Infof("[CategoryKB] Classified in %v: best=%s (%.3f) tier=%s, contrastive=%.3f, matched=%d categories",
		elapsed, bestCat, bestSim, bestTier, contrastive, len(matchedRules))

	return &CategoryKBClassifyResult{
		BestCategory:     bestCat,
		BestSimilarity:   bestSim,
		BestTier:         bestTier,
		ContrastiveScore: contrastive,
		MatchedRules:     matchedRules,
		Confidences:      confidences,
	}, nil
}

func (c *CategoryKBClassifier) isCategoryPrivate(category string) bool {
	entry, ok := c.taxonomy.Categories[category]
	if !ok {
		return false
	}
	return c.privateTiers[entry.Tier]
}

func (c *CategoryKBClassifier) isCategoryInTierGroup(category, group string) bool {
	entry, ok := c.taxonomy.Categories[category]
	if !ok {
		return false
	}
	return entry.Tier == group
}

// CategoryCount returns the number of loaded categories.
func (c *CategoryKBClassifier) CategoryCount() int {
	return len(c.kbs)
}
