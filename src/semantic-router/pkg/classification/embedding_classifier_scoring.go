package classification

import (
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// findAllMatchedRules aggregates candidate similarities per rule and returns all
// accepted matches before final top-k output shaping.
func (c *EmbeddingClassifier) findAllMatchedRules(scoredRules []EmbeddingRuleScore) []MatchedRule {
	hardMatches := make([]MatchedRule, 0, len(scoredRules))

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

		if len(queryEmbedding) != len(bank.prototypes[0].Embedding) {
			return nil, fmt.Errorf("embedding rule %q: query dimension %d differs from candidate dimension %d", rule.Name, len(queryEmbedding), len(bank.prototypes[0].Embedding))
		}
		bankScore := bank.score(queryEmbedding, c.embeddingAggregationOptions(rule))
		positiveScore := bankScore.Score
		var negativeScore float64
		if rule.HasNegativeCandidates() {
			negativeBank := c.negativeRulePrototypeBanks[rule.Name]
			if negativeBank == nil || len(negativeBank.prototypes) == 0 || len(negativeBank.prototypes[0].Embedding) != len(queryEmbedding) {
				return nil, fmt.Errorf("embedding rule %q: negative candidates are missing or have incompatible dimensions", rule.Name)
			}
			negativeScore = negativeBank.score(queryEmbedding, c.embeddingAggregationOptions(rule)).Score
			bankScore.Score -= negativeScore
		}
		logging.Infof("Rule %q: score=%.4f best=%.4f support=%.4f threshold=%.3f matched=%v (prototypes=%d)",
			rule.Name, bankScore.Score, bankScore.Best, bankScore.Support, rule.SimilarityThreshold,
			bankScore.Score >= float64(rule.SimilarityThreshold), bankScore.PrototypeCount)

		scoredRules = append(scoredRules, EmbeddingRuleScore{
			Name:           rule.Name,
			Score:          bankScore.Score,
			Best:           bankScore.Best,
			Support:        bankScore.Support,
			PositiveScore:  positiveScore,
			NegativeScore:  negativeScore,
			Contrastive:    rule.HasNegativeCandidates(),
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
	topK := 0
	if c.optimizationConfig.TopK != nil {
		topK = *c.optimizationConfig.TopK
	}
	if topK <= 0 || len(matches) <= topK {
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
		return defaultPrototypeScoreOptions(rule.EffectivePrototypeScoring(c.optimizationConfig.PrototypeScoring))
	}
}

// cosineSimilarity computes cosine similarity between two vectors.
// Assumes vectors are normalized (which they should be from BERT-style models).
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}

	var dotProduct float32
	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
	}

	return dotProduct
}

// GetPreloadStats returns statistics about preloaded embeddings.
func (c *EmbeddingClassifier) GetPreloadStats() int {
	c.preloadMu.Lock()
	defer c.preloadMu.Unlock()
	return len(c.candidateEmbeddings) + len(c.imageCandidateEmbeddings)
}

func (c *EmbeddingClassifier) rebuildRulePrototypeBanks() {
	c.rulePrototypeBanks = make(map[string]*prototypeBank, len(c.rules))
	c.negativeRulePrototypeBanks = make(map[string]*prototypeBank, len(c.rules))
	for _, rule := range c.rules {
		policy := rule.EffectivePrototypeScoring(c.optimizationConfig.PrototypeScoring)
		bank := newPrototypeBank(c.embeddingPrototypeExamples(rule.Candidates, rule.ImageCandidates), policy)
		c.rulePrototypeBanks[rule.Name] = bank
		logPrototypeBankSummary("Embedding Signal", rule.Name, bank)
		if rule.HasNegativeCandidates() {
			c.negativeRulePrototypeBanks[rule.Name] = newPrototypeBank(c.embeddingPrototypeExamples(rule.NegativeCandidates, rule.NegativeImageCandidates), policy)
		}
	}
}

func (c *EmbeddingClassifier) embeddingPrototypeExamples(texts, images []string) []prototypeExample {
	examples := make([]prototypeExample, 0, len(texts)+len(images))
	appendExamples := func(values []string, vectors map[string][]float32, modality string) {
		for _, value := range values {
			if vector := vectors[value]; len(vector) > 0 {
				examples = append(examples, prototypeExample{Key: modality + "\x00" + value, Text: value, Embedding: vector})
			}
		}
	}
	appendExamples(texts, c.candidateEmbeddings, "text")
	appendExamples(images, c.imageCandidateEmbeddings, "image")
	return examples
}
