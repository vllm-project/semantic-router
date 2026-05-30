package classification

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *ContrastivePreferenceClassifier) ClassifyDetailed(text string) (*PreferenceClassificationDetails, error) {
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is empty")
	}

	c.mu.RLock()
	if len(c.ruleBanks) == 0 {
		c.mu.RUnlock()
		return nil, fmt.Errorf("no embeddings loaded for contrastive preference classifier")
	}
	c.mu.RUnlock()

	out, err := getEmbeddingWithModelType(text, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	queryEmbedding := out.Embedding

	c.mu.RLock()
	defer c.mu.RUnlock()

	scores := make([]PreferenceRuleScore, 0, len(c.ruleBanks))
	for _, rule := range c.rules {
		bank, ok := c.ruleBanks[rule.Name]
		if !ok || bank == nil || len(bank.prototypes) == 0 {
			continue
		}
		bankScore := bank.score(queryEmbedding, defaultPrototypeScoreOptions(c.prototypeCfg))
		score := PreferenceRuleScore{
			Name:           rule.Name,
			Score:          float32(bankScore.Score),
			Best:           float32(bankScore.Best),
			Support:        float32(bankScore.Support),
			Threshold:      c.ruleThresholds[rule.Name],
			PrototypeCount: bankScore.PrototypeCount,
		}
		logging.Debugf("[Preference Contrastive] rule=%s score=%.4f best=%.4f support=%.4f prototypes=%d",
			rule.Name, score.Score, score.Best, score.Support, score.PrototypeCount)
		scores = append(scores, score)
	}

	if len(scores) == 0 {
		return nil, fmt.Errorf("no preference matched by contrastive classifier")
	}

	sort.Slice(scores, func(i, j int) bool {
		if scores[i].Score == scores[j].Score {
			return scores[i].Name < scores[j].Name
		}
		return scores[i].Score > scores[j].Score
	})

	details := &PreferenceClassificationDetails{
		Scores:    scores,
		BestRule:  scores[0].Name,
		BestScore: scores[0].Score,
	}
	if len(scores) > 1 {
		details.RunnerUpRule = scores[1].Name
		details.RunnerUp = scores[1].Score
		details.Margin = scores[0].Score - scores[1].Score
	}

	return details, nil
}

func (c *ContrastivePreferenceClassifier) collectExamples(rule config.PreferenceRule) []string {
	examples := make([]string, 0, 1+len(rule.Examples))

	if rule.Description != "" {
		examples = append(examples, rule.Description)
	}

	if len(rule.Examples) > 0 {
		examples = append(examples, rule.Examples...)
	}

	return examples
}

func (c *ContrastivePreferenceClassifier) rebuildRuleBanks() {
	c.ruleBanks = make(map[string]*prototypeBank, len(c.rules))
	for _, rule := range c.rules {
		embeddings := c.ruleEmbeddings[rule.Name]
		examples := c.collectExamples(rule)
		prototypeExamples := make([]prototypeExample, 0, len(embeddings))
		for i, embedding := range embeddings {
			if len(embedding) == 0 {
				continue
			}
			text := rule.Name
			if i < len(examples) {
				text = examples[i]
			}
			prototypeExamples = append(prototypeExamples, prototypeExample{
				Key:       fmt.Sprintf("%s:%d", rule.Name, i),
				Text:      text,
				Embedding: embedding,
			})
		}
		bank := newPrototypeBank(prototypeExamples, c.prototypeCfg)
		c.ruleBanks[rule.Name] = bank
		logPrototypeBankSummary("Preference Contrastive", rule.Name, bank)
	}
}
