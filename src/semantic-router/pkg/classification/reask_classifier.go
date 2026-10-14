package classification

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type ReaskMatch struct {
	RuleName      string
	MinSimilarity float64
	MatchedTurns  int
	LookbackTurns int
}

type ReaskClassifier struct {
	rules     []config.ReaskRule
	modelType string
	provider  embedding.Provider
	judgment  *decisionJudgment
}

func NewReaskClassifierWithProvider(rules []config.ReaskRule, modelType string, provider embedding.Provider) (*ReaskClassifier, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("reask rules cannot be empty")
	}
	if strings.TrimSpace(modelType) == "" {
		modelType = "qwen3"
	}
	return &ReaskClassifier{
		rules:     append([]config.ReaskRule(nil), rules...),
		modelType: strings.TrimSpace(modelType),
		provider:  provider,
	}, nil
}

func (c *ReaskClassifier) Classify(currentUserTurn string, priorUserTurns []string) ([]ReaskMatch, error) {
	return c.ClassifyContext(context.Background(), currentUserTurn, priorUserTurns)
}

func (c *ReaskClassifier) ClassifyContext(ctx context.Context, currentUserTurn string, priorUserTurns []string) ([]ReaskMatch, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	currentUserTurn = strings.TrimSpace(currentUserTurn)
	if currentUserTurn == "" || len(c.rules) == 0 || len(priorUserTurns) == 0 {
		return nil, nil
	}

	var similarities []float64
	var err error
	if c.judgment != nil {
		similarities, err = c.semanticSimilarities(ctx, currentUserTurn, priorUserTurns)
	} else {
		similarities, err = c.computeSimilarities(ctx, currentUserTurn, priorUserTurns, minimumReaskThreshold(c.rules))
	}
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	matches := make([]ReaskMatch, 0, len(c.rules))
	for _, rawRule := range c.rules {
		rule := rawRule.WithDefaults()
		if len(similarities) < rule.LookbackTurns {
			continue
		}

		requiredMin, streak := evaluateReaskStreak(similarities, float64(rule.Threshold), rule.LookbackTurns)
		if streak < rule.LookbackTurns {
			continue
		}

		matches = append(matches, ReaskMatch{
			RuleName:      rule.Name,
			MinSimilarity: requiredMin,
			MatchedTurns:  streak,
			LookbackTurns: rule.LookbackTurns,
		})
	}

	return retainMaxLookbackReaskMatches(matches), nil
}

func (c *ReaskClassifier) computeSimilarities(ctx context.Context, current string, priorUserTurns []string, minimumThreshold float64) ([]float64, error) {
	var currentEmbedding []float32
	cache := make(map[string][]float32, len(priorUserTurns))
	similarities := make([]float64, 0, len(priorUserTurns))

	for index := len(priorUserTurns) - 1; index >= 0; index-- {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		priorTurn := strings.TrimSpace(priorUserTurns[index])
		if priorTurn == "" {
			continue
		}
		if priorTurn == current {
			similarities = append(similarities, 1)
			continue
		}
		if currentEmbedding == nil {
			var err error
			currentEmbedding, err = c.embedFullText(ctx, current)
			if err != nil {
				return nil, fmt.Errorf("failed to compute current user turn embedding: %w", err)
			}
		}

		priorEmbedding, ok := cache[priorTurn]
		if !ok {
			embedding, err := c.embedFullText(ctx, priorTurn)
			if err != nil {
				return nil, fmt.Errorf("failed to compute prior user turn embedding: %w", err)
			}
			priorEmbedding = embedding
			cache[priorTurn] = priorEmbedding
		}

		similarity := float64(cosineSimilarity(currentEmbedding, priorEmbedding))
		similarities = append(similarities, similarity)
		if similarity < minimumThreshold {
			break
		}
	}

	return similarities, nil
}

func minimumReaskThreshold(rules []config.ReaskRule) float64 {
	minimumThreshold := float64(rules[0].WithDefaults().Threshold)
	for _, rawRule := range rules[1:] {
		threshold := float64(rawRule.WithDefaults().Threshold)
		if threshold < minimumThreshold {
			minimumThreshold = threshold
		}
	}
	return minimumThreshold
}

func (c *ReaskClassifier) embedFullText(ctx context.Context, text string) ([]float32, error) {
	return embedding.EmbedFullInput(ctx, c.provider, text, embedding.Options{})
}

func evaluateReaskStreak(similarities []float64, threshold float64, lookbackTurns int) (float64, int) {
	requiredMin := 1.0
	streak := 0

	for index, similarity := range similarities {
		if similarity < threshold {
			break
		}
		streak++
		if index < lookbackTurns && similarity < requiredMin {
			requiredMin = similarity
		}
	}

	return requiredMin, streak
}

func retainMaxLookbackReaskMatches(matches []ReaskMatch) []ReaskMatch {
	if len(matches) <= 1 {
		return matches
	}

	maxLookback := 0
	for _, match := range matches {
		if match.LookbackTurns > maxLookback {
			maxLookback = match.LookbackTurns
		}
	}

	filtered := make([]ReaskMatch, 0, len(matches))
	for _, match := range matches {
		if match.LookbackTurns == maxLookback {
			filtered = append(filtered, match)
		}
	}
	return filtered
}

// semanticSimilarities asks each pair about repeated intent. The code below
// still owns ordering and consecutive counting; no model invents a counter.
func (c *ReaskClassifier) semanticSimilarities(ctx context.Context, current string, prior []string) ([]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	turns := make([]string, 0, len(prior))
	for i := len(prior) - 1; i >= 0; i-- {
		if text := strings.TrimSpace(prior[i]); text != "" {
			turns = append(turns, text)
		}
	}
	unique, positions := []string{}, map[string]int{}
	for _, turn := range turns {
		if _, seen := positions[turn]; !seen {
			positions[turn] = len(unique)
			unique = append(unique, turn)
		}
	}
	judgments, failures := make([]float64, len(unique)), make([]error, len(unique))
	modelservice.Fan(ctx, len(unique), func(i int) {
		// Exact, nonempty repeated turns establish the same intent without a
		// probabilistic judgment. Compare the complete inputs, never a clipped
		// prefix; nonidentical pairs still need the model's full-input answer.
		if unique[i] == current {
			judgments[i] = 1
			return
		}
		answer, err := c.judgment.ask(ctx, modelservice.Request{Parts: map[string]string{"current": current, "prior": unique[i]}})
		judgments[i], failures[i] = answer.Noul, err
	})
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	scores := make([]float64, len(turns))
	for i, turn := range turns {
		index := positions[turn]
		if failures[index] != nil {
			return nil, fmt.Errorf("reask pair %d is unknown: %w", i, failures[index])
		}
		scores[i] = judgments[index]
	}
	return scores, nil
}
