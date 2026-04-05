package classification

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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
}

func NewReaskClassifier(rules []config.ReaskRule, modelType string) (*ReaskClassifier, error) {
	if len(rules) == 0 {
		return nil, fmt.Errorf("reask rules cannot be empty")
	}
	if strings.TrimSpace(modelType) == "" {
		modelType = "qwen3"
	}
	return &ReaskClassifier{
		rules:     append([]config.ReaskRule(nil), rules...),
		modelType: strings.TrimSpace(modelType),
	}, nil
}

func (c *ReaskClassifier) Classify(currentUserTurn string, priorUserTurns []string) ([]ReaskMatch, error) {
	currentUserTurn = strings.TrimSpace(currentUserTurn)
	if currentUserTurn == "" || len(c.rules) == 0 || len(priorUserTurns) == 0 {
		return nil, nil
	}

	currentOutput, err := getEmbeddingWithModelType(currentUserTurn, c.modelType, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compute current user turn embedding: %w", err)
	}

	similarities, err := c.computeSimilarities(currentOutput.Embedding, priorUserTurns)
	if err != nil {
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

func (c *ReaskClassifier) computeSimilarities(currentEmbedding []float32, priorUserTurns []string) ([]float64, error) {
	cache := make(map[string][]float32, len(priorUserTurns))
	similarities := make([]float64, 0, len(priorUserTurns))

	for index := len(priorUserTurns) - 1; index >= 0; index-- {
		priorTurn := strings.TrimSpace(priorUserTurns[index])
		if priorTurn == "" {
			continue
		}

		priorEmbedding, ok := cache[priorTurn]
		if !ok {
			output, err := getEmbeddingWithModelType(priorTurn, c.modelType, 0)
			if err != nil {
				return nil, fmt.Errorf("failed to compute prior user turn embedding: %w", err)
			}
			priorEmbedding = output.Embedding
			cache[priorTurn] = priorEmbedding
		}

		similarities = append(similarities, float64(cosineSimilarity(currentEmbedding, priorEmbedding)))
	}

	return similarities, nil
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
