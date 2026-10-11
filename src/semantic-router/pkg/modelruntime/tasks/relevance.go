package tasks

import "fmt"

// QueryDocument is one native tokenizer pair, not concatenated text.
type QueryDocument struct {
	Query    string
	Document string
}

// RelevanceScores preserves input order and each pair's complete token usage.
// Values are raw relevance logits; neither a probability nor a similarity.
type RelevanceScores struct {
	Scores []float32
	Inputs []InputUsage
}

func RelevanceScoreSemantics() ScoreSemantics {
	return ScoreSemantics{Unit: "relevance_logit", Direction: HigherIsPositive, Calibrated: false}
}

func ValidateRelevanceScores(input []QueryDocument, output RelevanceScores) error {
	if len(input) == 0 || len(input) != len(output.Scores) || len(input) != len(output.Inputs) {
		return fmt.Errorf("relevance scores and input usage must match the pair count")
	}
	for i, value := range output.Scores {
		if err := RelevanceScoreSemantics().Validate(ScoreResult{Value: float64(value)}); err != nil {
			return err
		}
		usage := output.Inputs[i]
		if usage.OriginalTokens <= 0 || usage.OriginalTokens != usage.ProcessedTokens || usage.Truncated {
			return fmt.Errorf("relevance scoring requires complete paired input usage")
		}
	}
	return nil
}

// InputMetadata reports total work across pairs. Per-pair budgets are checked
// by the native tokenizer, never against this batch aggregate.
func (r RelevanceScores) InputMetadata() *InputUsage {
	usage := &InputUsage{}
	for _, item := range r.Inputs {
		usage.OriginalTokens += item.OriginalTokens
		usage.ProcessedTokens += item.ProcessedTokens
		usage.Truncated = usage.Truncated || item.Truncated
	}
	return usage
}
