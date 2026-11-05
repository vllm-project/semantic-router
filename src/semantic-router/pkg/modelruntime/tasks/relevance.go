package tasks

import "fmt"

// QueryDocument keeps the query and retrieved document as separate inputs.
type QueryDocument struct {
	Query    string
	Document string
}

// RelevanceScores preserves input order and the scorer's input evidence.
// Native results carry relevance logits and tokenizer usage. Decision results
// carry ordinal relevance and explicit coverage, never fabricated pair tokens.
type RelevanceScores struct {
	Scores   []float32
	Inputs   []InputUsage
	Decision *DecisionRelevanceEvidence
}

// DecisionRelevanceEvidence keeps runtime coverage separate from token counts.
// InputTokens is the aggregate reported work, not a native pair tokenizer count.
type DecisionRelevanceEvidence struct {
	Coverage    []string
	InputTokens int
}

func (r RelevanceScores) Semantics() ScoreSemantics {
	if r.Decision != nil {
		low, high := 0.0, 3.0
		return ScoreSemantics{Unit: "decision_relevance", Direction: HigherIsPositive, Minimum: &low, Maximum: &high}
	}
	return RelevanceScoreSemantics()
}

func RelevanceScoreSemantics() ScoreSemantics {
	return ScoreSemantics{Unit: "relevance_logit", Direction: HigherIsPositive, Calibrated: false}
}

func ValidateRelevanceScores(input []QueryDocument, output RelevanceScores) error {
	if output.Decision != nil {
		return validateDecisionRelevance(input, output)
	}
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
	if r.Decision != nil {
		// The decision transport accounts input tokens. It does not report the
		// original/processed pair counts this metadata contract requires.
		return nil
	}
	usage := &InputUsage{}
	for _, item := range r.Inputs {
		usage.OriginalTokens += item.OriginalTokens
		usage.ProcessedTokens += item.ProcessedTokens
		usage.Truncated = usage.Truncated || item.Truncated
	}
	return usage
}

func validateDecisionRelevance(input []QueryDocument, output RelevanceScores) error {
	if len(input) == 0 || len(input) != len(output.Scores) || len(input) != len(output.Decision.Coverage) || len(output.Inputs) != 0 || output.Decision.InputTokens < 0 {
		return fmt.Errorf("decision relevance requires one score and coverage proof per candidate without native pair usage")
	}
	for i, value := range output.Scores {
		if err := output.Semantics().Validate(ScoreResult{Value: float64(value)}); err != nil {
			return err
		}
		if output.Decision.Coverage[i] != "complete" {
			return fmt.Errorf("decision relevance requires complete candidate coverage")
		}
	}
	return nil
}
