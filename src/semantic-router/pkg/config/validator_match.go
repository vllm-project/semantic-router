package config

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// validateMatch checks the match block a feature writes at path (its whole
// document path, which a match_empty error names). A feature that writes no
// block runs its default.
func validateMatch(cfg *RouterConfig, path string, block *MatchConfig, feature MatchFeature) error {
	if block == nil {
		return nil
	}
	row, ok := matchFeatures[feature]
	if !ok {
		return fmt.Errorf("%s: %q is not a feature with a match block", path, feature)
	}
	if block.Recall == nil && block.Judge == nil {
		return CheckError(path, CodeMatchEmpty, "a match block runs recall, judge or both; write at least one")
	}
	if block.Recall == nil && row.indexed && !row.listed {
		return fmt.Errorf("%s: %s recalls its candidates from an index, which a judge cannot read whole; add recall", path, row.noun)
	}
	if block.Recall != nil {
		if err := row.validateRecall(path+".recall", *block.Recall); err != nil {
			return err
		}
	}
	if block.Judge != nil {
		return validateMatchJudge(cfg, path+".judge", *block.Judge)
	}
	return nil
}

func (row matchFeature) validateRecall(path string, recall MatchRecall) error {
	if len(row.recallMethods) == 0 {
		return fmt.Errorf("%s: %s judges its candidates whole and recalls none; remove recall", path, row.noun)
	}
	method := recall.EffectiveMethod()
	if method != MatchRecallEmbedding && method != MatchRecallBM25 {
		return fmt.Errorf("%s.method must be %s or %s", path, MatchRecallEmbedding, MatchRecallBM25)
	}
	if !slices.Contains(row.recallMethods, method) {
		return fmt.Errorf("%s.method: %s recalls with %s only", path, row.noun, strings.Join(row.recallMethods, " or "))
	}
	if recall.TopK < 0 || (row.indexed && recall.TopK == 0) {
		return fmt.Errorf("%s.top_k must be at least 1: the index returns at most top_k candidates", path)
	}
	if bound := recall.MinSimilarity; bound != nil {
		if method == MatchRecallEmbedding && !within(*bound, -1, 1) {
			return fmt.Errorf("%s.min_similarity must be a cosine similarity within [-1, 1]", path)
		}
		if method == MatchRecallBM25 && !within(*bound, 0, math.MaxFloat64) {
			return fmt.Errorf("%s.min_similarity must be a BM25 score of at least 0", path)
		}
	}
	return nil
}

// validateMatchJudge checks a judge: its question, the deployment that
// answers it, which must answer its type, its keep rules and its cap. A
// choice judge reads at most 255 candidates, the options one question takes;
// more are truncated, so no recall bound is refused.
func validateMatchJudge(cfg *RouterConfig, path string, judge MatchJudge) error {
	switch {
	case judge.Preset != "" && judge.Question != nil:
		return fmt.Errorf("%s: a judge asks a preset or a question, not both", path)
	case judge.Question != nil:
		if err := validateMatchQuestion(*judge.Question); err != nil {
			return fmt.Errorf("%s.question: %w", path, err)
		}
	case !slices.Contains(presets.Judges(), judge.Preset):
		return fmt.Errorf("%s: preset must be %s, or write a question", path, strings.Join(presets.Judges(), ", "))
	}
	if strings.TrimSpace(judge.Deployment) != judge.Deployment {
		return fmt.Errorf("%s.deployment must name a model_runtime or systemone deployment; omit it to ask the decision model", path)
	}
	if err := validateQuestionDeployment(cfg, judge.Deployment, "a match judge"); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if err := validateDeclaredQuestionType(cfg, cfg.MatchJudgeDeployment(judge), judge.QuestionType()); err != nil {
		return fmt.Errorf("%s: %w", path, err)
	}
	if judge.MaxCandidates < 0 || (judge.Chooses() && judge.MaxCandidates > presets.MaxChoices) {
		if judge.Chooses() {
			return fmt.Errorf("%s.max_candidates must be within [1, %d]: a choice judge's candidates are the options of one question", path, presets.MaxChoices)
		}
		return fmt.Errorf("%s.max_candidates must be at least 1 when set", path)
	}
	if judge.TopK < 0 {
		return fmt.Errorf("%s.top_k must be at least 1 when set", path)
	}
	if bound := judge.MinScore; bound != nil {
		top := 1.0
		if judge.QuestionType() == SystemOneQuestionScore {
			top = float64(len(judge.Question.Levels) - 1)
		}
		if !within(*bound, 0, top) {
			return fmt.Errorf("%s.min_score must be within [0, %g], the range of the question's scores", path, top)
		}
	}
	if ratio := judge.MinRatio; ratio != nil && (!within(*ratio, 0, 1) || *ratio == 0) {
		return fmt.Errorf("%s.min_ratio must be within (0, 1]", path)
	}
	if judge.OnError != "" && judge.OnError != MatchOnErrorSkip && judge.OnError != MatchOnErrorFail {
		return fmt.Errorf("%s.on_error must be %s or %s", path, MatchOnErrorSkip, MatchOnErrorFail)
	}
	if judge.TimeoutMs < 0 || judge.TimeoutMs > MaxSystemOneTimeoutMs {
		return fmt.Errorf("%s.timeout_ms must be within [1, %d] when set", path, MaxSystemOneTimeoutMs)
	}
	return nil
}

// validateMatchQuestion checks a judge's custom question. A choice question
// takes no choices of its own: the candidates are its options.
func validateMatchQuestion(question SystemOneQuestion) error {
	switch question.Type {
	case SystemOneQuestionChoice:
		if strings.TrimSpace(question.Instructions) == "" {
			return fmt.Errorf("instructions are required")
		}
		if len(question.Choices) > 0 || len(question.Levels) > 0 || len(question.Labels) > 0 || question.Threshold != nil || question.Head != "" {
			return fmt.Errorf("a choice judge chooses among the candidates, which are its options; write only its instructions")
		}
		return nil
	case SystemOneQuestionNoul, SystemOneQuestionScore:
		return validateSystemOneQuestion(question)
	default:
		return fmt.Errorf("type must be choice, noul or score")
	}
}

func within(value, low, high float64) bool {
	return !math.IsNaN(value) && value >= low && value <= high
}
