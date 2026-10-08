package classification

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// decisionPIIBackend implements judgments without inventing token locations.
// A caller requiring locations receives a capability error; routing policy
// consumes presence/categories through evaluateDecisionPIISignal instead.
type decisionPIIBackend struct {
	judgment   *decisionJudgment
	categories modelservice.TaskPlan
}

func (*decisionPIIBackend) Init(string, bool, int) error { return nil }
func (*decisionPIIBackend) ClassifyTokens(context.Context, string) (tasks.TokenClassificationResult, error) {
	return tasks.TokenClassificationResult{}, fmt.Errorf("%w: this PII deployment provides presence and categories; exact locations require native span capability", binding.ErrCapability)
}

type privacyJudgmentResult struct {
	presence   float64
	categories map[string]float64
	err        error
}

func (b *decisionPIIBackend) judge(ctx context.Context, text string) privacyJudgmentResult {
	response, _, err := modelservice.ExecuteTaskPlans(ctx, b.judgment.decider, b.judgment.deployment,
		modelservice.Request{State: text, MaxTokens: b.judgment.scan}, []modelservice.TaskPlan{b.judgment.plan, b.categories})
	if err != nil {
		return privacyJudgmentResult{err: err}
	}
	presence, categories := response.Answers[b.judgment.plan.Question.ID], response.Answers[b.categories.Question.ID]
	if presence.Error != "" {
		return privacyJudgmentResult{err: decisionAnswerError(presence.Error)}
	}
	if categories.Error != "" {
		return privacyJudgmentResult{err: decisionAnswerError(categories.Error)}
	}
	if math.IsNaN(presence.Noul) || math.IsInf(presence.Noul, 0) || presence.Noul < 0 || presence.Noul > 1 {
		return privacyJudgmentResult{err: fmt.Errorf("PII judgment returned invalid probability")}
	}
	return privacyJudgmentResult{presence: presence.Noul, categories: categories.Probabilities}
}

func decisionPIIInference(inference PIIInference) *decisionPIIBackend {
	if backend, ok := inference.(*decisionPIIBackend); ok {
		return backend
	}
	if admitted, ok := inference.(admittedPIIInference); ok {
		return decisionPIIInference(admitted.backend)
	}
	return nil
}

func prepareDecisionPII(models *classifierModelRuntime) (*decisionPIIBackend, error) {
	judgment, err := newDecisionJudgment(models, "pii_classifier", "pii_presence", nil)
	if err != nil || judgment == nil {
		return nil, err
	}
	// Existing span models and explicit token contracts keep the precise path.
	spec, explicit := models.plan.Lookup(models.recipe, "pii_classifier")
	if judgment.card.Answers("span") && (!explicit || spec.Binding.Contract != config.DecisionTaskContract) {
		return nil, nil
	}
	if explicitSpanBinding(models, "pii_classifier") {
		return nil, fmt.Errorf("%w: explicit pii_classifier token_spans.v1 binding requires a span model; use decision.v1 for presence/categories", binding.ErrCapability)
	}
	definition, _ := modelservice.BuiltinTask("pii_categories")
	question := definition.Question
	question.ID = "pii_classifier:categories"
	plan, err := modelservice.CompileTask(definition, question, judgment.card)
	if err != nil {
		return nil, err
	}
	return &decisionPIIBackend{judgment: judgment, categories: plan}, nil
}

func (c *Classifier) evaluateDecisionPIISignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string, history []string, backend *decisionPIIBackend) {
	started := time.Now()
	contents := []string{}
	if text != "" {
		contents = append(contents, text)
	}
	for _, rule := range c.Config.PIIRules {
		if rule.IncludeHistory {
			for _, item := range history {
				if item != "" && !slices.Contains(contents, item) {
					contents = append(contents, item)
				}
			}
		}
	}
	judgments := make([]privacyJudgmentResult, len(contents))
	modelservice.Fan(ctx, len(contents), func(i int) {
		judgments[i] = c.judgePII(ctx, contents[i], backend)
		judgments[i].err = signalDeadline(ctx, judgments[i].err)
	})
	byContent := make(map[string]privacyJudgmentResult, len(contents))
	verified := len(contents) > 0 && len(c.Config.PIIRules) > 0
	for i, content := range contents {
		byContent[content] = judgments[i]
		if judgments[i].err != nil {
			verified = false
		}
	}
	for _, content := range history {
		if content != "" && !slices.Contains(contents, content) {
			verified = false
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if results.SignalErrors == nil {
		results.SignalErrors = map[string]string{}
	}
	for _, rule := range c.Config.PIIRules {
		entities := map[string]bool{}
		failed, unscanned := false, false
		code := ""
		for _, content := range collectPIIRuleContents(text, history, rule.IncludeHistory) {
			result := byContent[content]
			if result.err != nil {
				failed = true
				code = mergeSignalErrorCode(code, boundedSignalErrorCode(result.err, piiEvaluationFailedCode))
				unscanned = unscanned || UnscannedInput(result.err)
				continue
			}
			present, categoryFound := result.presence >= float64(rule.Threshold), false
			for label, probability := range result.categories {
				if probability >= float64(rule.Threshold) {
					entities[label], present, categoryFound = true, true, true
				}
			}
			if present {
				verified = false
				if !categoryFound {
					entities["PERSONAL_DATA"] = true
				}
			}
		}
		denied := findDeniedEntities(entities, rule.PIITypesAllowed)
		key := signalConfidenceKey(config.SignalTypePII, rule.Name)
		if failed {
			results.SignalErrors[key] = code
			if c.Config.PIIModel.IsBlock() || (unscanned && c.Config.PIIModel.UnscannedBlocks()) {
				if len(denied) == 0 {
					if results.SignalErrorMatches == nil {
						results.SignalErrorMatches = map[string]bool{}
					}
					results.SignalErrorMatches[key] = true
				}
				sentinel := PIIClassificationErrorType
				if unscanned {
					sentinel = PIIUnscannedType
				}
				denied = append(denied, sentinel)
			}
		}
		if len(denied) > 0 {
			results.PIIDetected = true
			results.MatchedPIIRules = append(results.MatchedPIIRules, rule.Name)
			for _, entity := range denied {
				if !slices.Contains(results.PIIEntities, entity) {
					results.PIIEntities = append(results.PIIEntities, entity)
				}
			}
			c.recordSignalMatch(config.SignalTypePII, rule.Name)
		}
		c.recordSignalExtraction(config.SignalTypePII, rule.Name, time.Since(started).Seconds())
	}
	for i, content := range contents {
		result := judgments[i]
		clean := result.err == nil
		for _, rule := range c.Config.PIIRules {
			if result.presence >= float64(rule.Threshold) {
				clean = false
			}
			for _, probability := range result.categories {
				if probability >= float64(rule.Threshold) {
					clean = false
				}
			}
		}
		results.PIIEvidence = append(results.PIIEvidence, NewPrivacyEvidence("request", content, result.err == nil, clean))
	}
	results.PIIContentVerified = verified
	results.Metrics.PII.ExecutionTimeMs = float64(time.Since(started).Microseconds()) / 1000
}

func (b *decisionPIIBackend) questionDeployment() string { return b.judgment.deployment }

func (c *Classifier) judgePII(ctx context.Context, text string, backend *decisionPIIBackend) privacyJudgmentResult {
	if admitted, ok := c.piiInference.(admittedPIIInference); ok {
		result, err := admitModelInference(ctx, admitted.gate, admitted.deployment, func() (privacyJudgmentResult, error) { r := backend.judge(ctx, text); return r, r.err })
		if err != nil {
			result.err = err
		}
		return result
	}
	return backend.judge(ctx, text)
}

func (a admittedPIIInference) questionDeployment() string {
	if asker, ok := a.backend.(questionAsker); ok {
		return asker.questionDeployment()
	}
	return ""
}
