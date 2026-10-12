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
	response, err := b.judgment.askPlans(ctx,
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
	plan, err := judgment.preparePlan(definition, question)
	if err != nil {
		return nil, err
	}
	return &decisionPIIBackend{judgment: judgment, categories: plan}, nil
}

func (c *Classifier) evaluateDecisionPIISignal(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string, history []string, backend *decisionPIIBackend) {
	c.evaluateDecisionPIISignalWithToolResults(ctx, results, mu, text, history, nil, false, backend)
}

func (c *Classifier) evaluateDecisionPIISignalWithToolResults(ctx context.Context, results *SignalResults, mu *sync.Mutex, text string, history, toolTexts []string, toolIncomplete bool, backend *decisionPIIBackend) {
	started := time.Now()
	var contents []piiCacheKey
	seen := map[piiCacheKey]bool{}
	for _, rule := range c.Config.PIIRules {
		for _, content := range collectPIIRuleContentsForSource(rule, text, history, toolTexts) {
			key := piiCacheKey{source: piiCacheSource(rule.Source), content: content}
			if !seen[key] {
				seen[key] = true
				contents = append(contents, key)
			}
		}
	}
	judgments := make([]privacyJudgmentResult, len(contents))
	var selected []int
	budget := piiToolResultScanBudget{remainingInferenceCalls: maxPIIToolResultInferenceCalls}
	for i, key := range contents {
		if key.source == config.PIISourceToolResult && !budget.consumeInferenceCall() {
			judgments[i].err = ErrTokenSpansTruncated
		} else {
			selected = append(selected, i)
		}
	}
	modelservice.Fan(ctx, len(selected), func(j int) {
		i := selected[j]
		judgments[i] = c.judgePII(ctx, contents[i].content, backend)
		judgments[i].err = signalDeadline(ctx, judgments[i].err)
	})
	byContent := make(map[piiCacheKey]privacyJudgmentResult, len(contents))
	verified := len(contents) > 0 && len(c.Config.PIIRules) > 0 && !toolIncomplete
	for i, key := range contents {
		byContent[key] = judgments[i]
		if judgments[i].err != nil {
			verified = false
		}
	}
	for _, content := range append(append([]string{text}, history...), toolTexts...) {
		if content != "" && !seen[piiCacheKey{source: "legacy", content: content}] && !seen[piiCacheKey{source: config.PIISourceToolResult, content: content}] {
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
		incomplete := rule.Source == config.PIISourceToolResult && toolIncomplete
		failed, unscanned := incomplete, false
		code := ""
		if incomplete {
			code = piiEvaluationIncompleteCode
		}
		for _, content := range collectPIIRuleContentsForSource(rule, text, history, toolTexts) {
			result := byContent[piiCacheKey{source: piiCacheSource(rule.Source), content: content}]
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
	for i, key := range contents {
		result := judgments[i]
		complete := result.err == nil && (key.source != config.PIISourceToolResult || !toolIncomplete)
		clean := complete
		for _, rule := range c.Config.PIIRules {
			if piiCacheSource(rule.Source) != key.source {
				continue
			}
			if result.presence >= float64(rule.Threshold) {
				clean = false
			}
			for _, probability := range result.categories {
				if probability >= float64(rule.Threshold) {
					clean = false
				}
			}
		}
		results.PIIEvidence = append(results.PIIEvidence, NewPrivacyEvidence("request", key.content, complete, clean))
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
