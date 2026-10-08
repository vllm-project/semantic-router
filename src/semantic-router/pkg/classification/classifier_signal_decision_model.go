package classification

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const decisionSignalErrorPrefix = "decision_"

// SetDecisionDecider replaces the model runtime used by decision signals (tests and embedders).
func (c *Classifier) SetDecisionDecider(decider modelservice.Decider) {
	c.decisionDecider = decider
}

func (c *Classifier) decider() modelservice.Decider {
	if c.decisionDecider != nil {
		return c.decisionDecider
	}
	if decider, ok := c.models.decider(); ok {
		return decider
	}
	return modelservice.Default()
}

// evaluateDecisionModelSignals asks every used decision question in one call
// per deployment, with the request text as the state. Calls to different
// deployments run in parallel. A question that names no deployment asks the
// decision model, with the request as it came, like the built-in signals
// whose call it joins. A failed or late call leaves its signals unknown; it
// never fails the request.
func (c *Classifier) evaluateDecisionModelSignals(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	wholeText string,
	usedSignals map[string]bool,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	byDeployment := make(map[string][]config.DecisionSignalRule)
	stateOf := make(map[string]string)
	var order []string
	for _, rule := range c.Config.DecisionRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeDecision, rule.Name) {
			continue
		}
		deployment := c.Config.DecisionQuestionDeployment(rule)
		if _, seen := byDeployment[deployment]; !seen {
			order = append(order, deployment)
			stateOf[deployment] = text
			if rule.Deployment == "" {
				stateOf[deployment] = wholeText
			}
		}
		byDeployment[deployment] = append(byDeployment[deployment], rule)
	}
	modelservice.Fan(ctx, len(order), func(i int) {
		c.evaluateDecisionDeployment(ctx, results, mu, stateOf[order[i]], order[i], byDeployment[order[i]])
	})
	mu.Lock()
	results.Metrics.Decision.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	mu.Unlock()
}

func (c *Classifier) evaluateDecisionDeployment(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	deployment string,
	rules []config.DecisionSignalRule,
) {
	// Decision questions route, so a long text is read only as far as its
	// first tokens; the deployment's scan budget keeps them in one call with
	// the stage's other questions to it.
	request := modelservice.Request{State: text, Questions: make([]modelservice.Question, 0, len(rules)), MaxTokens: c.Config.ModelDeployments[deployment].ScanBudget()}
	for _, rule := range rules {
		question := DecisionQuestion(rule.Name, rule.Question)
		question.Truncate = true
		request.Questions = append(request.Questions, question)
	}
	callCtx, cancel := decisionCallContext(ctx, rules)
	defer cancel()
	started := time.Now()
	response, err := c.decider().Decide(callCtx, deployment, request)
	latency := time.Since(started).Seconds()
	if err != nil {
		modelservice.RecordUnknown(deployment, modelservice.ErrorReason(err), len(rules))
	}
	mu.Lock()
	defer mu.Unlock()
	for _, rule := range rules {
		key := signalConfidenceKey(config.SignalTypeDecision, rule.Name)
		c.recordSignalExtraction(config.SignalTypeDecision, rule.Name, latency)
		if err != nil {
			results.SignalErrors[key] = decisionSignalErrorPrefix + modelservice.ErrorReason(err)
			continue
		}
		answer, ok := response.Answers[rule.Name]
		if !ok || answer.Error != "" {
			reason := answer.Error
			if !ok {
				reason = "missing_answer"
			}
			results.SignalErrors[key] = decisionSignalErrorPrefix + reason
			modelservice.RecordUnknown(deployment, reason, 1)
			continue
		}
		for _, matched := range applyDecisionAnswer(results, rule, answer) {
			results.MatchedDecisionRules = append(results.MatchedDecisionRules, matched)
			c.recordSignalMatch(config.SignalTypeDecision, matched)
		}
	}
}

// decisionCallContext bounds a deployment's call by the shortest timeout its
// rules set. Questions to the decision model that set none join the built-in
// signals' call, so they take its deadline rather than the default timeout.
// Once the stage sends a call, its questions wait for it as long as its
// latest caller does: a timeout bounds the wait for the other askers, never
// an answer of a call the stage waits for anyway (modelservice.Bundle).
func decisionCallContext(ctx context.Context, rules []config.DecisionSignalRule) (context.Context, context.CancelFunc) {
	var timeout time.Duration
	for _, rule := range rules {
		if rule.Deployment == "" && rule.TimeoutMs == 0 {
			continue
		}
		if t := rule.EffectiveTimeout(); timeout == 0 || t < timeout {
			timeout = t
		}
	}
	if timeout == 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

// DecisionQuestion converts a configured question to the runtime request form.
func DecisionQuestion(id string, question config.DecisionQuestion) modelservice.Question {
	converted := modelservice.Question{ID: id, Type: question.Type, Instructions: question.Instructions, Threshold: question.Threshold, Head: question.Head}
	for _, choice := range question.Choices {
		converted.Choices = append(converted.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
	}
	for _, label := range question.Labels {
		converted.Labels = append(converted.Labels, modelservice.Choice{Key: label.Key, Description: label.Description})
	}
	converted.Levels = append(converted.Levels, question.Levels...)
	return converted
}

// applyDecisionAnswer publishes an answer's values and returns the matched
// references: the rule name (Noul, Score), "rule:choice" (Choice) or
// "rule:label" for every matching Set or Span label.
func applyDecisionAnswer(results *SignalResults, rule config.DecisionSignalRule, answer modelservice.Answer) []string {
	key := signalConfidenceKey(config.SignalTypeDecision, rule.Name)
	predicate := rule.EffectivePredicate()
	switch rule.Question.Type {
	case config.DecisionQuestionNoul:
		results.SignalValues[key] = answer.Noul
		results.SignalConfidences[key] = answer.Noul
		if predicateMatches(answer.Noul, predicate) {
			return []string{rule.Name}
		}
	case config.DecisionQuestionScore:
		results.SignalValues[key] = answer.Score
		results.SignalConfidences[key] = maxProbability(answer.Probabilities)
		if predicateMatches(answer.Score, predicate) {
			return []string{rule.Name}
		}
	case config.DecisionQuestionSet:
		return applySetAnswer(results, key, rule, answer, predicate)
	case config.DecisionQuestionSpan:
		return applySpanAnswer(results, key, rule, answer, predicate)
	default:
		for option, probability := range answer.Probabilities {
			results.SignalValues[key+":"+option] = probability
			results.SignalConfidences[key+":"+option] = probability
		}
		chosen := answer.Probabilities[answer.Choice]
		results.SignalValues[key] = chosen
		results.SignalConfidences[key] = chosen
		if answer.Choice != "" && (predicate == nil || predicateMatches(chosen, predicate)) {
			return []string{rule.Name + ":" + answer.Choice}
		}
	}
	return nil
}

// applySetAnswer publishes each label's probability, and the highest under
// the rule itself. A label matches when its probability meets the predicate,
// or without one when the model selected it.
func applySetAnswer(results *SignalResults, key string, rule config.DecisionSignalRule, answer modelservice.Answer, predicate *config.NumericPredicate) []string {
	var matched []string
	best := 0.0
	for _, label := range rule.Question.Labels {
		probability := answer.Probabilities[label.Key]
		results.SignalValues[key+":"+label.Key] = probability
		results.SignalConfidences[key+":"+label.Key] = probability
		best = max(best, probability)
		if (predicate == nil && slices.Contains(answer.Selected, label.Key)) || (predicate != nil && predicateMatches(probability, predicate)) {
			matched = append(matched, rule.Name+":"+label.Key)
		}
	}
	results.SignalValues[key] = best
	results.SignalConfidences[key] = best
	return matched
}

// applySpanAnswer publishes each label's highest span probability (0 when the
// model found no span of it) and, under the rule itself, the highest
// probability any word reached. A label matches when one of its spans'
// probabilities meets the predicate, or without one when it has a span.
func applySpanAnswer(results *SignalResults, key string, rule config.DecisionSignalRule, answer modelservice.Answer, predicate *config.NumericPredicate) []string {
	var matched []string
	for _, label := range rule.Question.Labels {
		best, found := 0.0, false
		for _, span := range answer.Spans {
			if span.Label != label.Key {
				continue
			}
			best = max(best, span.Probability)
			found = found || predicate == nil || predicateMatches(span.Probability, predicate)
		}
		results.SignalValues[key+":"+label.Key] = best
		results.SignalConfidences[key+":"+label.Key] = best
		if found {
			matched = append(matched, rule.Name+":"+label.Key)
		}
	}
	results.SignalValues[key] = answer.Noul
	results.SignalConfidences[key] = answer.Noul
	return matched
}

func maxProbability(probabilities map[string]float64) float64 {
	best := 0.0
	for _, probability := range probabilities {
		if probability > best {
			best = probability
		}
	}
	return best
}
