package classification

import (
	"context"
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
// deployments run in parallel. A failed or late call leaves its signals
// unknown; it never fails the request.
func (c *Classifier) evaluateDecisionModelSignals(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	text string,
	usedSignals map[string]bool,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	byDeployment := make(map[string][]config.DecisionSignalRule)
	var order []string
	for _, rule := range c.Config.DecisionRules {
		if !signalRuleUsed(usedSignals, config.SignalTypeDecision, rule.Name) {
			continue
		}
		if _, seen := byDeployment[rule.Deployment]; !seen {
			order = append(order, rule.Deployment)
		}
		byDeployment[rule.Deployment] = append(byDeployment[rule.Deployment], rule)
	}
	var waitGroup sync.WaitGroup
	for _, deployment := range order {
		waitGroup.Add(1)
		go func(deployment string, rules []config.DecisionSignalRule) {
			defer waitGroup.Done()
			c.evaluateDecisionDeployment(ctx, results, mu, text, deployment, rules)
		}(deployment, byDeployment[deployment])
	}
	waitGroup.Wait()
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
	timeout := rules[0].EffectiveTimeout()
	request := modelservice.Request{State: text, Questions: make([]modelservice.Question, 0, len(rules))}
	for _, rule := range rules {
		if rule.EffectiveTimeout() < timeout {
			timeout = rule.EffectiveTimeout()
		}
		request.Questions = append(request.Questions, DecisionQuestion(rule.Name, rule.Question))
	}
	callCtx, cancel := context.WithTimeout(ctx, timeout)
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
		if matched := applyDecisionAnswer(results, rule, answer); matched != "" {
			results.MatchedDecisionRules = append(results.MatchedDecisionRules, matched)
			c.recordSignalMatch(config.SignalTypeDecision, matched)
		}
	}
}

// DecisionQuestion converts a configured question to the runtime request form.
func DecisionQuestion(id string, question config.DecisionQuestion) modelservice.Question {
	converted := modelservice.Question{ID: id, Type: question.Type, Instructions: question.Instructions}
	for _, choice := range question.Choices {
		converted.Choices = append(converted.Choices, modelservice.Choice{Key: choice.Key, Description: choice.Description})
	}
	converted.Levels = append(converted.Levels, question.Levels...)
	return converted
}

// applyDecisionAnswer publishes an answer's values and returns the matched
// reference: the rule name (Noul, Score) or "rule:choice" (Choice), or "".
func applyDecisionAnswer(results *SignalResults, rule config.DecisionSignalRule, answer modelservice.Answer) string {
	key := signalConfidenceKey(config.SignalTypeDecision, rule.Name)
	predicate := rule.EffectivePredicate()
	switch rule.Question.Type {
	case config.DecisionQuestionNoul:
		results.SignalValues[key] = answer.Noul
		results.SignalConfidences[key] = answer.Noul
		if predicateMatches(answer.Noul, predicate) {
			return rule.Name
		}
	case config.DecisionQuestionScore:
		results.SignalValues[key] = answer.Score
		results.SignalConfidences[key] = maxProbability(answer.Probabilities)
		if predicateMatches(answer.Score, predicate) {
			return rule.Name
		}
	default:
		for option, probability := range answer.Probabilities {
			results.SignalValues[key+":"+option] = probability
			results.SignalConfidences[key+":"+option] = probability
		}
		chosen := answer.Probabilities[answer.Choice]
		results.SignalValues[key] = chosen
		results.SignalConfidences[key] = chosen
		if answer.Choice != "" && (predicate == nil || predicateMatches(chosen, predicate)) {
			return rule.Name + ":" + answer.Choice
		}
	}
	return ""
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
