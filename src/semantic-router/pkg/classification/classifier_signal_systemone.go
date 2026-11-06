package classification

import (
	"context"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

const systemOneSignalErrorPrefix = "systemone_"

// SystemOneQuestionID is the ID a rule's question carries on the wire: the
// Router's own, with a ':', so it is never an ID a model family was trained
// on and never turns a trained adapter on.
func SystemOneQuestionID(rule string) string {
	return config.SignalTypeSystemOne + ":" + rule
}

// SetDecisionDecider replaces the model runtime used by systemone signals (tests and embedders).
func (c *Classifier) SetDecisionDecider(decider modelservice.Decider) {
	c.decisionDecider = decider
}

// SetSystemOneDecider replaces the remote System One clients used by
// systemone signals to systemone deployments (tests and embedders).
func (c *Classifier) SetSystemOneDecider(remote modelservice.Remote) {
	c.systemOneDecider = remote
}

func (c *Classifier) systemOne() modelservice.Remote {
	if c.systemOneDecider != nil {
		return c.systemOneDecider
	}
	if c.models != nil {
		return c.models.systemOne
	}
	return nil
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

// ConversationTurn is one user or assistant message before the current user
// turn, in conversation order.
type ConversationTurn struct {
	Role string
	Text string
}

// systemOneReading is what a stage gives its systemone rules to read: the
// current user turn as it came, the conversation before it, the request's
// tool results and the current turn's images and videos, and at the response
// stage the model's answer and its grounding context.
type systemOneReading struct {
	request     string
	history     []string
	turns       []ConversationTurn
	toolResults []string
	answer      string
	context     string
	images      []string
	videos      []string
}

// state is the request a rule's question is asked about. A rule that reads
// only the current turn asks about its text; any other reads named parts,
// those the runtime reads by role (request as the user part, answer, and the
// others as context). A part with nothing to read is left out.
func (r systemOneReading) state(rule config.SystemOneRule) modelservice.Request {
	input := rule.Input
	parts := map[string]string{}
	add := func(name, text string) {
		if strings.TrimSpace(text) != "" {
			parts[name] = text
		}
	}
	if input.History {
		add("history", strings.Join(r.history, "\n\n"))
	}
	if input.PreviousTurns > 0 {
		turns := r.turns[max(0, len(r.turns)-input.PreviousTurns):]
		lines := make([]string, len(turns))
		for index, turn := range turns {
			lines[index] = turn.Role + ": " + turn.Text
		}
		add("previous_turns", strings.Join(lines, "\n"))
	}
	if input.ToolResults {
		add("tool_results", strings.Join(r.toolResults, "\n\n"))
	}
	if input.Context {
		add("context", r.context)
	}
	if input.Answer {
		add("answer", r.answer)
	}
	request := modelservice.Request{State: r.request}
	if len(parts) > 0 {
		add("request", r.request)
		request = modelservice.Request{Parts: parts}
	}
	if input.Image {
		request.Images = slices.Clone(r.images)
	}
	if input.Video {
		request.Videos = slices.Clone(r.videos)
	}
	return request
}

// systemOneCall is the rules a stage asks one deployment about one state.
type systemOneCall struct {
	deployment string
	request    modelservice.Request
	rules      []config.SystemOneRule
}

// systemOneCalls groups rules by the deployment that answers them and the
// state they read, in rule order: every rule of a stage goes to its
// deployment in one call, its other states as further states of it.
func (c *Classifier) systemOneCalls(rules []config.SystemOneRule, reading systemOneReading) []*systemOneCall {
	var calls []*systemOneCall
	for _, rule := range rules {
		deployment := c.Config.SystemOneRuleDeployment(rule)
		state := reading.state(rule)
		index := slices.IndexFunc(calls, func(call *systemOneCall) bool {
			return call.deployment == deployment && sameState(call.request, state)
		})
		if index < 0 {
			calls = append(calls, &systemOneCall{deployment: deployment, request: state})
			index = len(calls) - 1
		}
		calls[index].rules = append(calls[index].rules, rule)
	}
	return calls
}

func sameState(a, b modelservice.Request) bool {
	if a.State != b.State || len(a.Parts) != len(b.Parts) || !slices.Equal(a.Images, b.Images) || !slices.Equal(a.Videos, b.Videos) {
		return false
	}
	for name, text := range a.Parts {
		if other, ok := b.Parts[name]; !ok || other != text {
			return false
		}
	}
	return true
}

// evaluateSystemOneSignals asks every used request-stage rule, one call per
// deployment, with the request as it came as the current turn. Calls to
// different deployments run in parallel. A rule that names no deployment
// asks the decision model, like the built-in signals whose call it joins. A
// failed or late call leaves its signals unknown; it never fails the request.
func (c *Classifier) evaluateSystemOneSignals(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	reading systemOneReading,
	usedSignals map[string]bool,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	start := time.Now()
	var rules []config.SystemOneRule
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageRequest && signalRuleUsed(usedSignals, config.SignalTypeSystemOne, rule.Name) {
			rules = append(rules, rule)
		}
	}
	c.askSystemOneRules(ctx, results, mu, rules, reading)
	mu.Lock()
	results.Metrics.SystemOne.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
	mu.Unlock()
}

// ResponseInput is what a response-stage rule reads: the request that was
// answered, the model's answer and its grounding context (retrieved
// documents and tool results), and the conversation of the request.
type ResponseInput struct {
	Request           string
	Answer            string
	Context           string
	PriorUserMessages []string
	Turns             []ConversationTurn
	ToolResultTexts   []string
	Images            []string
	Videos            []string
}

// ResponseSystemOneSignal is the response-stage systemone observation for one
// response, in the shape the decision engine consumes request-stage signals in.
type ResponseSystemOneSignal struct {
	MatchedRules []string
	Confidences  map[string]float64
	Values       map[string]float64
	Errors       map[string]string
}

// EvaluateResponseSystemOne asks the named response-stage rules about one
// answer, one call per deployment, as the request stage asks its rules. A
// name that is no response-stage rule is skipped.
func (c *Classifier) EvaluateResponseSystemOne(ctx context.Context, input ResponseInput, names []string) *ResponseSystemOneSignal {
	if ctx == nil {
		ctx = context.Background()
	}
	var rules []config.SystemOneRule
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageResponse && slices.Contains(names, rule.Name) {
			rules = append(rules, rule)
		}
	}
	results := &SignalResults{SignalConfidences: map[string]float64{}, SignalValues: map[string]float64{}, SignalErrors: map[string]string{}}
	var mu sync.Mutex
	c.askSystemOneRules(ctx, results, &mu, rules, systemOneReading{
		request: input.Request, history: input.PriorUserMessages, turns: input.Turns, toolResults: input.ToolResultTexts,
		answer: input.Answer, context: input.Context, images: input.Images, videos: input.Videos,
	})
	return &ResponseSystemOneSignal{
		MatchedRules: results.MatchedSystemOneRules, Confidences: results.SignalConfidences,
		Values: results.SignalValues, Errors: results.SignalErrors,
	}
}

func (c *Classifier) askSystemOneRules(ctx context.Context, results *SignalResults, mu *sync.Mutex, rules []config.SystemOneRule, reading systemOneReading) {
	calls := c.systemOneCalls(rules, reading)
	modelservice.Fan(ctx, len(calls), func(i int) {
		c.evaluateSystemOneDeployment(ctx, results, mu, calls[i])
	})
}

func (c *Classifier) evaluateSystemOneDeployment(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	call *systemOneCall,
) {
	deployment, rules := call.deployment, call.rules
	// The deployment's scan budget keeps a stage's questions to it in one
	// call: a full-coverage rule reads its input whole, up to the budget, and
	// any other only its first tokens.
	request := call.request
	request.MaxTokens = c.Config.ModelDeployments[deployment].ScanBudget()
	for _, rule := range rules {
		request.Questions = append(request.Questions, systemOneRuleQuestion(rule))
	}
	remote := c.Config.IsSystemOneDeployment(deployment)
	callCtx, cancel := systemOneCallContext(ctx, rules, remote, c.Config.ModelDeployments[deployment].CallTimeout())
	defer cancel()
	started := time.Now()
	var response modelservice.Response
	var err error
	if remote {
		response, err = askSystemOneDeployment(callCtx, c.systemOne(), deployment, request)
	} else if card, ok := c.decisionTaskCard(deployment); ok {
		response, err = modelservice.ExecuteQuestions(callCtx, c.decider(), deployment, card, request)
	} else if c.models != nil && c.models.runtime != nil && c.models.runtime.Services() != nil {
		err = modelservice.ErrUnavailable
	} else {
		response, err = c.decider().Decide(callCtx, deployment, request)
	}
	latency := time.Since(started).Seconds()
	if err != nil {
		modelservice.RecordUnknown(deployment, modelservice.ErrorReason(err), len(rules))
	}
	mu.Lock()
	defer mu.Unlock()
	for _, rule := range rules {
		key := signalConfidenceKey(config.SignalTypeSystemOne, rule.Name)
		c.recordSignalExtraction(config.SignalTypeSystemOne, rule.Name, latency)
		if err != nil {
			results.SignalErrors[key] = systemOneSignalErrorPrefix + modelservice.ErrorReason(err)
			continue
		}
		answer, ok := response.Answers[SystemOneQuestionID(rule.Name)]
		if !ok || answer.Error != "" {
			reason := answer.Error
			if !ok {
				reason = "missing_answer"
			}
			if ok && rule.FullCoverage() && unscanned(reason) && rule.EffectiveOnUnscanned() != config.SystemOneUnscannedUnknown {
				c.applyUnscanned(results, rule)
				continue
			}
			results.SignalErrors[key] = systemOneSignalErrorPrefix + reason
			modelservice.RecordUnknown(deployment, reason, 1)
			continue
		}
		for _, matched := range applySystemOneAnswer(results, rule, answer) {
			results.MatchedSystemOneRules = append(results.MatchedSystemOneRules, matched)
			c.recordSignalMatch(config.SignalTypeSystemOne, matched)
		}
	}
}

// systemOneRuleQuestion is the question a rule asks: a full-coverage rule
// reads its whole input, which ExecuteQuestions requires proof of, and any
// other only its first tokens.
func systemOneRuleQuestion(rule config.SystemOneRule) modelservice.Question {
	question := SystemOneQuestion(SystemOneQuestionID(rule.Name), rule.Question)
	question.TaskID, question.Stage = config.SignalTypeSystemOne, rule.EffectiveStage()
	question.Truncate = !rule.FullCoverage()
	return question
}

// unscanned reports whether an answer failed because the model could not
// read the whole input.
func unscanned(reason string) bool {
	switch reason {
	case "scan_budget_exceeded", "input_coverage_unknown", "max_length_exceeded":
		return true
	}
	return false
}

// applyUnscanned resolves a full-coverage rule whose model could not read
// the whole input as its on_unscanned says, when that is not unknown: the
// rule and its counted labels match, or nothing does. An unscanned answer
// has no values, so a condition's own predicate does not match it.
func (c *Classifier) applyUnscanned(results *SignalResults, rule config.SystemOneRule) {
	if rule.EffectiveOnUnscanned() != config.SystemOneUnscannedMatch {
		return
	}
	var matched []string
	if rule.MatchesItself() {
		matched = append(matched, rule.Name)
	}
	for _, label := range rule.CountedLabels() {
		matched = append(matched, rule.Name+":"+label)
	}
	for _, reference := range matched {
		results.MatchedSystemOneRules = append(results.MatchedSystemOneRules, reference)
		c.recordSignalMatch(config.SignalTypeSystemOne, reference)
	}
}

// askSystemOneDeployment asks a systemone deployment's remote API. Without
// its client the questions stay unknown.
func askSystemOneDeployment(ctx context.Context, remote modelservice.Remote, deployment string, request modelservice.Request) (modelservice.Response, error) {
	if remote == nil {
		return modelservice.Response{}, modelservice.ErrUnavailable
	}
	card, ok := remote.Card(deployment)
	if !ok {
		return modelservice.Response{}, modelservice.ErrUnknownDeployment
	}
	return modelservice.ExecuteQuestions(ctx, remote, deployment, card, request)
}

// systemOneCallContext bounds a deployment's call by the shortest timeout its
// rules set. Questions to a model_runtime decision model that set none join
// the built-in signals' call, so they take its deadline rather than the
// default timeout. Questions to a systemone deployment travel with the
// built-in signals that ask it too, but a rule without a timeout takes the
// deployment's own, else the default: the remote API's latency is its own.
// Once the stage sends a call, its questions wait for it as long as its
// latest caller does: a timeout bounds the wait for the other askers, never
// an answer of a call the stage waits for anyway (modelservice.Bundle).
func systemOneCallContext(ctx context.Context, rules []config.SystemOneRule, remote bool, deploymentTimeout time.Duration) (context.Context, context.CancelFunc) {
	var timeout time.Duration
	for _, rule := range rules {
		if rule.Deployment == "" && rule.TimeoutMs == 0 && !remote {
			continue
		}
		t := rule.EffectiveTimeout()
		if remote && rule.TimeoutMs == 0 && deploymentTimeout > 0 {
			t = deploymentTimeout
		}
		if timeout == 0 || t < timeout {
			timeout = t
		}
	}
	if timeout == 0 {
		return context.WithCancel(ctx)
	}
	return context.WithTimeout(ctx, timeout)
}

// SystemOneQuestion converts a configured question to the runtime request form.
func SystemOneQuestion(id string, question config.SystemOneQuestion) modelservice.Question {
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

// applySystemOneAnswer publishes an answer's values and returns the matched
// references: the rule name when the rule matches itself, and "rule:label"
// for the chosen option (choice) or every matching label (set, span). A
// value is published under "systemone:<rule>" (P(yes), the expected level,
// the chosen option's probability, or the highest probability of a counted
// label) and under "systemone:<rule>:<label>" for every option or label.
func applySystemOneAnswer(results *SignalResults, rule config.SystemOneRule, answer modelservice.Answer) []string {
	key := signalConfidenceKey(config.SignalTypeSystemOne, rule.Name)
	predicate := rule.EffectivePredicate()
	publish := func(name string, value float64) {
		results.SignalValues[name] = value
		results.SignalConfidences[name] = value
	}
	switch rule.Question.Type {
	case config.SystemOneQuestionNoul:
		publish(key, answer.Noul)
		if predicateMatches(answer.Noul, predicate) {
			return []string{rule.Name}
		}
	case config.SystemOneQuestionScore:
		results.SignalValues[key] = answer.Score
		results.SignalConfidences[key] = maxProbability(answer.Probabilities)
		if predicateMatches(answer.Score, predicate) {
			return []string{rule.Name}
		}
	case config.SystemOneQuestionSet, config.SystemOneQuestionSpan:
		return applyLabelledAnswer(rule, answer, predicate, key, publish)
	default:
		for option, probability := range answer.Probabilities {
			publish(key+":"+option, probability)
		}
		chosen := answer.Probabilities[answer.Choice]
		publish(key, chosen)
		if answer.Choice == "" || (predicate != nil && !predicateMatches(chosen, predicate)) {
			return nil
		}
		matched := []string{rule.Name + ":" + answer.Choice}
		if slices.Contains(rule.Labels, answer.Choice) {
			matched = append(matched, rule.Name)
		}
		return matched
	}
	return nil
}

// applyLabelledAnswer publishes each Set label's probability, or each Span
// label's most probable span (0 when the model found none), and the highest
// of the counted labels under the rule itself. A label matches when its
// probability (a Set) or one of its spans' (a Span) meets the predicate, or
// without one when the model selected it or found a span of it; an ignored
// label never does. The rule matches when a counted label does.
func applyLabelledAnswer(rule config.SystemOneRule, answer modelservice.Answer, predicate *config.NumericPredicate, key string, publish func(string, float64)) []string {
	counted := rule.CountedLabels()
	var matched []string
	best, matchesItself := 0.0, false
	for _, label := range rule.Question.Labels {
		probability, found := labelEvidence(rule.Question.Type, answer, label.Key, predicate)
		publish(key+":"+label.Key, probability)
		if !slices.Contains(counted, label.Key) {
			continue
		}
		best = max(best, probability)
		if found {
			matched = append(matched, rule.Name+":"+label.Key)
			matchesItself = true
		}
	}
	publish(key, best)
	if matchesItself {
		matched = append([]string{rule.Name}, matched...)
	}
	return matched
}

// labelEvidence is one label's published probability and whether it
// matches.
func labelEvidence(questionType string, answer modelservice.Answer, label string, predicate *config.NumericPredicate) (float64, bool) {
	if questionType == config.SystemOneQuestionSet {
		probability := answer.Probabilities[label]
		if predicate == nil {
			return probability, slices.Contains(answer.Selected, label)
		}
		return probability, predicateMatches(probability, predicate)
	}
	best, found := 0.0, false
	for _, span := range answer.Spans {
		if span.Label != label {
			continue
		}
		best = max(best, span.Probability)
		found = found || predicate == nil || predicateMatches(span.Probability, predicate)
	}
	return best, found
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
