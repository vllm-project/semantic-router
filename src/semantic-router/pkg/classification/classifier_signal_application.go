package classification

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// An application signal's rules ask their preset through the systemone path
// (config.ApplicationRules): in a request stage, every rule asks its
// deployment in the stage's one call to it, rules of one preset about one
// state share its question, and each publishes its matches and values under
// its own type.

// evaluateApplicationSignals asks the used request-stage rules of one
// application signal type.
func (c *Classifier) evaluateApplicationSignals(
	ctx context.Context,
	results *SignalResults,
	mu *sync.Mutex,
	signalType string,
	reading systemOneReading,
	usedSignals map[string]bool,
) {
	if ctx == nil {
		ctx = context.Background()
	}
	var rules []askedRule
	for _, rule := range c.applicationRules {
		if rule.Type == signalType && rule.Rule.EffectiveStage() == config.SystemOneStageRequest && signalRuleUsed(usedSignals, signalType, rule.Rule.Name) {
			rules = append(rules, applicationAsked(rule))
		}
	}
	if len(rules) == 0 {
		return
	}
	start := time.Now()
	c.askSystemOneRules(ctx, results, mu, rules, reading)
	mu.Lock()
	defer mu.Unlock()
	if results.Metrics != nil {
		if metrics := results.Metrics.metricsOf(signalType); metrics != nil {
			metrics.ExecutionTimeMs = float64(time.Since(start).Microseconds()) / 1000.0
		}
	}
	switch signalType {
	case config.SignalTypeDomain:
		results.DomainClassification = domainClassification(results, rules[0].rule)
	case config.SignalTypeJailbreak:
		summarizeJailbreak(results, rules)
	case config.SignalTypePII:
		summarizePII(results)
	}
}

// domainClassification is the subject area the domains rules' shared
// question chose, and its probability.
func domainClassification(results *SignalResults, rule config.SystemOneRule) *DomainClassificationResult {
	key := signalConfidenceKey(config.SignalTypeDomain, rule.Name)
	if _, failed := results.SignalErrors[key]; failed {
		return &DomainClassificationResult{}
	}
	classification := &DomainClassificationResult{}
	for _, option := range rule.Question.OptionKeys() {
		if probability, ok := results.SignalValues[key+":"+option]; ok && probability > classification.Confidence {
			classification.Category, classification.Confidence, classification.ConfidenceAvailable = option, probability, true
		}
	}
	return classification
}

// JailbreakUnscannedType is the jailbreak type reported when a rule matches
// because the decision model did not read all of the content: a guard does
// not pass content it did not read unless its on_unscanned allows it. It is a
// policy match, not model evidence.
const JailbreakUnscannedType = "unscanned"

// JailbreakUnavailableType is the jailbreak type reported when every rule
// that matches does so because its deployment gave no answer (on_unknown:
// match). It is a policy match, not model evidence.
const JailbreakUnavailableType = "unavailable"

// summarizeJailbreak records the attack the request-stage jailbreak rules
// saw: whether one matched, and the highest probability of an attack.
func summarizeJailbreak(results *SignalResults, rules []askedRule) {
	for _, asked := range rules {
		key := signalConfidenceKey(config.SignalTypeJailbreak, asked.rule.Name)
		if probability, ok := results.SignalValues[key]; ok {
			results.JailbreakScoreAvailable = true
			results.JailbreakConfidence = max(results.JailbreakConfidence, float32(probability))
		}
	}
	evidence, unread := false, true
	for _, reference := range results.MatchedJailbreakRules {
		if strings.Contains(reference, ":") {
			continue
		}
		results.JailbreakDetected = true
		key := signalConfidenceKey(config.SignalTypeJailbreak, reference)
		evidence = evidence || !results.SignalErrorMatches[key]
		unread = unread && unscanned(strings.TrimPrefix(results.SignalErrors[key], systemOneSignalErrorPrefix))
	}
	switch {
	case !results.JailbreakDetected:
	case !evidence && unread:
		results.JailbreakType = JailbreakUnscannedType
	case !evidence:
		results.JailbreakType = JailbreakUnavailableType
	case results.JailbreakScoreAvailable:
		results.JailbreakType = presets.Jailbreak
	default:
		results.JailbreakType = JailbreakUnscannedType
	}
}

// summarizePII records the personal data the request-stage pii rules found:
// whether one matched, and the labels it matched.
func summarizePII(results *SignalResults) {
	for _, reference := range results.MatchedPIIRules {
		rule, label, labelled := strings.Cut(reference, ":")
		if !labelled {
			results.PIIDetected = results.PIIDetected || rule != ""
			continue
		}
		if !slices.Contains(results.PIIEntities, label) {
			results.PIIEntities = append(results.PIIEntities, label)
		}
	}
}

// recordPrivacyEvidence records what a call's pii question read of the
// request text alone: Router Replay captures a prompt only when a complete
// read found no personal data in it.
func recordPrivacyEvidence(results *SignalResults, call *systemOneCall, response modelservice.Response, err error) {
	if !call.requestAlone || !slices.ContainsFunc(call.rules, func(asked askedRule) bool { return asked.preset == presets.PII }) {
		return
	}
	answer, answered := response.Answers[presetQuestionID(presets.PII)]
	complete := err == nil && answered && answer.Error == "" && answer.InputCoverage == "complete"
	evidence := NewPrivacyEvidence("request", call.request.State, complete, complete && len(answer.Spans) == 0)
	if !slices.Contains(results.PIIEvidence, evidence) {
		results.PIIEvidence = append(results.PIIEvidence, evidence)
	}
}

// ReplayPromptReadable reports whether ReplayPromptEvidence could read prompt
// whole within within: a prompt that one window of each pii rule's deployment
// holds is one call, and is always readable. A longer one is not readable
// when it needs more windows than one scan reads, or when its windows would
// take longer than scanReadShare of within at the deployment's measured pace.
// Before the deployment has a pace, it is readable unless the request's own
// guard scans ran out of time on its input (guardReadShort). Once it has one,
// the pace decides alone: a guard scan may read more than the prompt (earlier
// turns, tool results), or end in part on windows that failed, and its
// deadline is not within.
func (c *Classifier) ReplayPromptReadable(prompt string, within time.Duration, guardReadShort bool) bool {
	if c == nil || c.Config == nil {
		return true
	}
	for _, rule := range c.applicationRules {
		if rule.Type != config.SignalTypePII || rule.Rule.Input != (config.SystemOneInput{}) {
			continue
		}
		deployment := c.Config.SystemOneRuleDeployment(applicationAsked(rule).rule)
		budget := c.scanWindowBudget(deployment)
		if len(prompt) <= budget.size {
			continue
		}
		windows, grown := cutScanWindows(prompt, budget.size, budget.most, budget.overlap)
		if len(windows) > maxScanWindows {
			return false
		}
		perCall, perByte, ok := c.windowPaces.line(deployment, budget.parallel)
		if !ok {
			if guardReadShort {
				return false
			}
			continue
		}
		if float64(len(windows))*(perCall+perByte*float64(grown)) > scanReadShare*within.Seconds() {
			return false
		}
	}
	return true
}

// ReplayPromptEvidence reads a prompt with the recipe's pii rules, for Router
// Replay's privacy check of a prompt routing did not read alone.
func (c *Classifier) ReplayPromptEvidence(ctx context.Context, prompt string) PrivacyEvidence {
	unverified := NewPrivacyEvidence("request", prompt, false, false)
	if c == nil || c.Config == nil || prompt == "" {
		return unverified
	}
	var rules []askedRule
	for _, rule := range c.applicationRules {
		if rule.Type == config.SignalTypePII && rule.Rule.Input == (config.SystemOneInput{}) {
			rules = append(rules, applicationAsked(rule))
		}
	}
	if len(rules) == 0 {
		return unverified
	}
	results := newResponseSignalResults()
	c.askSystemOneRules(ctx, results, &sync.Mutex{}, rules, systemOneReading{request: prompt})
	for _, evidence := range results.PIIEvidence {
		if evidence.Stage == unverified.Stage && evidence.InputSHA256 == unverified.InputSHA256 {
			return evidence
		}
	}
	return unverified
}

// EvaluateResponseSignals asks the response-stage rules include selects (the
// systemone rules and the application rules with stage: response, by type and
// name) about one answer, one call per deployment, as the request stage asks
// its rules. A rule that reads a part the answer does not have, its text or
// its grounding context, is unknown without a call.
func (c *Classifier) EvaluateResponseSignals(ctx context.Context, input ResponseInput, include func(signalType, name string) bool) *ResponseSignal {
	if ctx == nil {
		ctx = context.Background()
	}
	results := newResponseSignalResults()
	var rules []askedRule
	ask := func(asked askedRule) {
		if !include(asked.signalType, asked.rule.Name) {
			return
		}
		if part := missingResponsePart(asked.rule.Input, input); part != "" {
			results.SignalErrors[signalConfidenceKey(asked.signalType, asked.rule.Name)] = systemOneSignalErrorPrefix + part + "_unavailable"
			return
		}
		rules = append(rules, asked)
	}
	for _, rule := range c.Config.SystemOneRules {
		if rule.EffectiveStage() == config.SystemOneStageResponse {
			ask(systemOneAsked(rule))
		}
	}
	for _, rule := range c.applicationRules {
		if rule.Rule.EffectiveStage() == config.SystemOneStageResponse {
			ask(applicationAsked(rule))
		}
	}
	c.askSystemOneRules(ctx, results, &sync.Mutex{}, rules, input.reading())
	signal := &ResponseSignal{
		Matched: map[string][]string{}, Reported: map[string][]string{}, Confidences: results.SignalConfidences,
		Values: results.SignalValues, Errors: results.SignalErrors, Spans: results.SignalSpans,
	}
	for _, asked := range rules {
		if matched := results.matchedRulesOf(asked.signalType); matched != nil && len(*matched) > 0 {
			signal.Matched[asked.signalType] = slices.Clone(*matched)
		}
		if reported := results.ReportedMatches(asked.signalType); len(reported) > 0 {
			signal.Reported[asked.signalType] = reported
		}
	}
	return signal
}

// missingResponsePart names the part of the answer a rule reads that it does
// not have: "answer" for an answer without text, such as one made of tool
// calls, and "context" for an answer with no grounding context to check it
// against.
func missingResponsePart(input config.SystemOneInput, response ResponseInput) string {
	switch {
	case input.Answer && strings.TrimSpace(response.Answer) == "":
		return "answer"
	case input.Context && strings.TrimSpace(response.Context) == "":
		return "context"
	}
	return ""
}

// PIIRuleSpans returns the personal data the named pii rule finds in text:
// the spans of its labels, other than those it ignores, that satisfy its
// predicate, and every other copy of their values (CoverRepeatedSpans), with
// offsets in code points. Masking replaces these spans. An
// error, including an answer that read only part of the text, means the text
// could not be read whole, so a caller that masks must not send it.
func (c *Classifier) PIIRuleSpans(ctx context.Context, ruleName string, text string) ([]modelservice.Span, error) {
	index := slices.IndexFunc(c.applicationRules, func(rule config.ApplicationRule) bool {
		return rule.Type == config.SignalTypePII && rule.Rule.Name == ruleName
	})
	if index < 0 {
		return nil, fmt.Errorf("pii rule %q is not declared in this recipe", ruleName)
	}
	return c.askedSpans(ctx, applicationAsked(c.applicationRules[index]), text)
}

// RuleSpans returns what a rule that finds text finds in text, in code
// points: a pii rule's personal data (PIIRuleSpans), a systemone span rule's
// spans, read the same way and labeled with the rule's name, or a keyword
// rule's places. An error means the text could not be read whole.
func (c *Classifier) RuleSpans(ctx context.Context, signalType, name, text string) ([]modelservice.Span, error) {
	switch signalType {
	case config.SignalTypePII:
		return c.PIIRuleSpans(ctx, name, text)
	case config.SignalTypeKeyword:
		return c.keywordClassifier.RuleSpans(name, text)
	case config.SignalTypeSystemOne:
		index := slices.IndexFunc(c.Config.SystemOneRules, func(rule config.SystemOneRule) bool { return rule.Name == name })
		if index < 0 || c.Config.SystemOneRules[index].Question.Type != config.SystemOneQuestionSpan {
			return nil, fmt.Errorf("systemone span rule %q is not declared in this recipe", name)
		}
		// Masking acts on the whole text, so the rule reads all of it,
		// whatever coverage a decision condition on it asks.
		rule := c.Config.SystemOneRules[index]
		rule.Coverage = config.SystemOneCoverageFull
		spans, err := c.askedSpans(ctx, systemOneAsked(rule), text)
		for i := range spans {
			spans[i].Label = name
		}
		return spans, err
	}
	return nil, fmt.Errorf("a %s rule does not find text", signalType)
}

// askedSpans asks one rule that finds text about text, in windows when it is
// longer than one, and requires every part of it to be read whole.
func (c *Classifier) askedSpans(ctx context.Context, asked askedRule, text string) ([]modelservice.Span, error) {
	deployment := c.Config.SystemOneRuleDeployment(asked.rule)
	if scan := newGuardScan(deployment, modelservice.Request{State: text}, c.scanWindowBudget(deployment)); scan != nil {
		return c.scanSpans(ctx, text, asked, scan)
	}
	call := &systemOneCall{deployment: deployment, request: modelservice.Request{State: text}, rules: []askedRule{asked}}
	var answer modelservice.Answer
	var answered bool
	var askErr error
	c.askCall(ctx, newResponseSignalResults(), &sync.Mutex{}, call, func(response modelservice.Response, err error) {
		answer, answered = response.Answers[asked.questionID()]
		askErr = err
	})
	switch {
	case askErr != nil:
		return nil, askErr
	case !answered:
		return nil, fmt.Errorf("%s rule %q: %w: missing answer", asked.signalType, asked.rule.Name, modelservice.ErrUnavailable)
	case answer.Error != "":
		return nil, fmt.Errorf("%s rule %q: %w: %s", asked.signalType, asked.rule.Name, modelservice.ErrUnavailable, answer.Error)
	case answer.InputCoverage != "complete":
		return nil, fmt.Errorf("%s rule %q read only part of the text", asked.signalType, asked.rule.Name)
	}
	return spansInText(asked, text, matchedSpans(asked.rule, answer))
}

// scanSpans is askedSpans of a text longer than one window: every window
// must be read whole, and their spans count at their places in the text.
func (c *Classifier) scanSpans(ctx context.Context, text string, asked askedRule, scan *guardScan) ([]modelservice.Span, error) {
	scan.rules = []askedRule{asked}
	readers, release := c.windowReaders(ctx, []*guardScan{scan})
	modelservice.Fan(ctx, len(readers), func(i int) { readers[i]() })
	release()
	read, covered, reason := scan.answers(asked)
	if covered < scan.cut {
		return nil, fmt.Errorf("%s rule %q read %d of the text's %d windows: %w: %s", asked.signalType, asked.rule.Name, covered, scan.cut, modelservice.ErrUnavailable, reason)
	}
	for _, entry := range read {
		if entry.answer.InputCoverage != "complete" {
			return nil, fmt.Errorf("%s rule %q read only part of the text", asked.signalType, asked.rule.Name)
		}
	}
	return spansInText(asked, text, matchedSpans(asked.rule, mergedSpanAnswer(read, scan)))
}

// spansInText refuses an answer with a span outside text, whose offsets
// masking would otherwise clamp onto a range that leaves part of the value in
// clear text, and adds every other copy of the values (CoverRepeatedSpans).
func spansInText(asked askedRule, text string, spans []modelservice.Span) ([]modelservice.Span, error) {
	points := utf8.RuneCountInString(text)
	for _, span := range spans {
		if span.Start < 0 || span.End > points || span.Start >= span.End {
			return nil, fmt.Errorf("%s rule %q: %w: span [%d,%d) is outside the text's %d code points",
				asked.signalType, asked.rule.Name, modelservice.ErrUnavailable, span.Start, span.End, points)
		}
	}
	return CoverRepeatedSpans(text, spans), nil
}

// AskPreset asks the deployment an application signal type's rules that name
// none resolve to one question of its preset about text, for the
// classification APIs, which answer outside a routing rule.
func (c *Classifier) AskPreset(ctx context.Context, signalType, preset, text string) (modelservice.Answer, error) {
	return c.askPreset(ctx, signalType, preset, presets.StageRequest, systemOneReading{request: text})
}

// GroundingBackends returns this recipe's grounding functions: the fusion
// looper's groundedness check asks the hallucination preset whether an
// answer is supported by its reference (the request's context, or the peer
// answers), as a hallucination rule would.
func (c *Classifier) GroundingBackends() *looper.GroundingBackends {
	if c == nil || c.Config == nil {
		return nil
	}
	return &looper.GroundingBackends{Detect: func(ctx context.Context, contextText, question, answer string) (looper.GroundingEvidence, error) {
		reading := systemOneReading{request: question, answer: answer, context: contextText}
		answered, err := c.askPreset(ctx, config.SignalTypeHallucination, presets.Hallucination, presets.StageResponse, reading)
		if err != nil {
			return looper.GroundingEvidence{}, err
		}
		evidence := looper.GroundingEvidence{Unsupported: len(answered.Spans) > 0}
		probability := float32(0)
		for _, span := range answered.Spans {
			evidence.Spans = append(evidence.Spans, span.Text)
			probability = max(probability, float32(span.Probability))
		}
		evidence.Probability = &probability
		return evidence, nil
	}}
}

// askPreset asks one question of a preset in a stage about what reading
// holds, of the deployment the type's rules that name none resolve to.
func (c *Classifier) askPreset(ctx context.Context, signalType, preset, stage string, reading systemOneReading) (modelservice.Answer, error) {
	if c == nil || c.Config == nil {
		return modelservice.Answer{}, fmt.Errorf("%w: no classifier", modelservice.ErrUnavailable)
	}
	asked, deployment, err := c.presetAsked(signalType, preset, stage)
	if err != nil {
		return modelservice.Answer{}, err
	}
	results := newResponseSignalResults()
	var answer modelservice.Answer
	var askErr error
	call := &systemOneCall{deployment: deployment, request: reading.state(asked.rule), rules: []askedRule{asked}}
	c.askCall(ctx, results, &sync.Mutex{}, call, func(response modelservice.Response, err error) {
		answer, askErr = response.Answers[asked.questionID()], err
	})
	if askErr != nil {
		return modelservice.Answer{}, askErr
	}
	if answer.Error != "" {
		return answer, fmt.Errorf("%w: %s", modelservice.ErrUnavailable, answer.Error)
	}
	return answer, nil
}

// presetAsked is the rule one question of a preset in a stage is asked as,
// and the deployment it asks: the one the type's rules that name none resolve
// to, as such a rule asks it (the decision model bounded only by the caller,
// a bound deployment by its call timeout).
func (c *Classifier) presetAsked(signalType, preset, stage string) (askedRule, string, error) {
	canonical, ok := presets.Lookup(preset, stage)
	if !ok {
		return askedRule{}, "", fmt.Errorf("unknown preset %q", preset)
	}
	deployment, source := c.Config.ApplicationTypeDeployment(signalType)
	rule := config.SystemOneRule{
		Name: signalType, Question: config.PresetSystemOneQuestion(canonical), Deployment: deployment, Stage: canonical.Stage,
		Input: config.SystemOneInput{Answer: canonical.Input.Answer, Context: canonical.Input.Context},
	}
	if source == "default" {
		rule.Deployment = ""
	}
	if canonical.Full {
		rule.Coverage = config.SystemOneCoverageFull
	}
	return askedRule{signalType: signalType, rule: rule, preset: preset}, deployment, nil
}
