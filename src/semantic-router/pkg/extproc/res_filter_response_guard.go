package extproc

import (
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// responseConditions reads a guard's rule trees over one answer and reports
// the response-stage observations they made; the recipe's classifier provides
// it (classification.ResponseConditions).
type responseConditions interface {
	Evaluate(tree config.RuleNode) decision.RuleOutcome
	Observed() *classification.ResponseSignal
}

// responseGuardRun is what the selected decision's response_guard made of one
// answer: each rule's outcome in order, up to the first that blocks, and the
// observations the rules read.
type responseGuardRun struct {
	conditions responseConditions
	rules      []responseGuardRuleRun
	// streamed marks an answer whose bytes were with the client before it
	// existed as a whole: the rules were read, and nothing was applied.
	streamed bool
}

// responseGuardRuleRun is one rule's outcome. Action is what the rule does
// with the answer: its own action when it matched, the guard's on_unknown when
// it could not be decided, and "" when it did not match.
type responseGuardRuleRun struct {
	state        decision.RuleState
	action       string
	codes        []string
	details      bool
	matchedRules []string
}

// runResponseGuard reads the selected decision's response_guard rules over
// answer, in order, until one blocks, and returns nil when the decision has no
// guard. Without the recipe's classifier no response-stage rule can be read,
// so every rule is unknown. A streamed answer is read the same way, and
// nothing of it is applied.
func (r *OpenAIRouter) runResponseGuard(ctx *RequestContext, answer string, streamed bool) *responseGuardRun {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	guard := ctx.VSRSelectedDecision.GetResponseGuardConfig()
	if guard == nil {
		return nil
	}
	run := &responseGuardRun{streamed: streamed}
	classifier := r.classifierForRequest(ctx)
	if classifier != nil {
		run.conditions = classifier.NewResponseConditions(selectionRequestContext(ctx), ctx.RoutedSignals, ctx.responseInput(answer))
	}
	cfg := classifierConfig(classifier)
	if cfg == nil {
		cfg = &config.RouterConfig{}
	}
	for _, rule := range guard.Rules {
		ruleRun := responseGuardRuleRun{state: decision.RuleUnknown, details: rule.Details}
		if run.conditions != nil {
			outcome := run.conditions.Evaluate(rule.When)
			ruleRun.state, ruleRun.matchedRules = outcome.State, outcome.MatchedRules
		}
		switch ruleRun.state {
		case decision.RuleTrue:
			ruleRun.action = rule.Action
		case decision.RuleUnknown:
			ruleRun.action = guard.EffectiveOnUnknown()
		}
		ruleRun.codes = responseGuardCodes(cfg, rule.When, ruleRun.state)
		run.rules = append(run.rules, ruleRun)
		if ruleRun.action == config.ResponseGuardBlock {
			break
		}
	}
	metrics.RecordPluginExecution(config.DecisionPluginResponseGuard, requestDecisionStateKey(ctx), run.outcome(), 0)
	if slices.ContainsFunc(run.rules, func(rule responseGuardRuleRun) bool { return rule.state == decision.RuleUnknown }) {
		metrics.RecordPluginError(config.DecisionPluginResponseGuard, "signal_unavailable")
	}
	return run
}

// responseInput is what response-stage rules read about the request and its
// answer: the grounding context is the retrieved documents and the tool
// results of the request as it was sent.
func (ctx *RequestContext) responseInput(answer string) classification.ResponseInput {
	input := ctx.ResponseStage
	input.Answer = answer
	var grounding []string
	if ctx.RAGRetrievedContext != "" {
		grounding = append(grounding, ctx.RAGRetrievedContext)
	}
	if ctx.SemanticRequest != nil {
		for _, message := range ctx.SemanticRequest.Messages {
			for _, content := range message.Content {
				if content.Kind != llmprotocol.ContentToolResult || content.ToolResult == nil {
					continue
				}
				if _, retrieved := ctx.RAGToolCallIDs[content.ToolResult.CallID]; retrieved {
					continue
				}
				if text := semanticText(content.ToolResult.Content); text != "" {
					grounding = append(grounding, text)
				}
			}
		}
	}
	input.Context = strings.Join(grounding, "\n\n")
	return input
}

// responseGuardCodes are the codes an annotating rule adds to
// x-vsr-response-warnings, one per response-stage signal type the rule reads:
// hallucination (unverified_factual when the answer could not be checked),
// response_jailbreak, and response_guard for any other rule.
func responseGuardCodes(cfg *config.RouterConfig, when config.RuleNode, state decision.RuleState) []string {
	var codes []string
	var walk func(node *config.RuleNode)
	walk = func(node *config.RuleNode) {
		if !node.IsLeaf() {
			for i := range node.Conditions {
				walk(&node.Conditions[i])
			}
			return
		}
		signalType := strings.ToLower(strings.TrimSpace(node.Type))
		if cfg.SignalStageOf(signalType, node.Name) != config.SignalStageResponse {
			return
		}
		code := headers.ResponseWarningGuard
		switch {
		case signalType == config.SignalTypeHallucination && state == decision.RuleUnknown:
			code = headers.ResponseWarningUnverifiedFactual
		case signalType == config.SignalTypeHallucination:
			code = headers.ResponseWarningHallucination
		case signalType == config.SignalTypeJailbreak:
			code = headers.ResponseWarningJailbreak
		}
		if !slices.Contains(codes, code) {
			codes = append(codes, code)
		}
	}
	walk(&when)
	if len(codes) == 0 {
		codes = []string{headers.ResponseWarningGuard}
	}
	return codes
}

// classifierConfig returns the router config a classifier was built from, or
// nil when there is no classifier.
func classifierConfig(classifier *classification.Classifier) *config.RouterConfig {
	if classifier == nil {
		return nil
	}
	return classifier.Config
}

func signalKey(signalType, name string) string {
	return signalType + ":" + name
}

// ruleOutcomeName names a rule's state in replay records and previews.
func ruleOutcomeName(state decision.RuleState) string {
	switch state {
	case decision.RuleTrue:
		return "matched"
	case decision.RuleUnknown:
		return "unknown"
	}
	return "not_matched"
}

// blocking is the rule that blocks the answer, or nil.
func (run *responseGuardRun) blocking() *responseGuardRuleRun {
	if run == nil || run.streamed {
		return nil
	}
	for i := range run.rules {
		if run.rules[i].action == config.ResponseGuardBlock {
			return &run.rules[i]
		}
	}
	return nil
}

// warnings are the codes the annotating rules add, in rule order.
func (run *responseGuardRun) warnings() []string {
	var codes []string
	if run == nil || run.streamed {
		return nil
	}
	for _, rule := range run.rules {
		if rule.action != config.ResponseGuardAnnotate {
			continue
		}
		for _, code := range rule.codes {
			if !slices.Contains(codes, code) {
				codes = append(codes, code)
			}
		}
	}
	return codes
}

// notedCodes are the codes of the annotating rules that write a note at the
// head of the answer.
func (run *responseGuardRun) notedCodes() []string {
	var codes []string
	if run == nil || run.streamed {
		return nil
	}
	for _, rule := range run.rules {
		if rule.action != config.ResponseGuardAnnotate || !rule.details {
			continue
		}
		for _, code := range rule.codes {
			if !slices.Contains(codes, code) {
				codes = append(codes, code)
			}
		}
	}
	return codes
}

func (run *responseGuardRun) outcome() string {
	switch {
	case run.streamed:
		return "not_enforced_streaming"
	case run.blocking() != nil:
		return "blocked"
	case len(run.warnings()) > 0:
		return "annotated"
	}
	return "passed"
}

// observed is what the response-stage rules the guard read made of the
// answer, or nil when the guard read none.
func (run *responseGuardRun) observed() *classification.ResponseSignal {
	if run == nil || run.conditions == nil {
		return nil
	}
	return run.conditions.Observed()
}

// jailbreakFinding reports how the guard's rules read the answer's
// response-stage jailbreak signal: matched, or unverified when it could not
// be read and a rule acted on that.
func (run *responseGuardRun) jailbreakFinding() (matched, unverified bool) {
	observed := run.observed()
	if observed == nil {
		return false, false
	}
	if len(observed.Reported[config.SignalTypeJailbreak]) > 0 {
		return true, false
	}
	if !hasSignalError(observed, config.SignalTypeJailbreak) {
		return false, false
	}
	for _, rule := range run.rules {
		if rule.state == decision.RuleUnknown && rule.action != config.ResponseGuardNone && slices.Contains(rule.codes, headers.ResponseWarningJailbreak) {
			return false, true
		}
	}
	return false, false
}

// responseGuardBlockMessage is all a client sees of a blocked answer. It names
// what was found and never that a check failed: telling a caller the guard is
// down hands an attacker a probe for when it is.
func responseGuardBlockMessage(rule *responseGuardRuleRun) string {
	switch {
	case slices.Contains(rule.codes, headers.ResponseWarningJailbreak):
		return "Response blocked: jailbreak content detected in LLM output"
	case slices.Contains(rule.codes, headers.ResponseWarningHallucination), slices.Contains(rule.codes, headers.ResponseWarningUnverifiedFactual):
		return "Response blocked: the answer is not supported by its grounding context"
	}
	return "Response blocked by the response guard"
}

// responseGuardNote is the note an annotating rule with details writes at the
// head of the answer.
func (run *responseGuardRun) responseGuardNote(code string) string {
	switch code {
	case headers.ResponseWarningHallucination:
		return hallucinationWarningText(run.observed())
	case headers.ResponseWarningUnverifiedFactual:
		return "[Unverified Response] This response contains factual claims that could not be verified due to missing context."
	case headers.ResponseWarningJailbreak:
		return "[Safety Warning] This response may contain content that tries to get around safety rules."
	}
	return "[Response Warning] A response check flagged this response."
}

// applyResponseGuardNotes writes the notes of the annotating rules with
// details at the head of the answer, and reports whether it changed.
func (r *OpenAIRouter) applyResponseGuardNotes(run *responseGuardRun, response *llmprotocol.Response) bool {
	codes := run.notedCodes()
	if len(codes) == 0 {
		return false
	}
	notes := make([]string, len(codes))
	for i, code := range codes {
		notes[i] = run.responseGuardNote(code)
	}
	return prependSemanticAssistantText(response, strings.Join(notes, "\n\n"))
}

// logResponseGuard says what the guard did with an answer it acted on.
func logResponseGuard(ctx *RequestContext, run *responseGuardRun) {
	if run == nil || run.outcome() == "passed" {
		return
	}
	logging.ComponentEvent("extproc", "response_guard_applied", map[string]interface{}{
		"request_id": ctx.RequestID,
		"decision":   ctx.VSRSelectedDecisionName,
		"outcome":    run.outcome(),
		"warnings":   run.warnings(),
	})
}
