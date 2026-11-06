package config

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// RemovedPluginTypes says what replaced each plugin type the Router no longer
// reads; the load error and the DSL compiler both name it.
var RemovedPluginTypes = map[string]string{
	"response_jailbreak": "a response_guard rule over a jailbreak signal with direction: response",
	"hallucination":      "a response_guard rule over the fact_check and hallucination signals",
}

// pluginForms are the plugin forms that ran their own detectors or read their
// own thresholds before plugins acted on named signals: response_jailbreak
// and hallucination (a response_guard rule over the signals they checked),
// masking's threshold (the pii rule masking names), and a system_prompt mode
// other than insert or replace. response_jailbreak comes first, so a
// decision's hallucination rule joins the guard its jailbreak rule started.
var pluginForms = []ConfigForm{
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("response_jailbreak"),
		What:    "plugin response_jailbreak was removed",
		Becomes: RemovedPluginTypes["response_jailbreak"],
		Migrate: migrateResponseJailbreakPlugin,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("hallucination"),
		What:    "plugin hallucination was removed",
		Becomes: RemovedPluginTypes["hallucination"],
		Migrate: migrateHallucinationPlugin,
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[].plugins[]"),
		Match:    isThresholdMasking,
		Describe: describeThresholdMasking,
		Migrate:  migrateThresholdMasking,
	},
	{
		Code:  CodeInvalidValue,
		Paths: RoutingScopes("decisions[].plugins[]"),
		Match: hasUnknownSystemPromptMode,
		Describe: func(at FormMatch) (string, string) {
			mode := scalarAt(mappingValue(at.Node, "configuration"), "mode")
			return fmt.Sprintf("system_prompt mode %q is neither insert nor replace", mode), "mode: replace, which is what it did"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(ensureMapping(at.Node, "configuration"), "mode", Scalar(SystemPromptModeReplace))
			return nil
		},
	},
}

// migrateResponseJailbreakPlugin writes an enabled response_jailbreak plugin
// as a response_guard rule. The plugin acted on the scope's response-direction
// jailbreak rules when it had any, and otherwise scanned answers itself at its
// threshold (else prompt_guard's, else 0.5), which a new response-direction
// rule now does. Its action header annotates; an answer the scan could not
// read counted as detected unless prompt_guard allowed both failures and
// unscanned content, which is the guard's on_unknown.
func migrateResponseJailbreakPlugin(m *Migration, at FormMatch) error {
	scope, decision, err := pluginSite(at)
	if err != nil {
		return err
	}
	configuration := mappingValue(at.Node, "configuration")
	if !enabledScalar(configuration) {
		removeItem(at.Parent, at.Node)
		m.Changed(at.Path, "the disabled response_jailbreak plugin is removed")
		return nil
	}
	promptGuard := mappingAt(at.Root, "global", "model_catalog", "modules", "prompt_guard")
	rules := signalRules(scope, SignalTypeJailbreak)
	var reads []*yaml.Node
	for _, rule := range rules.Content {
		if scalarAt(rule, "direction") == SignalDirectionResponse {
			reads = append(reads, conditionNode(SignalTypeJailbreak, scalarAt(rule, "name")))
		}
	}
	threshold := positiveScalar(mappingValue(configuration, "threshold"))
	switch {
	case len(reads) > 0 && threshold != nil:
		m.Changed(at.Path, "the plugin's threshold was not read beside the response-direction jailbreak rules; their thresholds apply")
	case len(reads) == 0:
		if threshold == nil {
			threshold = positiveScalar(mappingValue(promptGuard, "threshold"))
		}
		if threshold == nil {
			threshold = &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!float", Value: "0.5"}
		}
		name := freeRuleName(rules, scalarAt(decision, "name")+"_response")
		rules.Content = append(rules.Content, mappingOf(
			"name", Scalar(name),
			"direction", Scalar(SignalDirectionResponse),
			"threshold", threshold,
			"description", Scalar("Answers that carry jailbreak content."),
		))
		reads = append(reads, conditionNode(SignalTypeJailbreak, name))
		m.Changed(scopePrefix(at.Path)+"signals.jailbreak["+name+"]", "the response-direction jailbreak rule scans answers at the plugin's threshold")
	}
	action := ResponseGuardAnnotate
	switch scalarAt(configuration, "action") {
	case ResponseGuardBlock:
		action = ResponseGuardBlock
	case ResponseGuardNone:
		action = ResponseGuardNone
	}
	onUnknown := ResponseGuardNone
	if scalarAt(promptGuard, "on_error") == OnErrorBlock || scalarAt(promptGuard, "on_unscanned") != OnErrorAllow {
		onUnknown = action
	}
	writeResponseGuard(at, mappingOf("when", anyOf(reads), "action", Scalar(action)), onUnknown)
	return nil
}

// migrateHallucinationPlugin writes an enabled hallucination plugin as a
// response_guard rule. The plugin checked an answer only when the fact_check
// rule needs_fact_check matched, against the scope's hallucination detector;
// the rule reads both. hallucination_action header and block annotate (block
// never blocked), body annotates with the note, and unverified_factual_action
// (an answer it could not check) is the guard's on_unknown.
func migrateHallucinationPlugin(m *Migration, at FormMatch) error {
	scope, _, err := pluginSite(at)
	if err != nil {
		return err
	}
	configuration := mappingValue(at.Node, "configuration")
	if !enabledScalar(configuration) {
		removeItem(at.Parent, at.Node)
		m.Changed(at.Path, "the disabled hallucination plugin is removed")
		return nil
	}
	prefix := scopePrefix(at.Path)
	factChecks := signalRules(scope, SignalTypeFactCheck)
	if ruleNamed(factChecks, "needs_fact_check") == nil {
		factChecks.Content = append(factChecks.Content, mappingOf(
			"name", Scalar("needs_fact_check"),
			"description", Scalar("Prompts whose answers are worth checking against their context."),
		))
		m.Note(prefix+"signals.fact_check[needs_fact_check]", "the hallucination plugin checked answers only when a fact_check rule named needs_fact_check matched, and the scope declared none, so it never checked one; the guard now checks the answers this rule selects")
	}
	detectors := signalRules(scope, SignalTypeHallucination)
	detector := ""
	if len(detectors.Content) > 0 {
		detector = scalarAt(detectors.Content[0], "name")
	}
	if detector == "" {
		detector = freeRuleName(detectors, "ungrounded_claims")
		detectors.Content = append(detectors.Content, mappingOf(
			"name", Scalar(detector),
			"description", Scalar("Answers with claims their grounding context does not support."),
		))
	}

	action, details := ResponseGuardAnnotate, false
	switch scalarAt(configuration, "hallucination_action") {
	case "none":
		action = ResponseGuardNone
	case "body":
		details = true
	case "block":
		m.Note(at.Path, "hallucination_action: block never blocked an answer; it annotated it, as the rule does; write action: block to block")
	}
	onUnknown := ResponseGuardAnnotate
	switch unverified := scalarAt(configuration, "unverified_factual_action"); {
	case unverified == "none":
		onUnknown = ResponseGuardNone
	case (unverified == "body") != details:
		m.Changed(at.Path, "hallucination_action and unverified_factual_action placed their warnings apart; the rule's details now decide whether either writes a note")
	}
	if mappingValue(configuration, "include_hallucination_details") != nil && !details {
		m.Changed(at.Path, "include_hallucination_details is dropped: only a rule with details writes a note, which names the unsupported spans")
	}

	rule := mappingOf(
		"when", mappingOf("operator", Scalar("AND"), "conditions", sequenceOf(
			conditionNode(SignalTypeFactCheck, "needs_fact_check"),
			conditionNode(SignalTypeHallucination, detector),
		)),
		"action", Scalar(action),
	)
	if details {
		setNode(rule, "details", &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: "true"})
	}
	writeResponseGuard(at, rule, onUnknown)
	return nil
}

// isThresholdMasking matches a masking plugin written before masking named
// its pii rule: with its own threshold, or with no signal.
func isThresholdMasking(node *yaml.Node, _ string) bool {
	if !hasType(DecisionPluginMasking)(node, "") {
		return false
	}
	configuration := mappingValue(node, "configuration")
	return mappingValue(configuration, "threshold") != nil || scalarAt(configuration, "signal") == ""
}

func describeThresholdMasking(at FormMatch) (string, string) {
	if mappingValue(mappingValue(at.Node, "configuration"), "threshold") != nil {
		return "masking.threshold was removed", "masking.signal naming a pii rule, whose threshold applies"
	}
	return "masking without a signal was removed", "masking.signal naming the pii rule whose detections it masks"
}

// migrateThresholdMasking names a pii rule for masking. The plugin masked
// every type the PII classifier detected at its threshold (else the
// classifier's), which a new pii rule with no allowed types reproduces; a
// plugin that names its rule moves its threshold onto the rule.
func migrateThresholdMasking(m *Migration, at FormMatch) error {
	scope, decision, err := pluginSite(at)
	if err != nil {
		return err
	}
	configuration := ensureMapping(at.Node, "configuration")
	threshold := positiveScalar(mappingValue(configuration, "threshold"))
	deleteKey(configuration, "threshold")
	prefix := scopePrefix(at.Path)
	rules := signalRules(scope, SignalTypePII)
	if signal := scalarAt(configuration, "signal"); signal != "" {
		rule := ruleNamed(rules, signal)
		switch {
		case threshold == nil:
		case rule != nil && positiveScalar(mappingValue(rule, "threshold")) == nil:
			setNode(rule, "threshold", threshold)
		default:
			m.Note(prefix+"signals.pii["+signal+"]", "masking.threshold is dropped: masking masks what pii rule "+signal+" detects, at the rule's threshold")
		}
		return nil
	}
	if !enabledScalar(configuration) {
		removeItem(at.Parent, at.Node)
		m.Changed(at.Path, "the disabled masking plugin is removed")
		return nil
	}
	name := freeRuleName(rules, scalarAt(decision, "name")+"_masking")
	if threshold == nil {
		threshold = positiveScalar(mappingValue(mappingAt(at.Root, "global", "model_catalog", "modules", "classifier", "pii"), "threshold"))
	}
	rule := mappingOf("name", Scalar(name))
	if threshold != nil {
		setNode(rule, "threshold", threshold)
	} else {
		m.Note(prefix+"signals.pii["+name+"]", "masking masked at the PII classifier's operating point; set this rule's threshold to keep that cut, or it masks every detection")
	}
	setNode(rule, "description", Scalar("Personal data masked before the request leaves the Router."))
	rules.Content = append(rules.Content, rule)
	setNode(configuration, "signal", Scalar(name))
	return nil
}

// hasUnknownSystemPromptMode matches a system_prompt plugin whose mode is
// neither insert nor replace; the plugin replaced the prompt for any of them.
func hasUnknownSystemPromptMode(node *yaml.Node, _ string) bool {
	if !hasType(DecisionPluginSystemPrompt)(node, "") {
		return false
	}
	switch scalarAt(mappingValue(node, "configuration"), "mode") {
	case "", SystemPromptModeInsert, SystemPromptModeReplace:
		return false
	}
	return true
}

// writeResponseGuard puts rule in the decision's response_guard when it has
// one, removing the plugin at, and otherwise rewrites the plugin at as a
// response_guard. The guard's on_unknown is the stricter of its own and
// onUnknown.
func writeResponseGuard(at FormMatch, rule *yaml.Node, onUnknown string) {
	for _, plugin := range at.Parent.Content {
		plugin = resolveAlias(plugin)
		if plugin == at.Node || scalarAt(plugin, "type") != DecisionPluginResponseGuard {
			continue
		}
		configuration := ensureMapping(plugin, "configuration")
		rules := ensureSequence(configuration, "rules")
		rules.Content = append(rules.Content, rule)
		setNode(configuration, "on_unknown", Scalar(stricterGuardAction(scalarAt(configuration, "on_unknown"), onUnknown)))
		removeItem(at.Parent, at.Node)
		return
	}
	at.Node.Content = nil
	at.Node.Style = 0
	setNode(at.Node, "type", Scalar(DecisionPluginResponseGuard))
	setNode(at.Node, "configuration", mappingOf("rules", sequenceOf(rule), "on_unknown", Scalar(onUnknown)))
}

// stricterGuardAction is the stricter of a guard's on_unknown, which
// annotates when unset, and another action.
func stricterGuardAction(current, other string) string {
	if current == "" {
		current = ResponseGuardAnnotate
	}
	rank := map[string]int{ResponseGuardNone: 0, ResponseGuardAnnotate: 1, ResponseGuardBlock: 2}
	if rank[other] > rank[current] {
		return other
	}
	return current
}

// pluginSite is the routing scope and the decision that hold the plugin at.
func pluginSite(at FormMatch) (scope, decision *yaml.Node, err error) {
	scopes := []*yaml.Node{mappingAt(at.Root, "routing")}
	for _, recipe := range sequenceAt(at.Root, "recipes") {
		scopes = append(scopes, mappingAt(resolveAlias(recipe), "routing"))
	}
	for _, scope := range scopes {
		for _, decision := range sequenceAt(scope, "decisions") {
			if resolveAlias(mappingValue(resolveAlias(decision), "plugins")) == at.Parent {
				return scope, resolveAlias(decision), nil
			}
		}
	}
	return nil, nil, errors.New("the plugin is not in a decision of a routing scope")
}

// scopePrefix is the path of the routing scope a decision path is in.
func scopePrefix(path string) string {
	if i := strings.Index(path, "routing.decisions"); i >= 0 {
		return path[:i] + "routing."
	}
	return "routing."
}

// signalRules is the scope's sequence of rules of one signal type, which it
// starts when the scope declares none; a new signals block leads the scope.
func signalRules(scope *yaml.Node, signalType string) *yaml.Node {
	if mappingValue(scope, "signals") == nil {
		scope.Content = append([]*yaml.Node{Scalar("signals"), {Kind: yaml.MappingNode, Tag: "!!map"}}, scope.Content...)
	}
	return ensureSequence(ensureMapping(scope, "signals"), signalType)
}

func ruleNamed(rules *yaml.Node, name string) *yaml.Node {
	for _, rule := range rules.Content {
		if scalarAt(resolveAlias(rule), "name") == name {
			return resolveAlias(rule)
		}
	}
	return nil
}

// freeRuleName is base, or base with the first numeric suffix no rule has.
func freeRuleName(rules *yaml.Node, base string) string {
	name := base
	for i := 2; ruleNamed(rules, name) != nil; i++ {
		name = base + "_" + strconv.Itoa(i)
	}
	return name
}

func conditionNode(signalType, name string) *yaml.Node {
	return mappingOf("type", Scalar(signalType), "name", Scalar(name))
}

// anyOf is the one condition, or an OR over several.
func anyOf(conditions []*yaml.Node) *yaml.Node {
	if len(conditions) == 1 {
		return conditions[0]
	}
	return mappingOf("operator", Scalar("OR"), "conditions", sequenceOf(conditions...))
}

// enabledScalar reads a plugin's enabled flag; unset is disabled.
func enabledScalar(configuration *yaml.Node) bool {
	enabled, err := strconv.ParseBool(scalarAt(configuration, "enabled"))
	return err == nil && enabled
}

// positiveScalar is a copy of a number above zero as written, else nil.
func positiveScalar(node *yaml.Node) *yaml.Node {
	node = resolveAlias(node)
	if node == nil || node.Kind != yaml.ScalarNode {
		return nil
	}
	if value, err := strconv.ParseFloat(node.Value, 64); err != nil || value <= 0 {
		return nil
	}
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: node.Tag, Value: node.Value}
}

// mappingOf is a mapping of alternating keys and value nodes, in order.
func mappingOf(pairs ...any) *yaml.Node {
	mapping := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for i := 0; i+1 < len(pairs); i += 2 {
		setNode(mapping, pairs[i].(string), pairs[i+1].(*yaml.Node))
	}
	return mapping
}

func sequenceOf(items ...*yaml.Node) *yaml.Node {
	return &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq", Content: items}
}

func ensureSequence(mapping *yaml.Node, key string) *yaml.Node {
	sequence := resolveAlias(mappingValue(mapping, key))
	if sequence == nil || sequence.Kind != yaml.SequenceNode {
		sequence = sequenceOf()
		setNode(mapping, key, sequence)
	}
	return sequence
}

// removeItem removes item, or an alias of it, from a sequence.
func removeItem(sequence, item *yaml.Node) {
	kept := sequence.Content[:0]
	for _, candidate := range sequence.Content {
		if resolveAlias(candidate) != item {
			kept = append(kept, candidate)
		}
	}
	sequence.Content = kept
}

func resolveAlias(node *yaml.Node) *yaml.Node {
	for node != nil && node.Kind == yaml.AliasNode {
		node = node.Alias
	}
	return node
}
