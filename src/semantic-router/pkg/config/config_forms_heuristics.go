package config

import (
	"fmt"
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// heuristicForms are the heuristics that other signals absorbed: events and
// structure rules are keywords rules, the actions are one systemone choice
// question, and a conversation rule on image_content is an input_modality
// rule. Migrate rewrites the references to them as well.
var heuristicForms = []ConfigForm{
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.events"),
		What:    "the events signal is removed",
		Becomes: "keywords rules with operator: OR over each rule's terms",
		Migrate: migrateEventRules,
	},
	removedReferenceForm("event", "type: keyword", func(m *Migration, at FormMatch) error {
		m.Set(at.Node, "type", Scalar(SignalTypeKeyword))
		return nil
	}),
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.structure"),
		What:    "the structure signal is removed",
		Becomes: "keywords rules with operator: OR and a count or density predicate",
		Migrate: migrateStructureRules,
	},
	removedReferenceForm("structure", "type: keyword", func(m *Migration, at FormMatch) error {
		m.Set(at.Node, "type", Scalar(SignalTypeKeyword))
		return nil
	}),
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.actions"),
		What:    "the actions signal is removed",
		Becomes: "the systemone choice question " + actionQuestionName + " over " + strings.Join(actionChoiceKeys(), ", "),
		Migrate: migrateActionRules,
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("decisions[].rules.**"),
		Match:   hasType("action"),
		What:    "condition type action is removed",
		Becomes: "type: systemone, name: " + actionQuestionName + " and the action as its label",
		Migrate: func(m *Migration, at FormMatch) error {
			action := scalarAt(at.Node, "name")
			m.Set(at.Node, "type", Scalar(SignalTypeSystemOne))
			m.Set(at.Node, "name", Scalar(actionQuestionName))
			m.Set(at.Node, "label", Scalar(action))
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("projections.scores[].inputs[]"),
		Match:   hasType("action"),
		What:    "projection input type action is removed",
		Becomes: "type: systemone and name: " + actionQuestionName + ":<action>",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Node, "type", Scalar(SignalTypeSystemOne))
			m.Set(at.Node, "name", Scalar(actionQuestionName+":"+scalarAt(at.Node, "name")))
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: RoutingScopes("signals.conversation[].feature.source.type"),
		Match: func(node *yaml.Node, _ string) bool {
			return node.Kind == yaml.ScalarNode && node.Value == "image_content"
		},
		What:    "the conversation feature image_content is removed",
		Becomes: "an input_modality rule with modality: image, which counts the image parts of user messages",
		Migrate: migrateImageContentRule,
	},
}

// signalReferencePatterns are where a recipe names a signal rule:
// decision conditions and projection inputs.
var signalReferencePatterns = []string{
	"decisions[].rules.**",
	"projections.scores[].inputs[]",
}

// removedReferenceForm is the form of the references to a removed signal
// type, in every recipe.
func removedReferenceForm(signalType, becomes string, migrate func(*Migration, FormMatch) error) ConfigForm {
	var paths []string
	for _, pattern := range signalReferencePatterns {
		paths = append(paths, RoutingScopes(pattern)...)
	}
	return ConfigForm{
		Code:    CodeRemovedSignalType,
		Paths:   paths,
		Match:   hasType(signalType),
		What:    "condition type " + signalType + " is removed",
		Becomes: becomes,
		Migrate: migrate,
	}
}

// scopeReferences are the references to rules of signalType in the routing
// scope at scopePath.
func scopeReferences(scope *yaml.Node, scopePath, signalType string) []FormMatch {
	var found []FormMatch
	seen := map[*yaml.Node]bool{}
	for _, pattern := range signalReferencePatterns {
		for _, at := range matchPattern(scope, strings.Split(pattern, ".")) {
			if seen[at.Node] || !hasType(signalType)(at.Node, at.Path) {
				continue
			}
			seen[at.Node] = true
			at.Path = scopePath + at.Path
			found = append(found, at)
		}
	}
	return found
}

// readsValue reports whether a reference reads its rule's value rather than
// whether it matched: a condition with its own predicate, or a projection
// input whose value_source is raw or confidence.
func readsValue(reference *yaml.Node) bool {
	source := scalarAt(reference, "value_source")
	return mappingValue(reference, "predicate") != nil || source == ProjectionValueSourceRaw || source == ProjectionValueSourceConfidence
}

// noteValueReferences notes each reference of a scope to the rules of
// signalType that reads a value which means something else once they move.
func noteValueReferences(m *Migration, scope *yaml.Node, scopePath, signalType string, changes func(name string, reference *yaml.Node) bool, message string) {
	for _, reference := range scopeReferences(scope, scopePath, signalType) {
		if readsValue(reference.Node) && changes(scalarAt(reference.Node, "name"), reference.Node) {
			m.Note(reference.Path, message)
		}
	}
}

// signalsScope is the recipe of a match at signals.<type> and its path.
func signalsScope(at FormMatch) (*yaml.Node, string) {
	return at.Above(2), strings.TrimSuffix(at.Path, "signals."+at.Key)
}

// moveRules moves the rewritten rules of a match at signals.<type> to the
// end of signals.<to>. A name both collections hold is refused: their
// references would name one rule.
func moveRules(m *Migration, at FormMatch, to string) error {
	for _, rule := range at.Node.Content {
		if name := scalarAt(rule, "name"); signalRuleNamed(at.Parent, to, name) {
			return fmt.Errorf("%s already has a rule named %q; rename one of them and its references first", to, name)
		}
	}
	return m.Rename(at.Parent, at.Key, to)
}

func signalRuleNamed(signals *yaml.Node, collection, name string) bool {
	return slices.ContainsFunc(sequenceAt(signals, collection), func(rule *yaml.Node) bool {
		return scalarAt(rule, "name") == name
	})
}

func appendSignalRule(signals *yaml.Node, collection string, rule *yaml.Node) {
	rules := mappingValue(signals, collection)
	if rules == nil || rules.Kind != yaml.SequenceNode {
		rules = &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
		setNode(signals, collection, rules)
	}
	rules.Content = append(rules.Content, rule)
}

// plainScalar is a scalar the encoder writes unquoted and the loader reads
// by its value, such as a number or a boolean.
func plainScalar(value string) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Value: value}
}

func orderedMapping(keysAndValues ...*yaml.Node) *yaml.Node {
	return &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map", Content: keysAndValues}
}

func boolAt(node *yaml.Node, key string) bool {
	var value bool
	if field := mappingValue(node, key); field != nil {
		_ = field.Decode(&value)
	}
	return value
}

func predicateAt(node *yaml.Node, key string) (*NumericPredicate, error) {
	field := mappingValue(node, key)
	if field == nil {
		return nil, nil
	}
	var predicate NumericPredicate
	if err := field.Decode(&predicate); err != nil {
		return nil, fmt.Errorf("%s: %w", key, err)
	}
	return &predicate, nil
}

func (p NumericPredicate) holds(value float64) bool {
	return (p.GT == nil || value > *p.GT) && (p.GTE == nil || value >= *p.GTE) &&
		(p.LT == nil || value < *p.LT) && (p.LTE == nil || value <= *p.LTE)
}

// matchesAnyPresence reports whether, on a count, the predicate means "at
// least one".
func (p NumericPredicate) matchesAnyPresence() bool {
	return p.LT == nil && p.LTE == nil && !p.holds(0) && p.holds(1)
}

// The event classifier's terms: the severities it knew, and its urgency
// pattern written out, whose time.sensitive and critical.window took any
// character between the words.
var (
	eventSeverities      = []string{"critical", "high", "medium", "low"}
	eventTemporalMarkers = []string{"urgent", "immediate", "asap", "deadline", "time sensitive", "time-sensitive", "now", "critical window", "critical-window"}
)

// migrateEventRules writes each event rule as a keywords rule with operator
// OR over its terms: the event classifier matched a rule when any of them
// occurred as a word, ignoring case, as a literal keyword does.
func migrateEventRules(m *Migration, at FormMatch) error {
	scope, scopePath := signalsScope(at)
	for _, rule := range at.Node.Content {
		temporal := boolAt(rule, "temporal")
		keywords := eventTerms(rule, temporal)
		for _, key := range []string{"event_types", "severities", "action_codes", "temporal"} {
			m.Delete(rule, key)
		}
		m.Set(rule, "operator", Scalar("OR"))
		m.Set(rule, "keywords", stringSequence(keywords...))
		if temporal {
			m.Note(scopePath+"signals.keywords["+scalarAt(rule, "name")+"]",
				"the temporal markers time.sensitive and critical.window matched any character between their words; the rule lists their space and hyphen forms")
		}
	}
	noteValueReferences(m, scope, scopePath, "event", func(string, *yaml.Node) bool { return true },
		"it read the event rule's confidence; a keywords rule publishes a value only with count or density")
	return moveRules(m, at, "keywords")
}

// eventTerms are an event rule's terms, each once, in the order the event
// classifier tried them. Severities it did not know never matched.
func eventTerms(rule *yaml.Node, temporal bool) []string {
	var terms []string
	add := func(term string) {
		term = strings.TrimSpace(term)
		if term != "" && !slices.ContainsFunc(terms, func(existing string) bool { return strings.EqualFold(existing, term) }) {
			terms = append(terms, term)
		}
	}
	for _, term := range sequenceAt(rule, "event_types") {
		add(term.Value)
	}
	for _, severity := range sequenceAt(rule, "severities") {
		if known := strings.ToLower(strings.TrimSpace(severity.Value)); slices.Contains(eventSeverities, known) {
			add(known)
		}
	}
	for _, code := range sequenceAt(rule, "action_codes") {
		add(code.Value)
	}
	if temporal {
		for _, marker := range eventTemporalMarkers {
			add(marker)
		}
	}
	return terms
}

// structureDensityScale turns structure density, matches per text unit (a
// word, or a CJK character), into matches per 1,000 characters, at five
// characters a unit.
const structureDensityScale = 200

// utf8BytePatterns count a text's UTF-8 bytes as keyword occurrences: every
// character matches the first, and each later pattern matches the characters
// of more than one, two and three bytes.
var utf8BytePatterns = []string{`(?s).`, `[^\x00-\x7F]`, `[^\x00-\x{7FF}]`, `[^\x00-\x{FFFF}]`}

// promptCompressionEnabled reports whether the text the signals read is
// compressed before they run.
func promptCompressionEnabled(root *yaml.Node) bool {
	if enabledScalar(mappingAt(root, "global", "model_catalog", "modules", "prompt_compression")) {
		return true
	}
	for _, plugin := range sequenceAt(root, "global", "plugins") {
		if scalarAt(plugin, "type") == "context" && enabledScalar(mappingAt(plugin, "configuration", "compress", "signals")) {
			return true
		}
	}
	return false
}

// migrateStructureRules writes each structure rule as a keywords rule with
// operator OR: a regex source is a regex keyword, a keyword_set its literal
// keywords, and each ordered sequence a regex over its markers. A count
// predicate stays as written; a density one is scaled to 1,000 characters.
func migrateStructureRules(m *Migration, at FormMatch) error {
	scope, scopePath := signalsScope(at)
	counts := map[string]bool{}
	for _, rule := range at.Node.Content {
		feature := strings.ToLower(strings.TrimSpace(scalarAt(mappingAt(rule, "feature"), "type")))
		source := strings.ToLower(strings.TrimSpace(scalarAt(mappingAt(rule, "feature", "source"), "type")))
		counts[scalarAt(rule, "name")] = feature == "count" && source != "sequence"
		if err := rewriteStructureRule(m, scopePath, rule, feature, source); err != nil {
			return fmt.Errorf("rule %q: %w", scalarAt(rule, "name"), err)
		}
	}
	noteValueReferences(m, scope, scopePath, "structure", func(name string, reference *yaml.Node) bool {
		return !counts[name] || scalarAt(reference, "value_source") == ProjectionValueSourceConfidence
	}, "it read the structure rule's value or confidence; the keywords rule publishes its count or density, per 1,000 characters, and no confidence")
	return moveRules(m, at, "keywords")
}

func rewriteStructureRule(m *Migration, scopePath string, rule *yaml.Node, feature, sourceType string) error {
	source := mappingAt(rule, "feature", "source")
	written := mappingValue(rule, "predicate")
	predicate, err := predicateAt(rule, "predicate")
	if err != nil {
		return err
	}
	method, keywords := "", []string(nil)
	switch sourceType {
	case "regex":
		method, keywords = "regex", []string{scalarAt(source, "pattern")}
	case "keyword_set":
		for _, keyword := range sequenceAt(source, "keywords") {
			keywords = append(keywords, keyword.Value)
		}
	case "sequence":
		method = "regex"
		for _, sequence := range sequenceAt(source, "sequences") {
			markers := make([]string, 0, len(sequence.Content))
			for _, marker := range sequence.Content {
				markers = append(markers, regexp.QuoteMeta(marker.Value))
			}
			keywords = append(keywords, "(?s)"+strings.Join(markers, ".*"))
		}
	case "text_bytes":
		method, keywords = "regex", utf8BytePatterns
		m.Changed(scopePath+"signals.keywords["+scalarAt(rule, "name")+"]",
			"text_bytes is the count of these patterns: each character matches once, and once more for every UTF-8 byte past its first")
		if promptCompressionEnabled(m.Root()) {
			m.Changed(scopePath+"signals.keywords["+scalarAt(rule, "name")+"]",
				"it counts the bytes of the text the keywords rules read, which compress.signals compresses; text_bytes counted the original text: "+
					"add keyword to compress.signals.skip_signals to count that")
		}
	default:
		return fmt.Errorf("feature.source.type %q has no keywords form", sourceType)
	}
	// The byte patterns must not fold case: a character that folds to ASCII
	// would no longer match the non-ASCII ones.
	caseSensitive := boolAt(source, "case_sensitive") || sourceType == "text_bytes"
	m.Delete(rule, "feature")
	m.Delete(rule, "predicate")
	m.Set(rule, "operator", Scalar("OR"))
	if method != "" {
		m.Set(rule, "method", Scalar(method))
	}
	m.Set(rule, "keywords", stringSequence(keywords...))
	if caseSensitive {
		m.Set(rule, "case_sensitive", plainScalar("true"))
	}
	if predicate == nil {
		return nil
	}
	switch {
	case feature == "density":
		m.Set(rule, "density", predicateNode(predicate, structureDensityScale))
		m.Note(scopePath+"signals.keywords["+scalarAt(rule, "name")+"].density",
			"structure density counted matches per text unit, a word or a CJK character; this predicate is scaled to matches per 1,000 characters at five characters a unit, so re-fit it on real prompts")
	case feature == "count" && sourceType != "sequence":
		m.Set(rule, "count", written)
	default:
		// The feature was 0 or 1; the rule counts the patterns that match.
		switch none, some := predicate.holds(0), predicate.holds(1); {
		case none && some:
			m.Set(rule, "count", orderedMapping(Scalar("gte"), plainScalar("0")))
		case none:
			m.Set(rule, "count", orderedMapping(Scalar("lt"), plainScalar("1")))
		case !some:
			m.Set(rule, "count", orderedMapping(Scalar("lt"), plainScalar("0")))
		}
	}
	return nil
}

// predicateNode writes a predicate with each bound multiplied by scale.
func predicateNode(predicate *NumericPredicate, scale float64) *yaml.Node {
	node := orderedMapping()
	for _, bound := range []struct {
		key   string
		value *float64
	}{{"gt", predicate.GT}, {"gte", predicate.GTE}, {"lt", predicate.LT}, {"lte", predicate.LTE}} {
		if bound.value != nil {
			scaled := math.Round(*bound.value*scale*1e6) / 1e6
			node.Content = append(node.Content, Scalar(bound.key), plainScalar(strconv.FormatFloat(scaled, 'f', -1, 64)))
		}
	}
	return node
}

// The systemone question the actions become: the action vocabulary the
// English phrase classifier chose from, described by the phrases it read.
const (
	actionQuestionName         = "action"
	actionQuestionDescription  = "The action the request asks for."
	actionQuestionInstructions = "Which action does this request ask for?"
)

var actionChoices = []SystemOneOption{
	{Key: "generate", Description: "Write new code or another new artifact, such as a function, script, class, component or project scaffold."},
	{Key: "explain", Description: "Explain, describe or summarize something, or answer a question about it, without changing it."},
	{Key: "fix", Description: "Fix or debug something that is broken, failing or raising an error."},
	{Key: "refactor", Description: "Restructure, rename, simplify or clean up existing code without changing what it does."},
	{Key: "test", Description: "Write or improve tests, test cases or test coverage."},
	{Key: "other", Description: "Anything else, or no action is named."},
}

func actionChoiceKeys() []string {
	keys := make([]string, len(actionChoices))
	for index, choice := range actionChoices {
		keys[index] = choice.Key
	}
	return keys
}

// migrateActionRules writes the actions as the systemone choice question
// action, whose options describe each action by its rule's description or
// the phrases the classifier read; conditions then name the action as a
// label.
func migrateActionRules(m *Migration, at FormMatch) error {
	scope, scopePath := signalsScope(at)
	if signalRuleNamed(at.Parent, SignalTypeSystemOne, actionQuestionName) {
		return fmt.Errorf("systemone already has a rule named %q; rename it and its references first", actionQuestionName)
	}
	described := map[string]string{}
	for _, rule := range at.Node.Content {
		described[scalarAt(rule, "name")] = scalarAt(rule, "description")
	}
	choices := &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
	for _, choice := range actionChoices {
		description := choice.Description
		if declared := described[choice.Key]; declared != "" {
			description = declared
		}
		choices.Content = append(choices.Content, orderedMapping(
			Scalar("key"), Scalar(choice.Key), Scalar("description"), Scalar(description)))
	}
	appendSignalRule(at.Parent, SignalTypeSystemOne, orderedMapping(
		Scalar("name"), Scalar(actionQuestionName),
		Scalar("description"), Scalar(actionQuestionDescription),
		Scalar("question"), orderedMapping(
			Scalar("type"), Scalar(SystemOneQuestionChoice),
			Scalar("instructions"), Scalar(actionQuestionInstructions),
			Scalar("choices"), choices),
	))
	m.Delete(at.Parent, at.Key)
	noteValueReferences(m, scope, scopePath, "action", func(string, *yaml.Node) bool { return true },
		"it read the share of the request's action phrases; it now reads the probability the decision model gives the action")
	return nil
}

// migrateImageContentRule writes a conversation rule on image_content as an
// input_modality rule on images, and its references follow.
func migrateImageContentRule(m *Migration, at FormMatch) error {
	rule, conversation, signals, scope := at.Above(3), at.Above(4), at.Above(5), at.Above(6)
	scopePath := at.Path[:strings.LastIndex(at.Path, "signals.conversation[")]
	name := scalarAt(rule, "name")
	if signalRuleNamed(signals, SignalTypeInputModality, name) {
		return fmt.Errorf("input_modality already has a rule named %q; rename one of them and its references first", name)
	}
	counted := strings.EqualFold(scalarAt(mappingAt(rule, "feature"), "type"), "count")
	predicate, err := predicateAt(rule, "predicate")
	if err != nil {
		return err
	}
	modalityPath := scopePath + "signals.input_modality[" + name + "]"
	if counted && (predicate == nil || !predicate.matchesAnyPresence()) {
		m.Note(modalityPath, "it matched on its count of images, every request without a predicate; an input_modality rule matches a request with an image and publishes the count, so put the predicate on its conditions")
	}
	for _, reference := range scopeReferences(scope, scopePath, SignalTypeConversation) {
		if scalarAt(reference.Node, "name") != name {
			continue
		}
		m.Set(reference.Node, "type", Scalar(SignalTypeInputModality))
		if !counted && readsValue(reference.Node) && scalarAt(reference.Node, "value_source") != ProjectionValueSourceConfidence {
			m.Note(reference.Path, "it read 1 when the request had an image; an input_modality rule publishes the number of images")
		}
	}
	m.Delete(rule, "feature")
	m.Delete(rule, "predicate")
	m.Set(rule, "modality", Scalar(InputModalityImage))
	conversation.Content = slices.DeleteFunc(conversation.Content, func(item *yaml.Node) bool { return item == rule })
	if len(conversation.Content) == 0 {
		m.Delete(signals, "conversation")
	}
	appendSignalRule(signals, SignalTypeInputModality, rule)
	return nil
}
