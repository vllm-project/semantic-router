package config

import (
	"fmt"
	"math"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// applicationForms are the forms of the learned signals from before every one
// of them asked a preset of the decision model: the signal types the Router no
// longer has, the fields of the application signals it no longer reads, and
// the modules, system lines and bindings only those signals used. A type's
// conditions are rewritten before its rules, and the modules last, since a
// module's threshold becomes the predicate of the rules that inherited it.
var applicationForms = slices.Concat(removedSignalTypeForms, applicationFieldForms, applicationModuleForms)

// conditionPaths are the places a recipe names a signal: decision
// conditions at any depth and projection score inputs.
var conditionPaths = slices.Concat(
	RoutingScopes("decisions[].rules.**"),
	RoutingScopes("projections.scores[].inputs[]"),
)

const preferenceQuestionName = "preference"

var removedSignalTypeForms = []ConfigForm{
	{
		Code:    CodeRemovedSignalType,
		Paths:   conditionPaths,
		Match:   hasType("reask"),
		What:    "condition type reask is turn_relation",
		Becomes: "type: turn_relation",
		Migrate: setConditionType(SignalTypeTurnRelation),
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.reasks"),
		What:    "the reask signal is a turn_relation rule",
		Becomes: "signals.turn_relation with labels [reask]; input.previous_turns is twice lookback_turns",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, rule := range sequenceItems(at.Node) {
				next := copiedRule(rule, "name", "description")
				setNode(next, "labels", stringSequence("reask"))
				// lookback_turns counted user turns; previous_turns counts each
				// of them and the answer that followed it.
				if turns, ok := intAt(rule, "lookback_turns"); ok {
					setNode(next, "input", newMapping("previous_turns", intNode(2*turns)))
				}
				if scalarAt(rule, "threshold") != "" {
					m.Note(at.Path, "a reask rule's threshold has no counterpart: the decision model tells a re-ask from the turns themselves")
				}
				appendRule(at.Parent, SignalTypeTurnRelation, next)
			}
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.topic_continuity"),
		What:    "the topic_continuity signal is a turn_relation rule",
		Becomes: "signals.turn_relation with labels [continue, shift]; limits.max_prior_turns becomes input.previous_turns",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, rule := range sequenceItems(at.Node) {
				next := copiedRule(rule, "name", "description")
				setNode(next, "labels", stringSequence("continue", "shift"))
				if turns := mappingValue(mappingValue(rule, "limits"), "max_prior_turns"); turns != nil {
					setNode(next, "input", newMapping("previous_turns", turns))
				}
				appendRule(at.Parent, SignalTypeTurnRelation, next)
			}
			m.Note(at.Path, "topic_continuity thresholds, limits and include_assistant have no counterpart: the decision model reads the previous turns")
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   conditionPaths,
		Match:   hasType("classifier"),
		What:    "condition type classifier is systemone",
		Becomes: "type: systemone",
		Migrate: setConditionType(SignalTypeSystemOne),
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.classifiers"),
		What:    "the classifier signal is a systemone question",
		Becomes: "signals.systemone with a choice question over the classifier's labels",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, rule := range sequenceItems(at.Node) {
				next := copiedRule(rule, "name", "description")
				instructions := scalarAt(rule, "instructions")
				if instructions == "" {
					instructions = "Which of these labels applies to the request?"
				}
				var choices []*yaml.Node
				for _, label := range stringItems(mappingValue(rule, "labels")) {
					choices = append(choices, newMapping("key", Scalar(label)))
				}
				setNode(next, "question", newMapping("type", Scalar(SystemOneQuestionChoice), "instructions", Scalar(instructions), "choices", sequence(choices...)))
				appendRule(at.Parent, SignalTypeSystemOne, next)
				if hasAnyKey("deployment", "model", "model_path", "type")(rule, "") {
					m.Note(at.Path, "classifier "+scalarAt(rule, "name")+" asked a classifier model; the systemone rule asks the decision model, or a deployment that answers decision questions")
				}
			}
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:  CodeRemovedSignalType,
		Paths: conditionPaths,
		Match: hasType("preference"),
		What:  "condition type preference reads a label of one systemone question",
		Describe: func(at FormMatch) (string, string) {
			return "condition type preference reads a label of one systemone question", "type: systemone, name: " + preferenceQuestionName + ", label: <preference>"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			preference, question := scalarAt(at.Node, "name"), preferenceQuestion(scopeSignals(m.Root(), at.Path))
			m.Set(at.Node, "type", Scalar(SignalTypeSystemOne))
			if strings.Contains(at.Path, ".projections.scores[") {
				// A projection input reads one option as "<question>:<option>".
				m.Set(at.Node, "name", Scalar(question+":"+preference))
				return nil
			}
			m.Set(at.Node, "name", Scalar(question))
			m.Set(at.Node, "label", Scalar(preference))
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.preferences"),
		What:    "the preference signal is one systemone choice question",
		Becomes: "signals.systemone[" + preferenceQuestionName + "] choosing among the preferences",
		Migrate: func(m *Migration, at FormMatch) error {
			var choices []*yaml.Node
			for _, rule := range sequenceItems(at.Node) {
				choice := newMapping("key", Scalar(scalarAt(rule, "name")))
				description := strings.TrimSuffix(strings.TrimSpace(scalarAt(rule, "description")), ".")
				if examples := stringItems(mappingValue(rule, "examples")); len(examples) > 0 {
					if description != "" {
						description += ". "
					}
					description += "Examples: " + strings.Join(examples, "; ")
				}
				if description != "" {
					setNode(choice, "description", Scalar(description))
				}
				choices = append(choices, choice)
			}
			// A request that expressed no preference matched no preference rule.
			none := "none"
			for _, rule := range sequenceItems(at.Node) {
				if scalarAt(rule, "name") == none {
					none = "no_preference"
				}
			}
			choices = append(choices, newMapping("key", Scalar(none), "description", Scalar("The request expresses none of these preferences.")))
			question := newMapping("type", Scalar(SystemOneQuestionChoice), "instructions", Scalar("Which response preference does the request express?"), "choices", sequence(choices...))
			appendRule(at.Parent, SignalTypeSystemOne, newMapping("name", Scalar(preferenceQuestion(at.Parent)), "question", question))
			m.Note(at.Path, "preference thresholds have no counterpart: the systemone question chooses one preference")
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   []string{"recipes[]"},
		Match:   func(node *yaml.Node, _ string) bool { return len(embeddingPartitions(node)) > 0 },
		What:    "projection partitions over embedding rules are removed with the embedding signal",
		Becomes: "drop them; a systemone choice question chooses one of a partition's members",
		Migrate: func(m *Migration, at FormMatch) error {
			partitions := mappingValue(mappingValue(at.Node, "projections"), "partitions")
			for _, partition := range embeddingPartitions(at.Node) {
				partitions.Content = slices.DeleteFunc(partitions.Content, func(item *yaml.Node) bool { return item == partition })
				m.Note(at.Path+".projections.partitions["+scalarAt(partition, "name")+"]", "a systemone choice question over the partition's members chooses one of them")
			}
			if len(partitions.Content) == 0 {
				m.Delete(mappingValue(at.Node, "projections"), "partitions")
				deleteEmpty(at.Node, "projections")
			}
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   conditionPaths,
		Match:   hasType("embedding"),
		What:    "condition type embedding is systemone",
		Becomes: "type: systemone",
		Migrate: setConditionType(SignalTypeSystemOne),
	},
	{
		Code:  CodeRemovedSignalType,
		Paths: RoutingScopes("signals.embeddings[]"),
		Match: func(node *yaml.Node, _ string) bool { return !textEmbeddingRule(node) },
		Describe: func(at FormMatch) (string, string) {
			if strings.EqualFold(scalarAt(at.Node, "query_modality"), "audio") {
				return "the embedding signal is removed, and no decision model reads audio, so no question replaces a rule with an audio query", "drop the rule: audio routing returns with a decision model that reads audio input"
			}
			return "the embedding signal is removed, and a rule with image anchors has no question that replaces it", "write a systemone question for it: input.image reads the request's images"
		},
	},
	{
		Code:  CodeRemovedSignalType,
		Paths: RoutingScopes("signals.embeddings"),
		Match: func(node *yaml.Node, _ string) bool {
			return !slices.ContainsFunc(sequenceItems(node), func(rule *yaml.Node) bool { return !textEmbeddingRule(rule) })
		},
		What:    "the embedding signal is a systemone question",
		Becomes: "signals.systemone with a yes/no question that quotes the rule's examples",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, rule := range sequenceItems(at.Node) {
				question := newMapping("type", Scalar(SystemOneQuestionNoul), "instructions", Scalar(embeddingQuestion(rule)))
				next := newMapping("name", Scalar(scalarAt(rule, "name")), "question", question)
				if strings.EqualFold(scalarAt(rule, "query_modality"), "image") {
					setNode(next, "input", newMapping("image", boolScalar()))
				}
				appendRule(at.Parent, SignalTypeSystemOne, next)
			}
			m.Note(at.Path, "an embedding rule's threshold was a similarity; re-check each systemone rule's predicate against real traffic")
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:    CodeRemovedSignalType,
		Paths:   RoutingScopes("signals.kb"),
		What:    "the kb signal is removed",
		Becomes: "nothing: its rules never match; ask a systemone set question over the knowledge base's labels instead",
		Migrate: migrateKBSignal,
	},
	{
		Code:  CodeRemovedSignalType,
		Paths: conditionPaths,
		Match: func(node *yaml.Node, path string) bool {
			return hasType("kb")(node, path) || hasType("kb_metric")(node, path)
		},
		What:    "kb conditions and kb_metric projection inputs are removed",
		Becomes: "nothing: a kb condition never matches, and a score drops a kb_metric input; read a systemone question's labels instead",
		Migrate: func(m *Migration, at FormMatch) error {
			if strings.Contains(at.Path, ".projections.") {
				dropScoreInput(m, at, kbInputName(at.Node), kbNeverMatches)
				return nil
			}
			settleRemovedCondition(m, at, "kb rule", func(node *yaml.Node) bool { return hasType("kb")(node, "") }, kbNeverMatches)
			return nil
		},
	},
	{
		Code:    CodeRemovedSection,
		Paths:   []string{"global.model_catalog.kbs"},
		What:    "global.model_catalog.kbs is removed with the kb signal",
		Becomes: "drop it; a systemone question names its labels",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog")
			return nil
		},
	},
}

// kbNeverMatches is why a kb condition never matches.
var kbNeverMatches = neverMatches{
	why:     "no v0.5 signal reads a knowledge base",
	restore: "to keep it, condition it on a systemone question over the knowledge base's labels",
}

// migrateKBSignal drops a recipe's kb rules, which never match, each with the
// systemone set question over its knowledge base's labels that replaces it.
func migrateKBSignal(m *Migration, at FormMatch) error {
	groups := kbGroups(at.Root)
	for _, rule := range sequenceItems(resolveAlias(at.Node)) {
		name, target := scalarAt(rule, "name"), mappingAt(rule, "target")
		labels := groups[scalarAt(rule, "kb")][scalarAt(target, "value")]
		if !strings.EqualFold(scalarAt(target, "kind"), "group") || len(labels) == 0 {
			labels = []string{scalarAt(target, "value")}
		}
		keys := make([]string, len(labels))
		for i, label := range labels {
			keys[i] = "{key: " + label + "}"
		}
		m.Note(at.Path+"["+name+"]", fmt.Sprintf("kb rule %s is dropped: %s, so it never matches; to route on its labels, ask signals.systemone: "+
			"[{name: %s, question: {type: set, instructions: \"Which of these does the request concern?\", labels: [%s]}}]",
			name, kbNeverMatches.why, name, strings.Join(keys, ", ")))
	}
	m.Delete(at.Parent, at.Key)
	deleteEmpty(at.Ancestors[len(at.Ancestors)-1], "signals")
	return nil
}

// kbGroups are the labels of each group of each knowledge base
// global.model_catalog.kbs declares.
func kbGroups(root *yaml.Node) map[string]map[string][]string {
	groups := map[string]map[string][]string{}
	for _, kb := range sequenceAt(root, "global", "model_catalog", "kbs") {
		byGroup := map[string][]string{}
		mapping := mappingAt(kb, "groups")
		for i := 0; mapping != nil && i+1 < len(mapping.Content); i += 2 {
			byGroup[mapping.Content[i].Value] = stringItems(resolveAlias(mapping.Content[i+1]))
		}
		groups[scalarAt(kb, "name")] = byGroup
	}
	return groups
}

// kbInputName names a kb or kb_metric projection input for a note.
func kbInputName(input *yaml.Node) string {
	if metric := scalarAt(input, "metric"); metric != "" {
		return "kb_metric " + metric
	}
	return "kb " + scalarAt(input, "name")
}

var applicationFieldForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.domains[].mmlu_categories"),
		What:    "a domain rule's mmlu_categories are its labels",
		Becomes: "labels",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, category := range sequenceItems(resolveAlias(at.Node)) {
				if domain := strings.ReplaceAll(category.Value, "_", " "); !IsSupportedRoutingDomainName(category.Value) && IsSupportedRoutingDomainName(domain) {
					m.Changed(at.Path, "mmlu category "+category.Value+" is the domain "+domain)
					category.Value = domain
				}
			}
			return m.Rename(at.Parent, at.Key, "labels")
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: []string{"recipes[]"},
		Match: func(node *yaml.Node, _ string) bool {
			return len(sequenceAt(node, "signals", "domains")) == 0 && len(domainConditionNames(node)) > 0
		},
		What:    "domain conditions no longer declare the domains they name",
		Becomes: "signals.domains with one rule per domain the conditions name",
		Migrate: func(m *Migration, at FormMatch) error {
			var rules []*yaml.Node
			for _, name := range domainConditionNames(at.Node) {
				rule := newMapping("name", Scalar(name))
				if !IsSupportedRoutingDomainName(name) {
					setNode(rule, "labels", stringSequence("other"))
				}
				rules = append(rules, rule)
			}
			setNode(ensureMapping(at.Node, "signals"), "domains", sequence(rules...))
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: RoutingScopes("signals.fact_check[]"),
		Match: func(node *yaml.Node, _ string) bool {
			return scalarAt(node, "name") == "no_fact_check_needed" && mappingValue(node, "labels") == nil
		},
		What:    "a fact_check rule's name no longer chooses its option, and no_fact_check_needed would match a request that needs a fact check",
		Becomes: "labels: [NO_FACT_CHECK_NEEDED] and a predicate above one minus the fact-check threshold",
		Migrate: func(m *Migration, at FormMatch) error {
			threshold := 0.7
			if value, err := strconv.ParseFloat(scalarAt(mappingAt(m.Root(), "global", "model_catalog", "modules", "hallucination_mitigation", "fact_check"), "threshold"), 64); err == nil && value > 0 {
				threshold = value
			}
			setNode(at.Node, "labels", stringSequence("NO_FACT_CHECK_NEEDED"))
			setNode(at.Node, "predicate", newMapping("gt", floatScalar(1-threshold)))
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: RoutingScopes("signals.user_feedbacks[]"),
		Match: func(node *yaml.Node, _ string) bool {
			_, legacy := legacyFeedbackLabels[scalarAt(node, "name")]
			return legacy && mappingValue(node, "labels") == nil
		},
		Describe: func(at FormMatch) (string, string) {
			return "a user_feedbacks rule's name no longer chooses its feedback", "labels: [" + legacyFeedbackLabels[scalarAt(at.Node, "name")] + "]"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			setNode(at.Node, "labels", stringSequence(legacyFeedbackLabels[scalarAt(at.Node, "name")]))
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: RoutingScopes("signals.modality[]"),
		Match: func(node *yaml.Node, _ string) bool {
			_, legacy := legacyModalityLabels[scalarAt(node, "name")]
			return legacy && mappingValue(node, "labels") == nil
		},
		Describe: func(at FormMatch) (string, string) {
			return "a modality rule's name no longer chooses its output (AR, DIFFUSION and BOTH are text, image and text_image)", "labels: [" + legacyModalityLabels[scalarAt(at.Node, "name")] + "]"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			setNode(at.Node, "labels", stringSequence(legacyModalityLabels[scalarAt(at.Node, "name")]))
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.complexity[]"),
		Match:   hasAnyKey(complexityRemovedFields...),
		What:    "a complexity rule no longer scores example banks against thresholds",
		Becomes: "drop hard, easy, threshold, the boundaries, composer and prototype_scoring; the complexity preset answers easy, medium or hard",
		Migrate: func(m *Migration, at FormMatch) error {
			for _, key := range complexityRemovedFields {
				m.Delete(at.Node, key)
			}
			m.Note(at.Path, "a complexity rule asks the decision model; name the level with labels or a condition's label")
			return nil
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: RoutingScopes("decisions[].rules.**"),
		Match: func(node *yaml.Node, path string) bool {
			return hasType(SignalTypeComplexity)(node, path) && strings.Contains(scalarAt(node, "name"), ":")
		},
		What:    "a complexity condition names its level as a label",
		Becomes: "name: <rule>, label: <level>",
		Migrate: func(m *Migration, at FormMatch) error {
			rule, level, _ := strings.Cut(scalarAt(at.Node, "name"), ":")
			m.Set(at.Node, "name", Scalar(rule))
			m.Set(at.Node, "label", Scalar(level))
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.jailbreak[]"),
		Match:   hasAnyKey("threshold", "direction", "include_history", "method", "jailbreak_patterns", "benign_patterns"),
		What:    "a jailbreak rule's threshold, direction and include_history are its predicate, stage and input.history, and method and patterns are removed",
		Becomes: "predicate: {gte: <threshold>}, stage, input.history",
		Migrate: func(m *Migration, at FormMatch) error {
			moveThreshold(m, at.Node, "gte")
			if scalarAt(at.Node, "direction") == SystemOneStageResponse {
				setNode(at.Node, "stage", Scalar(SystemOneStageResponse))
			}
			m.Delete(at.Node, "direction")
			moveInput(m, at.Node, "include_history", "history")
			if scalarAt(at.Node, "method") == "contrastive" || mappingValue(at.Node, "jailbreak_patterns") != nil {
				m.Note(at.Path, "contrastive jailbreak patterns are removed: the jailbreak preset reads the whole input")
			}
			for _, key := range []string{"method", "jailbreak_patterns", "benign_patterns"} {
				m.Delete(at.Node, key)
			}
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.safety[].hazard"),
		What:    "a safety rule's hazard part is removed",
		Becomes: "labels: the harm categories of its categories, predicate: {gte: <hazard.threshold>}",
		Migrate: migrateSafetyHazard,
	},
	{
		Code:  CodeRemovedField,
		Paths: RoutingScopes("signals.safety[]"),
		Match: func(node *yaml.Node, _ string) bool {
			return hasAnyKey("threshold", "unsafe_labels", "model")(node, "") && mappingValue(node, "hazard") == nil
		},
		What:    "a safety rule's threshold is its predicate, and its model labels, unsafe_labels and model are removed",
		Becomes: "predicate: {gte: <threshold>}; labels name harm categories",
		Migrate: func(m *Migration, at FormMatch) error {
			moveThreshold(m, at.Node, "gte")
			if mappingValue(at.Node, "unsafe_labels") != nil {
				m.Delete(at.Node, "labels")
			}
			m.Delete(at.Node, "unsafe_labels")
			if scalarAt(at.Node, "model") != "" {
				m.Note(at.Path, "a safety rule's model is removed: the rule asks the decision model unless a deployment is set")
			}
			m.Delete(at.Node, "model")
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("signals.pii[]"),
		Match:   hasAnyKey("threshold", "pii_types_allowed", "include_history", "source"),
		What:    "a pii rule's threshold, pii_types_allowed, include_history and source are its predicate, ignore_labels and input",
		Becomes: "predicate: {gte: <threshold>}, ignore_labels, input.history, input.tool_results",
		Migrate: func(m *Migration, at FormMatch) error {
			moveThreshold(m, at.Node, "gte")
			if allowed := mappingValue(at.Node, "pii_types_allowed"); allowed != nil {
				var ignored, dropped []string
				for _, label := range stringItems(allowed) {
					if slices.Contains(piiLabels(), label) {
						ignored = append(ignored, label)
					} else {
						dropped = append(dropped, label)
					}
				}
				m.Delete(at.Node, "pii_types_allowed")
				if len(ignored) > 0 {
					setNode(at.Node, "ignore_labels", stringSequence(ignored...))
				}
				if len(dropped) > 0 {
					m.Note(at.Path, "pii_types_allowed named types the pii preset does not locate: "+strings.Join(dropped, ", "))
				}
			}
			moveInput(m, at.Node, "include_history", "history")
			if scalarAt(at.Node, "source") == "tool_result" {
				setNode(ensureMapping(at.Node, "input"), "tool_results", boolScalar())
			}
			m.Delete(at.Node, "source")
			return nil
		},
	},
}

var applicationModuleForms = []ConfigForm{
	removedModuleForm("safety", "", "", nil),
	removedModuleForm("complexity", "", "", nil),
	removedModuleForm("feedback_detector", "threshold", "a user_feedbacks rule's predicate", []string{"user_feedbacks"}),
	removedModuleForm("modality_detector", "confidence_threshold", "a modality rule's predicate", []string{"modality"}),
	removedModuleForm("classifier.domain", "threshold", "a domain rule's predicate", []string{"domains"}),
	removedModuleForm("classifier.mcp", "", "", nil),
	removedModuleForm("classifier.preference", "", "", nil),
	removedModuleForm("hallucination_mitigation.fact_check", "threshold", "a fact_check rule's predicate", []string{"fact_check"}),
	removedModuleForm("prompt_guard", "threshold", "a jailbreak rule's predicate", []string{"jailbreak"}),
	removedModuleForm("classifier.pii", "threshold", "a pii rule's predicate", []string{"pii"}),
	removedModuleForm("hallucination_mitigation.detector", "threshold", "a hallucination rule's predicate", []string{"hallucination"}),
	removedModuleForm("hallucination_mitigation", "", "", nil),
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.services.api.batch_classification"},
		Match:   hasAnyKey("max_concurrency", "concurrency_threshold"),
		What:    "batch_classification.max_concurrency and concurrency_threshold gated the removed classifier modules",
		Becomes: "drop them; global.model_catalog.admission gates a deployment",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Node, "max_concurrency")
			m.Delete(at.Node, "concurrency_threshold")
			return nil
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: []string{"global.model_catalog.admission.*"},
		Match: func(_ *yaml.Node, path string) bool {
			return slices.Contains(removedAdmissionKeys, strings.TrimPrefix(path, "global.model_catalog.admission."))
		},
		What:    "the admission gate of a removed module",
		Becomes: "drop it; the admission of the deployment that answers the signal gates it",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog", "admission")
			return nil
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: []string{"global.model_catalog.system.*"},
		Match: func(_ *yaml.Node, path string) bool {
			return slices.Contains(removedSystemLines, strings.TrimPrefix(path, "global.model_catalog.system."))
		},
		What:    "the system line of a removed module",
		Becomes: "drop it; the decision model answers the signal",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog", "system")
			return nil
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: slices.Concat(RoutingScopes("model_bindings.*"), []string{"global.model_catalog.bindings.*"}),
		Match: func(node *yaml.Node, path string) bool {
			return removedBindingKey(bindingKeyOf(path), node)
		},
		Describe: func(at FormMatch) (string, string) {
			if signalType, ok := renamedBindingKey(at.Key, at.Node); ok {
				return "binding " + at.Key + " is named for the signal type it serves", signalType
			}
			return "binding " + at.Key + " served a removed signal or a removed classifier model", "drop it; the decision model answers the signal"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			if signalType, ok := renamedBindingKey(at.Key, at.Node); ok {
				return m.Rename(at.Parent, at.Key, signalType)
			}
			m.Delete(at.Parent, at.Key)
			if strings.HasPrefix(at.Path, "global.") {
				deleteEmpty(m.Root(), "global", "model_catalog", "bindings")
			} else {
				deleteEmpty(scopeRouting(m.Root(), at.Path), "model_bindings")
			}
			return nil
		},
	},
}

var complexityRemovedFields = []string{"hard", "easy", "threshold", "hard_above", "easy_below", "hard_below", "easy_above", "composer", "prototype_scoring"}

var legacyFeedbackLabels = map[string]string{
	"satisfied": "SAT", "need_clarification": "NEED_CLARIFICATION", "wrong_answer": "WRONG_ANSWER", "want_different": "WANT_DIFFERENT",
}

var legacyModalityLabels = map[string]string{"AR": "text", "DIFFUSION": "image", "BOTH": "text_image"}

var removedSystemLines = []string{
	"safety", "hazard", "domain_classifier", "fact_check_classifier", "feedback_detector",
	"prompt_guard", "pii_classifier", "hallucination_detector", "hallucination_explainer",
}

// removedAdmissionKeys gated the removed modules, by signal type or by the
// module's consumer name.
var removedAdmissionKeys = []string{
	"safety", "hazard", "domain", "fact_check", "user_feedback",
	"domain_classifier", "fact_check_classifier", "feedback_detector",
	"jailbreak", "pii", "hallucination", "prompt_guard", "pii_classifier", "hallucination_detector",
}

// removedBindingConsumers are the binding keys of removed signals and modules;
// a decision-task binding of a module the decision model now answers is
// renamed for its signal type.
var removedBindingConsumers = map[string]string{
	"domain_classifier": SignalTypeDomain, "fact_check_classifier": SignalTypeFactCheck,
	"feedback_detector": SignalTypeUserFeedback, "modality_detector": SignalTypeModality,
	"prompt_guard": SignalTypeJailbreak, "pii_classifier": SignalTypePII, "hallucination_detector": SignalTypeHallucination,
	"hazard": "", "classifier": "", "preference": "", "reask": "", "hallucination_explainer": "",
}

// The contracts a binding object named: a decision-task binding of a module
// the decision model now answers, and a remote three-class complexity scorer.
const (
	decisionTaskContract      = "decision.v1"
	labelDistributionContract = "label_distribution.v1"
)

func removedBindingKey(key string, binding *yaml.Node) bool {
	if _, removed := removedBindingConsumers[key]; removed || strings.HasPrefix(key, "classifier.") {
		return true
	}
	// A three-class complexity scorer was a remote classifier; the complexity
	// preset is answered by the decision model.
	return (key == SignalTypeComplexity || strings.HasPrefix(key, SignalTypeComplexity+".")) &&
		scalarAt(binding, "contract") == labelDistributionContract
}

func renamedBindingKey(key string, binding *yaml.Node) (string, bool) {
	signalType := removedBindingConsumers[key]
	return signalType, signalType != "" && scalarAt(binding, "contract") == decisionTaskContract
}

// removedModuleForm is the form of a module only removed signals used. Its
// threshold, when set, becomes the predicate of the rules of collections that
// inherited it and set none.
func removedModuleForm(module, thresholdKey, threshold string, collections []string) ConfigForm {
	path := "global.model_catalog.modules." + module
	becomes := "drop it; the decision model answers the signal"
	if threshold != "" {
		becomes = "drop it; its " + thresholdKey + " is " + threshold
	}
	return ConfigForm{
		Code:    CodeRemovedSection,
		Paths:   []string{path},
		What:    path + " is removed",
		Becomes: becomes,
		Migrate: func(m *Migration, at FormMatch) error {
			if value := mappingValue(at.Node, thresholdKey); thresholdKey != "" && value != nil {
				for _, signals := range allScopeSignals(m.Root()) {
					for _, collection := range collections {
						for _, rule := range sequenceAt(signals, collection) {
							if mappingValue(rule, "predicate") == nil && inheritsModuleThreshold(collection, rule) {
								setNode(rule, "predicate", newMapping("gte", value))
							}
						}
					}
				}
			}
			if hasAnyKey("model_id", "model_ref", "backend")(at.Node, "") {
				m.Note(at.Path, "the module's model is no longer used: bind a deployment that answers the signal's preset with model_bindings")
			}
			if hasAnyKey("keywords", "both_keywords")(at.Node, "") {
				m.Note(at.Path, "the module's keywords are removed: a keywords rule matches them")
			}
			migrateGuardFailures(m, at, collections)
			m.Delete(at.Parent, at.Key)
			parents := strings.Split(path, ".")
			deleteEmpty(m.Root(), parents[:len(parents)-1]...)
			return nil
		},
	}
}

// migrateGuardFailures carries a guard module's failure handling over to its
// jailbreak or pii rules. on_error: block matched a request whose classifier
// failed, so the module's request-stage rules get on_unknown: match; allow,
// the default, is the rules' own default. Content the model does not read in
// full matches whatever on_unscanned said.
func migrateGuardFailures(m *Migration, at FormMatch, collections []string) {
	if !slices.Contains(collections, "jailbreak") && !slices.Contains(collections, "pii") {
		return
	}
	if scalarAt(at.Node, "on_error") == "block" {
		for _, signals := range allScopeSignals(m.Root()) {
			for _, collection := range collections {
				for _, rule := range sequenceAt(signals, collection) {
					if mappingValue(rule, "on_unknown") == nil && scalarAt(rule, "stage") != SystemOneStageResponse {
						setNode(rule, "on_unknown", Scalar(string(RuleOnUnknownMatch)))
					}
				}
			}
		}
	}
	if scalarAt(at.Node, "on_unscanned") == "allow" {
		m.Note(at.Path+".on_unscanned", "allow has no replacement: content the model does not read in full matches the rule")
	}
}

// inheritsModuleThreshold reports whether a rule read its module's threshold:
// a fact_check rule only for its fact-check-needed option.
func inheritsModuleThreshold(collection string, rule *yaml.Node) bool {
	if collection != "fact_check" {
		return true
	}
	labels := stringItems(mappingValue(rule, "labels"))
	return len(labels) == 0 || slices.Equal(labels, []string{"FACT_CHECK_NEEDED"})
}

// embeddingPartitions are a recipe's projection partitions with a
// member that is an embedding rule.
func embeddingPartitions(scope *yaml.Node) []*yaml.Node {
	var embeddings []string
	for _, rule := range sequenceAt(scope, "signals", "embeddings") {
		embeddings = append(embeddings, scalarAt(rule, "name"))
	}
	var partitions []*yaml.Node
	for _, partition := range sequenceAt(scope, "projections", "partitions") {
		if slices.ContainsFunc(stringItems(mappingValue(partition, "members")), func(member string) bool { return slices.Contains(embeddings, member) }) {
			partitions = append(partitions, partition)
		}
	}
	return partitions
}

// piiLabels are the labels the pii preset locates.
func piiLabels() []string {
	preset, _ := presets.Lookup(presets.PII, presets.StageRequest)
	return preset.OptionKeys()
}

// textEmbeddingRule reports whether an embedding rule compares a text or
// image query against text anchors only, which a question can quote.
func textEmbeddingRule(rule *yaml.Node) bool {
	return len(sequenceItems(mappingValue(rule, "image_candidates"))) == 0 && len(sequenceItems(mappingValue(rule, "negative_image_candidates"))) == 0 &&
		!strings.EqualFold(scalarAt(rule, "query_modality"), "audio")
}

func embeddingQuestion(rule *yaml.Node) string {
	question := "Is the request about any of these: " + strings.Join(stringItems(mappingValue(rule, "candidates")), "; ") + "?"
	if negative := stringItems(mappingValue(rule, "negative_candidates")); len(negative) > 0 {
		question += " Answer no if it is about any of these instead: " + strings.Join(negative, "; ") + "."
	}
	return question
}

// domainConditionNames are the domains a recipe's decisions name, in
// order.
func domainConditionNames(scope *yaml.Node) []string {
	var names []string
	var walk func(node *yaml.Node)
	walk = func(node *yaml.Node) {
		if node == nil {
			return
		}
		if node.Kind == yaml.MappingNode && scalarAt(node, "type") == SignalTypeDomain {
			if name := scalarAt(node, "name"); name != "" && !slices.Contains(names, name) {
				names = append(names, name)
			}
		}
		for _, child := range node.Content {
			walk(child)
		}
	}
	for _, decision := range sequenceAt(scope, "decisions") {
		walk(mappingValue(decision, "rules"))
	}
	return names
}

// preferenceQuestion is the name of the systemone question the preferences
// of a scope become: "preference", unless a systemone rule has it.
func preferenceQuestion(signals *yaml.Node) string {
	for _, rule := range sequenceAt(signals, SignalTypeSystemOne) {
		if scalarAt(rule, "name") == preferenceQuestionName {
			return preferenceQuestionName + "s"
		}
	}
	return preferenceQuestionName
}

// scopeRouting is the recipe a path lies in.
func scopeRouting(root *yaml.Node, path string) *yaml.Node {
	label, _, _ := strings.Cut(strings.TrimPrefix(path, "recipes["), "]")
	for index, recipe := range sequenceAt(root, "recipes") {
		if scalarAt(recipe, "name") == label || strconv.Itoa(index) == label {
			return resolveAlias(recipe)
		}
	}
	return nil
}

// scopeSignals is the signals mapping of the recipe a path lies in.
func scopeSignals(root *yaml.Node, path string) *yaml.Node {
	return mappingAt(scopeRouting(root, path), "signals")
}

// allScopeSignals are the signals mappings of every recipe.
func allScopeSignals(root *yaml.Node) []*yaml.Node {
	var scopes []*yaml.Node
	for _, recipe := range sequenceAt(root, "recipes") {
		scopes = append(scopes, mappingAt(recipe, "signals"))
	}
	return slices.DeleteFunc(scopes, func(signals *yaml.Node) bool { return signals == nil })
}

func setConditionType(signalType string) func(*Migration, FormMatch) error {
	return func(m *Migration, at FormMatch) error {
		m.Set(at.Node, "type", Scalar(signalType))
		return nil
	}
}

// moveThreshold writes a rule's threshold as its predicate's bound, unless it
// has a predicate.
func moveThreshold(m *Migration, rule *yaml.Node, bound string) {
	if value := mappingValue(rule, "threshold"); value != nil && mappingValue(rule, "predicate") == nil {
		setNode(rule, "predicate", newMapping(bound, value))
	}
	m.Delete(rule, "threshold")
}

// moveInput writes a rule's boolean field as an input field.
func moveInput(m *Migration, rule *yaml.Node, field, input string) {
	if scalarAt(rule, field) == "true" {
		setNode(ensureMapping(rule, "input"), input, boolScalar())
	}
	m.Delete(rule, field)
}

func hasAnyKey(keys ...string) func(*yaml.Node, string) bool {
	return func(node *yaml.Node, _ string) bool {
		return slices.ContainsFunc(keys, func(key string) bool { return mappingValue(node, key) != nil })
	}
}

// appendRule appends a rule to a signals mapping's collection.
func appendRule(signals *yaml.Node, collection string, rule *yaml.Node) {
	items := mappingValue(signals, collection)
	if items == nil || items.Kind != yaml.SequenceNode {
		items = sequence()
		setNode(signals, collection, items)
	}
	items.Content = append(items.Content, rule)
}

// copiedRule is a new rule mapping with the given fields of rule.
func copiedRule(rule *yaml.Node, keys ...string) *yaml.Node {
	next := newMapping()
	for _, key := range keys {
		if value := mappingValue(rule, key); value != nil {
			setNode(next, key, value)
		}
	}
	return next
}

func sequenceItems(node *yaml.Node) []*yaml.Node {
	if node == nil || node.Kind != yaml.SequenceNode {
		return nil
	}
	return node.Content
}

func stringItems(node *yaml.Node) []string {
	var values []string
	for _, item := range sequenceItems(node) {
		if item.Kind == yaml.ScalarNode {
			values = append(values, item.Value)
		}
	}
	return values
}

// newMapping builds a mapping from alternating keys and values.
func newMapping(pairs ...any) *yaml.Node {
	node := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for i := 0; i+1 < len(pairs); i += 2 {
		setNode(node, pairs[i].(string), pairs[i+1].(*yaml.Node))
	}
	return node
}

func sequence(items ...*yaml.Node) *yaml.Node {
	return &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq", Content: items}
}

func stringSequence(values ...string) *yaml.Node {
	node := sequence()
	node.Style = yaml.FlowStyle
	for _, value := range values {
		node.Content = append(node.Content, Scalar(value))
	}
	return node
}

func boolScalar() *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: "true"}
}

func floatScalar(value float64) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!float", Value: strconv.FormatFloat(math.Round(value*1e6)/1e6, 'f', -1, 64)}
}

// bindingKeyOf is the binding key a model_bindings or bindings path ends in;
// a key may hold dots.
func bindingKeyOf(path string) string {
	if key, ok := strings.CutPrefix(path, "global.model_catalog.bindings."); ok {
		return key
	}
	_, key, _ := strings.Cut(path, "].model_bindings.")
	return key
}

// hazardHarmCategories are the harm categories (presets.HarmCategoryOptions)
// of each Vela 1.0 Hazard label: the AEGIS 2.0 categories the head was trained
// on and the MLCommons hazard codes of the same harms.
var hazardHarmCategories = map[string][]string{
	"violence":             {"Violence", "Threat", "S1"},
	"criminal_activity":    {"Criminal Planning/Confessions", "Illegal Activity", "Fraud/Deception", "S2"},
	"sexual_content":       {"Sexual", "S3", "S12"},
	"child_exploitation":   {"Sexual (minor)", "S4"},
	"hate":                 {"Hate/Identity Hate", "S10"},
	"harassment_abuse":     {"Harassment", "Profanity"},
	"regulated_substances": {"Controlled/Regulated Substances"},
	"weapons":              {"Guns and Illegal Weapons", "S9"},
	"self_harm":            {"Suicide and Self Harm", "S11"},
	"privacy":              {"PII/Privacy", "S7"},
	"specialized_advice":   {"Unauthorized Advice", "S6"},
	"misinformation":       {"Political/Misinformation/Conspiracy"},
}

// harmCategoriesOf are the harm categories a hazard category names: a Vela
// 1.0 Hazard label's, else the harm category of its name.
func harmCategoriesOf(category string) []string {
	if categories, ok := hazardHarmCategories[category]; ok {
		return categories
	}
	for _, option := range presets.HarmCategoryOptions() {
		if strings.EqualFold(option.Key, category) {
			return []string{option.Key}
		}
	}
	return nil
}

// migrateSafetyHazard writes a safety rule with a hazard part as a rule of
// the harm categories its hazard categories name. The rule matched an unsafe
// request whose score in one of the categories reached hazard.threshold; a
// rule with labels matches a request harmful in one of them, so
// hazard.threshold is its predicate. A rule whose categories name no harm
// category keeps its unsafe verdict alone.
func migrateSafetyHazard(m *Migration, at FormMatch) error {
	rule, hazard := at.Parent, at.Node
	path := strings.TrimSuffix(at.Path, ".hazard")
	categories := stringItems(mappingValue(hazard, "categories"))
	var labels, named, unnamed []string
	for _, category := range categories {
		harms := harmCategoriesOf(category)
		if len(harms) == 0 {
			unnamed = append(unnamed, category)
			continue
		}
		named = append(named, category)
		for _, label := range harms {
			if !slices.Contains(labels, label) {
				labels = append(labels, label)
			}
		}
	}
	if scalarAt(rule, "model") != "" {
		m.Note(path, "a safety rule's model is removed: the rule asks the decision model unless a deployment is set")
	}
	for _, key := range []string{"deployment", "model"} {
		if value := scalarAt(hazard, key); value != "" {
			m.Note(path, "hazard."+key+" "+value+" is removed: the rule's deployment, else the decision model, answers the harm categories")
		}
	}
	threshold := mappingValue(hazard, "threshold")
	for _, key := range []string{"labels", "unsafe_labels", "model", "hazard"} {
		m.Delete(rule, key)
	}
	if len(labels) == 0 {
		moveThreshold(m, rule, "gte")
		unmatched := "the hazard part names no category"
		if len(categories) > 0 {
			unmatched = "hazard categories " + strings.Join(categories, ", ") + " name no harm category"
		}
		m.Note(path, unmatched+", so the rule matches an unsafe request as its threshold did, "+
			"and hazard.threshold has no counterpart; name the harm categories it meant in labels")
		return nil
	}
	if gate := scalarAt(rule, "threshold"); gate != "" {
		m.Note(path, "the rule's threshold "+gate+" has no counterpart: it asked for an unsafe verdict before the hazard categories were scored, "+
			"and a rule with labels asks whether the request is harmful in one of them")
	}
	m.Delete(rule, "threshold")
	setNode(rule, "labels", stringSequence(labels...))
	if threshold != nil && mappingValue(rule, "predicate") == nil {
		setNode(rule, "predicate", newMapping("gte", threshold))
	}
	if len(unnamed) > 0 {
		m.Note(path, "hazard categories "+strings.Join(unnamed, ", ")+" name no harm category; name the ones they meant in labels")
	}
	m.Changed(path, "hazard categories "+strings.Join(named, ", ")+" are the harm categories "+strings.Join(labels, ", "))
	return nil
}
