package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (d *decompiler) decompileKeywordSignals() {
	for _, kw := range d.cfg.KeywordRules {
		d.write("SIGNAL keyword %s {\n", quoteName(kw.Name))
		if kw.Operator != "" {
			d.write("  operator: %q\n", kw.Operator)
		}
		if len(kw.Keywords) > 0 {
			d.write("  keywords: %s\n", formatStringArray(kw.Keywords))
		}
		if kw.CaseSensitive {
			d.write("  case_sensitive: true\n")
		}
		if kw.Method != "" {
			d.write("  method: %q\n", kw.Method)
		}
		if kw.FuzzyMatch {
			d.write("  fuzzy_match: true\n")
		}
		if kw.FuzzyThreshold != 0 {
			d.write("  fuzzy_threshold: %d\n", kw.FuzzyThreshold)
		}
		if kw.BM25Threshold != 0 {
			d.write("  bm25_threshold: %v\n", kw.BM25Threshold)
		}
		if kw.NgramThreshold != 0 {
			d.write("  ngram_threshold: %v\n", kw.NgramThreshold)
		}
		if kw.NgramArity != 0 {
			d.write("  ngram_arity: %d\n", kw.NgramArity)
		}
		if kw.IncludeHistory {
			d.write("  include_history: true\n")
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileLanguageSignals() {
	for _, lang := range d.cfg.LanguageRules {
		d.write("SIGNAL language %s {\n", quoteName(lang.Name))
		if lang.Description != "" {
			d.write("  description: %q\n", lang.Description)
		}
		if lang.Threshold != 0 {
			d.write("  threshold: %g\n", lang.Threshold)
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileContextSignals() {
	for _, ctx := range d.cfg.ContextRules {
		d.write("SIGNAL context %s {\n", quoteName(ctx.Name))
		if ctx.Description != "" {
			d.write("  description: %q\n", ctx.Description)
		}
		if ctx.MinTokens != "" {
			d.write("  min_tokens: %q\n", string(ctx.MinTokens))
		}
		if ctx.MaxTokens != "" {
			d.write("  max_tokens: %q\n", string(ctx.MaxTokens))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileStructureSignals() {
	for _, structure := range d.cfg.StructureRules {
		d.write("SIGNAL structure %s {\n", quoteName(structure.Name))
		if structure.Description != "" {
			d.write("  description: %q\n", structure.Description)
		}
		d.write("  feature: %s\n", formatPluginConfigValue(structureFeatureToMap(structure.Feature)))
		if structure.Predicate != nil {
			d.write("  predicate: %s\n", formatPluginConfigValue(structurePredicateToMap(structure.Predicate)))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileConversationSignals() {
	for _, conv := range d.cfg.ConversationRules {
		d.write("SIGNAL conversation %s {\n", quoteName(conv.Name))
		if conv.Description != "" {
			d.write("  description: %q\n", conv.Description)
		}
		d.write("  feature: %s\n", formatPluginConfigValue(conversationFeatureToMap(conv.Feature)))
		if conv.Predicate != nil {
			d.write("  predicate: %s\n", formatPluginConfigValue(structurePredicateToMap(conv.Predicate)))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileMetadataSignals() {
	for _, rule := range d.cfg.MetadataRules {
		d.write("SIGNAL metadata %s {\n", quoteName(rule.Name))
		if rule.Description != "" {
			d.write("  description: %q\n", rule.Description)
		}
		d.write("  key: %q\n", rule.Key)
		predicate := map[string]interface{}{}
		switch {
		case rule.Predicate.Equals != nil:
			predicate["equals"] = *rule.Predicate.Equals
		case len(rule.Predicate.In) > 0:
			predicate["in"] = rule.Predicate.In
		case rule.Predicate.Exists != nil:
			predicate["exists"] = *rule.Predicate.Exists
		}
		d.write(
			"  predicate: %s\n",
			formatPluginConfigValue(predicate),
		)
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileInputModalitySignals() {
	for _, rule := range d.cfg.InputModalityRules {
		d.write("SIGNAL input_modality %s {\n", quoteName(rule.Name))
		if rule.Description != "" {
			d.write("  description: %q\n", rule.Description)
		}
		d.write("  modality: %q\n", rule.Modality)
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileSystemOneSignals() {
	for _, rule := range d.cfg.SystemOneRules {
		d.write("SIGNAL systemone %s {\n", quoteName(rule.Name))
		if rule.Description != "" {
			d.write("  description: %q\n", rule.Description)
		}
		if rule.Deployment != "" {
			d.write("  deployment: %q\n", rule.Deployment)
		}
		d.write("  question: %s\n", formatPluginConfigValue(systemOneQuestionValue(rule.Question)))
		for _, labels := range []struct {
			field string
			keys  []string
		}{{"labels", rule.Labels}, {"ignore_labels", rule.IgnoreLabels}} {
			if len(labels.keys) > 0 {
				d.write("  %s: %s\n", labels.field, formatPluginConfigValue(stringListValue(labels.keys)))
			}
		}
		if predicate := numericPredicateValue(rule.Predicate); predicate != nil {
			d.write("  predicate: %s\n", formatPluginConfigValue(predicate))
		}
		if rule.Stage != "" {
			d.write("  stage: %q\n", rule.Stage)
		}
		if input := systemOneInputValue(rule.Input); len(input) > 0 {
			d.write("  input: %s\n", formatPluginConfigValue(input))
		}
		if rule.Coverage != "" {
			d.write("  coverage: %q\n", rule.Coverage)
		}
		if rule.OnUnscanned != "" {
			d.write("  on_unscanned: %q\n", rule.OnUnscanned)
		}
		if rule.TimeoutMs > 0 {
			d.write("  timeout_ms: %d\n", rule.TimeoutMs)
		}
		d.write("}\n\n")
	}
}

func stringListValue(values []string) []interface{} {
	list := make([]interface{}, len(values))
	for index, value := range values {
		list[index] = value
	}
	return list
}

// systemOneInputValue holds the input fields a rule sets.
func systemOneInputValue(input config.SystemOneInput) map[string]interface{} {
	value := map[string]interface{}{}
	for key, set := range map[string]bool{
		"history": input.History, "answer": input.Answer, "context": input.Context, "tool_results": input.ToolResults,
		"image": input.Image, "audio": input.Audio, "video": input.Video,
	} {
		if set {
			value[key] = true
		}
	}
	if input.PreviousTurns > 0 {
		value["previous_turns"] = input.PreviousTurns
	}
	return value
}

func systemOneQuestionValue(question config.SystemOneQuestion) map[string]interface{} {
	value := map[string]interface{}{"type": question.Type, "instructions": question.Instructions}
	if len(question.Choices) > 0 {
		value["choices"] = decisionChoicesValue(question.Choices)
	}
	if len(question.Levels) > 0 {
		levels := make([]interface{}, 0, len(question.Levels))
		for _, level := range question.Levels {
			levels = append(levels, level)
		}
		value["levels"] = levels
	}
	if len(question.Labels) > 0 {
		value["labels"] = decisionChoicesValue(question.Labels)
	}
	if question.Threshold != nil {
		value["threshold"] = *question.Threshold
	}
	if question.Head != "" {
		value["head"] = question.Head
	}
	return value
}

func decisionChoicesValue(choices []config.SystemOneOption) []interface{} {
	entries := make([]interface{}, 0, len(choices))
	for _, choice := range choices {
		entry := map[string]interface{}{"key": choice.Key}
		if choice.Description != "" {
			entry["description"] = choice.Description
		}
		entries = append(entries, entry)
	}
	return entries
}

func numericPredicateValue(predicate *config.NumericPredicate) map[string]interface{} {
	if predicate == nil {
		return nil
	}
	value := map[string]interface{}{}
	for key, bound := range map[string]*float64{"gt": predicate.GT, "gte": predicate.GTE, "lt": predicate.LT, "lte": predicate.LTE} {
		if bound != nil {
			value[key] = *bound
		}
	}
	return value
}

func (d *decompiler) decompileAuthzSignals() {
	for _, rb := range d.cfg.RoleBindings {
		d.write("SIGNAL authz %s {\n", quoteName(rb.Name))
		if rb.Role != "" {
			d.write("  role: %q\n", rb.Role)
		}
		if rb.Description != "" {
			d.write("  description: %q\n", rb.Description)
		}
		if len(rb.Subjects) > 0 {
			d.write("  subjects: [")
			for i, subj := range rb.Subjects {
				if i > 0 {
					d.write(", ")
				}
				d.write("{ kind: %q, name: %q }", subj.Kind, subj.Name)
			}
			d.write("]\n")
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileEventSignals() {
	for _, rule := range d.cfg.EventRules {
		d.write("SIGNAL event %s {\n", quoteName(rule.Name))
		if len(rule.EventTypes) > 0 {
			d.write("  event_types: %s\n", formatStringArray(rule.EventTypes))
		}
		if len(rule.Severities) > 0 {
			d.write("  severities: %s\n", formatStringArray(rule.Severities))
		}
		if len(rule.ActionCodes) > 0 {
			d.write("  action_codes: %s\n", formatStringArray(rule.ActionCodes))
		}
		if rule.Temporal {
			d.write("  temporal: true\n")
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileProjectionSignals() {
	for _, partition := range d.cfg.Projections.Partitions {
		d.decompileProjectionPartition(partition)
	}
	for _, score := range d.cfg.Projections.Scores {
		d.decompileProjectionScore(score)
	}
	for _, mapping := range d.cfg.Projections.Mappings {
		d.decompileProjectionMapping(mapping)
	}
}

func (d *decompiler) decompileProjectionPartition(partition config.ProjectionPartition) {
	d.write("PROJECTION partition %s {\n", quoteName(partition.Name))
	if partition.Semantics != "" {
		d.write("  semantics: %q\n", partition.Semantics)
	}
	if partition.Temperature != 0 {
		d.write("  temperature: %v\n", partition.Temperature)
	}
	if len(partition.Members) > 0 {
		d.write("  members: %s\n", formatStringArray(partition.Members))
	}
	if partition.Default != "" {
		d.write("  default: %q\n", partition.Default)
	}
	d.write("}\n\n")
}

func (d *decompiler) decompileProjectionScore(score config.ProjectionScore) {
	d.write("PROJECTION score %s {\n", quoteName(score.Name))
	if score.Method != "" {
		d.write("  method: %q\n", score.Method)
	}
	if len(score.Inputs) > 0 {
		d.write("  inputs: %s\n", formatProjectionScoreInputs(score.Inputs))
	}
	d.write("}\n\n")
}

func (d *decompiler) decompileProjectionMapping(mapping config.ProjectionMapping) {
	d.write("PROJECTION mapping %s {\n", quoteName(mapping.Name))
	if mapping.Source != "" {
		d.write("  source: %q\n", mapping.Source)
	}
	if mapping.Method != "" {
		d.write("  method: %q\n", mapping.Method)
	}
	if mapping.Calibration != nil {
		d.write("  calibration: %s\n", formatProjectionMappingCalibration(mapping.Calibration))
	}
	if len(mapping.Outputs) > 0 {
		d.write("  outputs: %s\n", formatProjectionMappingOutputs(mapping.Outputs))
	}
	d.write("}\n\n")
}
