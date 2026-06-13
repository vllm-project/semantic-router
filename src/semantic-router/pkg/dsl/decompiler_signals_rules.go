package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (d *decompiler) decompileDomainSignals() {
	for _, cat := range d.cfg.Categories {
		d.write("SIGNAL domain %s {\n", quoteName(cat.Name))
		if cat.Description != "" {
			d.write("  description: %q\n", cat.Description)
		}
		if len(cat.MMLUCategories) > 0 {
			d.write("  mmlu_categories: %s\n", formatStringArray(cat.MMLUCategories))
		}
		if len(cat.ModelScores) > 0 {
			d.write("  model_scores: %s\n", formatPluginConfigValue(modelScoresToList(cat.ModelScores)))
		}
		d.write("}\n\n")
	}
}

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
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileEmbeddingSignals() {
	for _, emb := range d.cfg.EmbeddingRules {
		d.write("SIGNAL embedding %s {\n", quoteName(emb.Name))
		if emb.SimilarityThreshold != 0 {
			d.write("  threshold: %v\n", emb.SimilarityThreshold)
		}
		if len(emb.Candidates) > 0 {
			d.write("  candidates: %s\n", formatStringArray(emb.Candidates))
		}
		if emb.AggregationMethodConfiged != "" {
			d.write("  aggregation_method: %q\n", string(emb.AggregationMethodConfiged))
		}
		if emb.QueryModality != "" && emb.QueryModality != config.QueryModalityText {
			d.write("  query_modality: %q\n", string(emb.QueryModality))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileFactCheckSignals() {
	for _, fc := range d.cfg.FactCheckRules {
		d.write("SIGNAL fact_check %s {\n", quoteName(fc.Name))
		if fc.Description != "" {
			d.write("  description: %q\n", fc.Description)
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileUserFeedbackSignals() {
	for _, uf := range d.cfg.UserFeedbackRules {
		d.write("SIGNAL user_feedback %s {\n", quoteName(uf.Name))
		if uf.Description != "" {
			d.write("  description: %q\n", uf.Description)
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileReaskSignals() {
	for _, rule := range d.cfg.ReaskRules {
		d.write("SIGNAL reask %s {\n", quoteName(rule.Name))
		if rule.Description != "" {
			d.write("  description: %q\n", rule.Description)
		}
		if rule.Threshold != 0 {
			d.write("  threshold: %g\n", rule.Threshold)
		}
		if rule.LookbackTurns != 0 {
			d.write("  lookback_turns: %d\n", rule.LookbackTurns)
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompilePreferenceSignals() {
	for _, pref := range d.cfg.PreferenceRules {
		d.write("SIGNAL preference %s {\n", quoteName(pref.Name))
		if pref.Description != "" {
			d.write("  description: %q\n", pref.Description)
		}
		if len(pref.Examples) > 0 {
			d.write("  examples: %s\n", formatStringArray(pref.Examples))
		}
		if pref.Threshold != 0 {
			d.write("  threshold: %g\n", pref.Threshold)
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

func (d *decompiler) decompileComplexitySignals() {
	for _, comp := range d.cfg.ComplexityRules {
		d.write("SIGNAL complexity %s {\n", quoteName(comp.Name))
		if comp.Threshold != 0 {
			d.write("  threshold: %v\n", comp.Threshold)
		}
		if comp.Description != "" {
			d.write("  description: %q\n", comp.Description)
		}
		if comp.Composer != nil {
			d.write("  composer: %s\n", decompileComposerObj(comp.Composer))
		}
		if len(comp.Hard.Candidates) > 0 || len(comp.Hard.ImageCandidates) > 0 {
			d.write("  hard: %s\n", formatPluginConfigValue(complexityCandidatesToMap(comp.Hard)))
		}
		if len(comp.Easy.Candidates) > 0 || len(comp.Easy.ImageCandidates) > 0 {
			d.write("  easy: %s\n", formatPluginConfigValue(complexityCandidatesToMap(comp.Easy)))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileModalitySignals() {
	for _, mod := range d.cfg.ModalityRules {
		d.write("SIGNAL modality %s {\n", quoteName(mod.Name))
		if mod.Description != "" {
			d.write("  description: %q\n", mod.Description)
		}
		d.write("}\n\n")
	}
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

func (d *decompiler) decompileJailbreakSignals() {
	for _, jb := range d.cfg.JailbreakRules {
		d.write("SIGNAL jailbreak %s {\n", quoteName(jb.Name))
		if jb.Method != "" {
			d.write("  method: %q\n", jb.Method)
		}
		if jb.Threshold != 0 {
			d.write("  threshold: %v\n", jb.Threshold)
		}
		if jb.IncludeHistory {
			d.write("  include_history: true\n")
		}
		if jb.Description != "" {
			d.write("  description: %q\n", jb.Description)
		}
		if len(jb.JailbreakPatterns) > 0 {
			d.write("  jailbreak_patterns: %s\n", formatStringArray(jb.JailbreakPatterns))
		}
		if len(jb.BenignPatterns) > 0 {
			d.write("  benign_patterns: %s\n", formatStringArray(jb.BenignPatterns))
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompilePIISignals() {
	for _, pii := range d.cfg.PIIRules {
		d.write("SIGNAL pii %s {\n", quoteName(pii.Name))
		if pii.Threshold != 0 {
			d.write("  threshold: %v\n", pii.Threshold)
		}
		if len(pii.PIITypesAllowed) > 0 {
			d.write("  pii_types_allowed: %s\n", formatStringArray(pii.PIITypesAllowed))
		}
		if pii.IncludeHistory {
			d.write("  include_history: true\n")
		}
		if pii.Description != "" {
			d.write("  description: %q\n", pii.Description)
		}
		d.write("}\n\n")
	}
}

func (d *decompiler) decompileKBSignals() {
	for _, kb := range d.cfg.KBRules {
		d.write("SIGNAL kb %s {\n", quoteName(kb.Name))
		if kb.KB != "" {
			d.write("  kb: %q\n", kb.KB)
		}
		d.write("  target: { kind: %q, value: %q }\n", kb.Target.Kind, kb.Target.Value)
		if kb.Match != "" {
			d.write("  match: %q\n", kb.Match)
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

func modelScoresToList(scores []config.ModelScore) []interface{} {
	items := make([]interface{}, 0, len(scores))
	for _, score := range scores {
		item := map[string]interface{}{
			"model": score.Model,
			"score": score.Score,
		}
		if score.UseReasoning != nil {
			item["use_reasoning"] = *score.UseReasoning
		}
		items = append(items, item)
	}
	return items
}

func complexityCandidatesToMap(candidates config.ComplexityCandidates) map[string]interface{} {
	fields := make(map[string]interface{})
	if len(candidates.Candidates) > 0 {
		fields["candidates"] = candidates.Candidates
	}
	if len(candidates.ImageCandidates) > 0 {
		fields["image_candidates"] = candidates.ImageCandidates
	}
	return fields
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
