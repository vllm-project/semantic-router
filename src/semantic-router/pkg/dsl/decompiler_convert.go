package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (d *decompiler) categoryToSignal(cat *config.Category) *SignalDecl {
	fields := make(map[string]Value)
	if cat.Description != "" {
		fields["description"] = StringValue{V: cat.Description}
	}
	if len(cat.MMLUCategories) > 0 {
		fields["mmlu_categories"] = stringsToArray(cat.MMLUCategories)
	}
	if len(cat.ModelScores) > 0 {
		fields["model_scores"] = modelScoresValue(cat.ModelScores)
	}
	return &SignalDecl{SignalType: "domain", Name: cat.Name, Fields: fields}
}

func (d *decompiler) keywordToSignal(kw *config.KeywordRule) *SignalDecl {
	fields := make(map[string]Value)
	if kw.Operator != "" {
		fields["operator"] = StringValue{V: kw.Operator}
	}
	if len(kw.Keywords) > 0 {
		fields["keywords"] = stringsToArray(kw.Keywords)
	}
	if kw.CaseSensitive {
		fields["case_sensitive"] = BoolValue{V: true}
	}
	if kw.Method != "" {
		fields["method"] = StringValue{V: kw.Method}
	}
	if kw.FuzzyMatch {
		fields["fuzzy_match"] = BoolValue{V: true}
	}
	if kw.FuzzyThreshold != 0 {
		fields["fuzzy_threshold"] = IntValue{V: kw.FuzzyThreshold}
	}
	if kw.BM25Threshold != 0 {
		fields["bm25_threshold"] = FloatValue{V: float64(kw.BM25Threshold)}
	}
	if kw.NgramThreshold != 0 {
		fields["ngram_threshold"] = FloatValue{V: float64(kw.NgramThreshold)}
	}
	if kw.NgramArity != 0 {
		fields["ngram_arity"] = IntValue{V: kw.NgramArity}
	}
	return &SignalDecl{SignalType: "keyword", Name: kw.Name, Fields: fields}
}

func (d *decompiler) embeddingToSignal(emb *config.EmbeddingRule) *SignalDecl {
	fields := make(map[string]Value)
	if emb.SimilarityThreshold != 0 {
		fields["threshold"] = FloatValue{V: float64(emb.SimilarityThreshold)}
	}
	if len(emb.Candidates) > 0 {
		fields["candidates"] = stringsToArray(emb.Candidates)
	}
	if emb.AggregationMethodConfiged != "" {
		fields["aggregation_method"] = StringValue{V: string(emb.AggregationMethodConfiged)}
	}
	if emb.QueryModality != "" && emb.QueryModality != config.QueryModalityText {
		fields["query_modality"] = StringValue{V: string(emb.QueryModality)}
	}
	return &SignalDecl{SignalType: "embedding", Name: emb.Name, Fields: fields}
}

func (d *decompiler) factCheckToSignal(fc *config.FactCheckRule) *SignalDecl {
	fields := make(map[string]Value)
	if fc.Description != "" {
		fields["description"] = StringValue{V: fc.Description}
	}
	return &SignalDecl{SignalType: "fact_check", Name: fc.Name, Fields: fields}
}

func (d *decompiler) userFeedbackToSignal(uf *config.UserFeedbackRule) *SignalDecl {
	fields := make(map[string]Value)
	if uf.Description != "" {
		fields["description"] = StringValue{V: uf.Description}
	}
	return &SignalDecl{SignalType: "user_feedback", Name: uf.Name, Fields: fields}
}

func (d *decompiler) reaskToSignal(rule *config.ReaskRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	if rule.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(rule.Threshold)}
	}
	if rule.LookbackTurns != 0 {
		fields["lookback_turns"] = IntValue{V: rule.LookbackTurns}
	}
	return &SignalDecl{SignalType: "reask", Name: rule.Name, Fields: fields}
}

func (d *decompiler) preferenceToSignal(pref *config.PreferenceRule) *SignalDecl {
	fields := make(map[string]Value)
	if pref.Description != "" {
		fields["description"] = StringValue{V: pref.Description}
	}
	if len(pref.Examples) > 0 {
		fields["examples"] = stringsToArray(pref.Examples)
	}
	if pref.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(pref.Threshold)}
	}
	return &SignalDecl{SignalType: "preference", Name: pref.Name, Fields: fields}
}

func (d *decompiler) languageToSignal(lang *config.LanguageRule) *SignalDecl {
	fields := make(map[string]Value)
	if lang.Description != "" {
		fields["description"] = StringValue{V: lang.Description}
	}
	if lang.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(lang.Threshold)}
	}
	return &SignalDecl{SignalType: "language", Name: lang.Name, Fields: fields}
}

func (d *decompiler) contextToSignal(ctx *config.ContextRule) *SignalDecl {
	fields := make(map[string]Value)
	if ctx.MinTokens != "" {
		fields["min_tokens"] = StringValue{V: string(ctx.MinTokens)}
	}
	if ctx.MaxTokens != "" {
		fields["max_tokens"] = StringValue{V: string(ctx.MaxTokens)}
	}
	if ctx.Description != "" {
		fields["description"] = StringValue{V: ctx.Description}
	}
	return &SignalDecl{SignalType: "context", Name: ctx.Name, Fields: fields}
}

func (d *decompiler) structureToSignal(rule *config.StructureRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	fields["feature"] = structureFeatureValue(rule.Feature)
	if rule.Predicate != nil {
		fields["predicate"] = structurePredicateValue(rule.Predicate)
	}
	return &SignalDecl{SignalType: "structure", Name: rule.Name, Fields: fields}
}

func (d *decompiler) conversationToSignal(rule *config.ConversationRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	fields["feature"] = conversationFeatureValue(rule.Feature)
	if rule.Predicate != nil {
		fields["predicate"] = structurePredicateValue(rule.Predicate)
	}
	return &SignalDecl{SignalType: "conversation", Name: rule.Name, Fields: fields}
}

func (d *decompiler) complexityToSignal(comp *config.ComplexityRule) *SignalDecl {
	fields := make(map[string]Value)
	if comp.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(comp.Threshold)}
	}
	if comp.Description != "" {
		fields["description"] = StringValue{V: comp.Description}
	}
	if len(comp.Hard.Candidates) > 0 || len(comp.Hard.ImageCandidates) > 0 {
		fields["hard"] = complexityCandidatesValue(comp.Hard)
	}
	if len(comp.Easy.Candidates) > 0 || len(comp.Easy.ImageCandidates) > 0 {
		fields["easy"] = complexityCandidatesValue(comp.Easy)
	}
	if comp.Composer != nil {
		fields["composer"] = ruleCombinationValue(comp.Composer)
	}
	return &SignalDecl{SignalType: "complexity", Name: comp.Name, Fields: fields}
}

func (d *decompiler) modalityToSignal(mod *config.ModalityRule) *SignalDecl {
	fields := make(map[string]Value)
	if mod.Description != "" {
		fields["description"] = StringValue{V: mod.Description}
	}
	return &SignalDecl{SignalType: "modality", Name: mod.Name, Fields: fields}
}

func (d *decompiler) roleBindingToSignal(rb *config.RoleBinding) *SignalDecl {
	fields := make(map[string]Value)
	if rb.Role != "" {
		fields["role"] = StringValue{V: rb.Role}
	}
	if rb.Description != "" {
		fields["description"] = StringValue{V: rb.Description}
	}
	if len(rb.Subjects) > 0 {
		var items []Value
		for _, subj := range rb.Subjects {
			items = append(items, ObjectValue{Fields: map[string]Value{
				"kind": StringValue{V: subj.Kind},
				"name": StringValue{V: subj.Name},
			}})
		}
		fields["subjects"] = ArrayValue{Items: items}
	}
	return &SignalDecl{SignalType: "authz", Name: rb.Name, Fields: fields}
}

func (d *decompiler) jailbreakToSignal(jb *config.JailbreakRule) *SignalDecl {
	fields := make(map[string]Value)
	if jb.Method != "" {
		fields["method"] = StringValue{V: jb.Method}
	}
	if jb.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(jb.Threshold)}
	}
	if jb.IncludeHistory {
		fields["include_history"] = BoolValue{V: true}
	}
	if jb.Description != "" {
		fields["description"] = StringValue{V: jb.Description}
	}
	if len(jb.JailbreakPatterns) > 0 {
		fields["jailbreak_patterns"] = stringsToArray(jb.JailbreakPatterns)
	}
	if len(jb.BenignPatterns) > 0 {
		fields["benign_patterns"] = stringsToArray(jb.BenignPatterns)
	}
	return &SignalDecl{SignalType: "jailbreak", Name: jb.Name, Fields: fields}
}

func (d *decompiler) piiToSignal(pii *config.PIIRule) *SignalDecl {
	fields := make(map[string]Value)
	if pii.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(pii.Threshold)}
	}
	if len(pii.PIITypesAllowed) > 0 {
		fields["pii_types_allowed"] = stringsToArray(pii.PIITypesAllowed)
	}
	if pii.IncludeHistory {
		fields["include_history"] = BoolValue{V: true}
	}
	if pii.Description != "" {
		fields["description"] = StringValue{V: pii.Description}
	}
	return &SignalDecl{SignalType: "pii", Name: pii.Name, Fields: fields}
}

func (d *decompiler) kbSignalToDecl(rule *config.KBSignalRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.KB != "" {
		fields["kb"] = StringValue{V: rule.KB}
	}
	fields["target"] = ObjectValue{Fields: map[string]Value{
		"kind":  StringValue{V: rule.Target.Kind},
		"value": StringValue{V: rule.Target.Value},
	}}
	if rule.Match != "" {
		fields["match"] = StringValue{V: rule.Match}
	}
	return &SignalDecl{SignalType: "kb", Name: rule.Name, Fields: fields}
}

func (d *decompiler) eventRuleToDecl(rule *config.EventRule) *SignalDecl {
	fields := make(map[string]Value)
	if len(rule.EventTypes) > 0 {
		fields["event_types"] = stringsToArray(rule.EventTypes)
	}
	if len(rule.Severities) > 0 {
		fields["severities"] = stringsToArray(rule.Severities)
	}
	if len(rule.ActionCodes) > 0 {
		fields["action_codes"] = stringsToArray(rule.ActionCodes)
	}
	if rule.Temporal {
		fields["temporal"] = BoolValue{V: true}
	}
	return &SignalDecl{SignalType: "event", Name: rule.Name, Fields: fields}
}

func modelScoresValue(scores []config.ModelScore) ArrayValue {
	items := make([]Value, 0, len(scores))
	for _, score := range scores {
		fields := map[string]Value{
			"model": StringValue{V: score.Model},
			"score": FloatValue{V: score.Score},
		}
		if score.UseReasoning != nil {
			fields["use_reasoning"] = BoolValue{V: *score.UseReasoning}
		}
		items = append(items, ObjectValue{Fields: fields})
	}
	return ArrayValue{Items: items}
}

func complexityCandidatesValue(candidates config.ComplexityCandidates) ObjectValue {
	fields := make(map[string]Value)
	if len(candidates.Candidates) > 0 {
		fields["candidates"] = stringsToArray(candidates.Candidates)
	}
	if len(candidates.ImageCandidates) > 0 {
		fields["image_candidates"] = stringsToArray(candidates.ImageCandidates)
	}
	return ObjectValue{Fields: fields}
}

func ruleCombinationValue(node *config.RuleCombination) ObjectValue {
	fields := make(map[string]Value)
	if node == nil {
		return ObjectValue{Fields: fields}
	}
	if node.Type != "" {
		fields["type"] = StringValue{V: node.Type}
		fields["name"] = StringValue{V: node.Name}
		return ObjectValue{Fields: fields}
	}
	if node.Operator != "" {
		fields["operator"] = StringValue{V: node.Operator}
	}
	if len(node.Conditions) > 0 {
		items := make([]Value, 0, len(node.Conditions))
		for i := range node.Conditions {
			items = append(items, ruleCombinationValue(&node.Conditions[i]))
		}
		fields["conditions"] = ArrayValue{Items: items}
	}
	return ObjectValue{Fields: fields}
}

func (d *decompiler) decisionToRoute(dec *config.Decision) *RouteDecl {
	route := &RouteDecl{
		Name:        dec.Name,
		Description: dec.Description,
		Priority:    dec.Priority,
		Tier:        dec.Tier,
	}

	// WHEN
	route.When = decompileRuleNodeToExpr(&dec.Rules)

	// MODEL
	for _, mr := range dec.ModelRefs {
		ref := &ModelRef{
			Model:     mr.Model,
			Reasoning: mr.UseReasoning,
			Effort:    mr.ReasoningEffort,
			LoRA:      mr.LoRAName,
			Weight:    mr.Weight,
		}
		// Pull param_size from model_config.
		if mc, ok := d.cfg.ModelConfig[mr.Model]; ok {
			ref.ParamSize = mc.ParamSize
		}
		route.Models = append(route.Models, ref)
	}

	// ALGORITHM
	if dec.Algorithm != nil && dec.Algorithm.Type != "" {
		algoSpec := &AlgoSpec{AlgoType: dec.Algorithm.Type}
		algoSpec.Fields = d.algorithmToFields(dec.Algorithm)
		route.Algorithm = algoSpec
	}

	// PLUGINs
	for _, p := range dec.Plugins {
		ref := &PluginRef{Name: sanitizeName(p.Type)}
		fields := pluginConfigToFields(&p)
		if len(fields) > 0 {
			ref.Fields = fields
		}
		route.Plugins = append(route.Plugins, ref)
	}

	for _, iter := range dec.CandidateIterations {
		route.CandidateIterations = append(route.CandidateIterations, candidateIterationConfigToDecl(iter))
	}

	for _, e := range dec.Emits {
		route.Emits = append(route.Emits, emitDirectiveToDecl(e))
	}

	return route
}

func candidateIterationConfigToDecl(iter config.CandidateIterationConfig) *CandidateIterationDecl {
	decl := &CandidateIterationDecl{
		Variable: iter.Variable,
		Source:   iter.Source,
	}
	for _, model := range iter.Models {
		decl.Models = append(decl.Models, configModelRefToDSLModelRef(model))
	}
	for _, output := range iter.Outputs {
		decl.Outputs = append(decl.Outputs, &CandidateIterationOutputDecl{
			Type:  output.Type,
			Value: output.Value,
		})
	}
	return decl
}

func configModelRefToDSLModelRef(model config.ModelRef) *ModelRef {
	return &ModelRef{
		Model:     model.Model,
		Reasoning: model.UseReasoning,
		Effort:    model.ReasoningEffort,
		LoRA:      model.LoRAName,
		Weight:    model.Weight,
	}
}

func decompileRuleNodeToExpr(node *config.RuleCombination) BoolExpr {
	if node == nil {
		return nil
	}
	if node.Type != "" {
		return &SignalRefExpr{SignalType: node.Type, SignalName: node.Name}
	}
	switch node.Operator {
	case "AND":
		exprs := flattenRuleNodeToExprs(node, "AND")
		if len(exprs) == 0 {
			return nil
		}
		result := exprs[0]
		for i := 1; i < len(exprs); i++ {
			result = &BoolAnd{Left: result, Right: exprs[i]}
		}
		return result
	case "OR":
		exprs := flattenRuleNodeToExprs(node, "OR")
		if len(exprs) == 0 {
			return nil
		}
		result := exprs[0]
		for i := 1; i < len(exprs); i++ {
			result = &BoolOr{Left: result, Right: exprs[i]}
		}
		return result
	case "NOT":
		if len(node.Conditions) == 1 {
			return &BoolNot{Expr: decompileRuleNodeToExpr(&node.Conditions[0])}
		}
	}
	return nil
}

func flattenRuleNodeToExprs(node *config.RuleCombination, op string) []BoolExpr {
	if node.Operator == op {
		var exprs []BoolExpr
		for i := range node.Conditions {
			exprs = append(exprs, flattenRuleNodeToExprs(&node.Conditions[i], op)...)
		}
		return exprs
	}
	return []BoolExpr{decompileRuleNodeToExpr(node)}
}

func modelRefOptions(mr *config.ModelRef, modelConfig map[string]config.ModelParams) string {
	var opts []string
	if mr.UseReasoning != nil {
		if *mr.UseReasoning {
			opts = append(opts, "reasoning = true")
		} else {
			opts = append(opts, "reasoning = false")
		}
	}
	if mr.ReasoningEffort != "" {
		opts = append(opts, fmt.Sprintf("effort = %q", mr.ReasoningEffort))
	}
	if mr.LoRAName != "" {
		opts = append(opts, fmt.Sprintf("lora = %q", mr.LoRAName))
	}
	if mr.Weight != 0 {
		opts = append(opts, fmt.Sprintf("weight = %g", mr.Weight))
	}
	// Pull param_size from model_config.
	if mc, ok := modelConfig[mr.Model]; ok {
		if mc.ParamSize != "" {
			opts = append(opts, fmt.Sprintf("param_size = %q", mc.ParamSize))
		}
	}
	return strings.Join(opts, ", ")
}
