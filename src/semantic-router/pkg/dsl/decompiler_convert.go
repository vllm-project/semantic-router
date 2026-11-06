package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (d *decompiler) keywordToSignal(kw *config.KeywordRule) *SignalDecl {
	fields := make(map[string]Value)
	if kw.Description != "" {
		fields["description"] = StringValue{V: kw.Description}
	}
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
	if kw.IncludeHistory {
		fields["include_history"] = BoolValue{V: true}
	}
	if kw.Count != nil {
		fields["count"] = numericPredicateObject(kw.Count)
	}
	if kw.Density != nil {
		fields["density"] = numericPredicateObject(kw.Density)
	}
	return &SignalDecl{SignalType: "keyword", Name: kw.Name, Fields: fields}
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

func (d *decompiler) conversationToSignal(rule *config.ConversationRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	fields["feature"] = conversationFeatureValue(rule.Feature)
	if rule.Predicate != nil {
		fields["predicate"] = numericPredicateObject(rule.Predicate)
	}
	return &SignalDecl{SignalType: "conversation", Name: rule.Name, Fields: fields}
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

func (d *decompiler) decisionToRoute(dec *config.Decision) *RouteDecl {
	route := &RouteDecl{
		Name:        dec.Name,
		Description: dec.Description,
		OnUnknown:   string(dec.Rules.OnUnknown),
		Priority:    dec.Priority,
		Tier:        dec.Tier,
	}

	if dec.Action != nil {
		route.Action = &ActionDecl{
			Type:        dec.Action.Type,
			Destination: dec.Action.Destination,
		}
	}

	// WHEN
	route.When = decompileRuleNodeToExpr(&dec.Rules)

	// MODEL
	for _, mr := range dec.ModelRefs {
		ref := &ModelRef{
			Model:     mr.Model,
			Reasoning: mr.UseReasoning,
			Mode:      mr.ReasoningMode,
			Effort:    mr.ReasoningEffort,
			LoRA:      mr.LoRAName,
			Weight:    mr.Weight,
		}
		ref.ParamSize = routeParamSize(d.cfg.ModelConfig, mr.Model)
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
		Mode:      model.ReasoningMode,
		Effort:    model.ReasoningEffort,
		LoRA:      model.LoRAName,
		Weight:    model.Weight,
	}
}

// normalizedRuleOperator trims and upper-cases a rule-tree operator for
// comparison. A rule tree loaded through the standard config path has
// already been normalized (see config.NormalizeRuleOperator), but the
// decompiler also runs directly against configs assembled without going
// through that path (e.g. a Kubernetes CR merged for preview), so decompiling
// must not assume the operator is already canonical: an unnormalized value
// like "or" must still be recognized as OR rather than falling through to a
// nil expression (dropping the WHEN clause and matching every request) or an
// invalid lowercase keyword in decompiled DSL text.
func normalizedRuleOperator(operator string) string {
	return strings.ToUpper(strings.TrimSpace(operator))
}

func decompileRuleNodeToExpr(node *config.RuleCombination) BoolExpr {
	if node == nil {
		return nil
	}
	if node.Type != "" {
		return decompileSignalRefNode(node)
	}
	switch normalizedRuleOperator(node.Operator) {
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

func decompileSignalRefNode(
	node *config.RuleCombination,
) *SignalRefExpr {
	fields := map[string]Value{}
	if node.Label != "" {
		fields["label"] = StringValue{V: node.Label}
	}
	if node.Predicate != nil {
		fields["predicate"] = numericPredicateObject(node.Predicate)
	}
	if node.OnError != "" {
		fields["on_error"] = StringValue{V: node.OnError}
	}
	return &SignalRefExpr{
		SignalType: node.Type,
		SignalName: node.Name,
		Fields:     fields,
	}
}

func flattenRuleNodeToExprs(node *config.RuleCombination, op string) []BoolExpr {
	if normalizedRuleOperator(node.Operator) == op {
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
	if mr.ReasoningMode != "" {
		opts = append(opts, fmt.Sprintf("mode = %q", mr.ReasoningMode))
	}
	if mr.LoRAName != "" {
		opts = append(opts, fmt.Sprintf("lora = %q", mr.LoRAName))
	}
	if mr.Weight != 0 {
		opts = append(opts, fmt.Sprintf("weight = %g", mr.Weight))
	}
	if size := routeParamSize(modelConfig, mr.Model); size != "" {
		opts = append(opts, fmt.Sprintf("param_size = %q", size))
	}
	return strings.Join(opts, ", ")
}

// routeParamSize is the param_size a route repeats from model_config. A
// catalog-backed model's size belongs to its built-in card, not to the route.
func routeParamSize(modelConfig map[string]config.ModelParams, model string) string {
	mc, ok := modelConfig[model]
	if !ok || mc.Catalog != "" {
		return ""
	}
	return mc.ParamSize
}
