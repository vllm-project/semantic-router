package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Compiler) compileRoutes() {
	for _, r := range c.prog.Routes {
		c.config.Decisions = append(c.config.Decisions, c.compileRoute(r))
	}
}

func (c *Compiler) compileRoute(r *RouteDecl) config.Decision {
	decision := config.Decision{
		Name:        r.Name,
		Description: r.Description,
		Priority:    r.Priority,
		Tier:        r.Tier,
		Rules:       c.compileRouteRules(r),
	}

	for _, m := range r.Models {
		c.appendModelRef(&decision, m)
	}

	for _, iter := range r.CandidateIterations {
		compiled := c.compileCandidateIteration(iter)
		decision.CandidateIterations = append(decision.CandidateIterations, compiled)
		if compiled.Source == "models" && iterEmitsVariable(compiled) {
			for _, model := range iter.Models {
				c.appendModelRefIfMissing(&decision, model)
			}
		}
	}

	if r.Algorithm != nil {
		decision.Algorithm = c.compileAlgorithm(r.Algorithm)
	}

	for _, pr := range r.Plugins {
		if dp := c.compilePluginRef(pr); dp != nil {
			decision.Plugins = append(decision.Plugins, *dp)
		}
	}

	c.compileRouteEmits(r, &decision)
	return decision
}

func (c *Compiler) compileRouteRules(r *RouteDecl) config.RuleCombination {
	if r.When == nil {
		return config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{}}
	}
	rules := c.compileBoolExpr(r.When)
	if rules.Operator == "" && rules.Type != "" {
		return config.RuleCombination{
			Operator:   "AND",
			Conditions: []config.RuleNode{rules},
		}
	}
	return rules
}

func (c *Compiler) compileRouteEmits(r *RouteDecl, decision *config.Decision) {
	seenEmitKinds := make(map[string]bool, len(r.Emits))
	for _, e := range r.Emits {
		if e == nil {
			continue
		}
		if seenEmitKinds[e.Kind] {
			c.addError(e.Pos, "ROUTE %s EMIT: duplicate EMIT kind %q in the same route. Each kind may appear at most once", r.Name, e.Kind)
			continue
		}
		seenEmitKinds[e.Kind] = true
		if !c.validateEmitForCompile(r, e) {
			continue
		}
		decision.Emits = append(decision.Emits, compileEmitDecl(e))
	}
}

func (c *Compiler) appendModelRef(decision *config.Decision, m *ModelRef) {
	if decision == nil || m == nil {
		return
	}
	ref := config.ModelRef{
		Model:    m.Model,
		LoRAName: m.LoRA,
		Weight:   m.Weight,
	}
	if m.Reasoning != nil {
		ref.UseReasoning = m.Reasoning
	}
	if m.Effort != "" {
		ref.ReasoningEffort = m.Effort
	}
	decision.ModelRefs = append(decision.ModelRefs, ref)

	// Populate model_config for route-local model metadata fields.
	if m.ParamSize != "" {
		if c.config.ModelConfig == nil {
			c.config.ModelConfig = make(map[string]config.ModelParams)
		}
		mc := c.config.ModelConfig[m.Model]
		mc.ParamSize = m.ParamSize
		c.config.ModelConfig[m.Model] = mc
	}
}

func (c *Compiler) appendModelRefIfMissing(decision *config.Decision, m *ModelRef) {
	if decision == nil || m == nil {
		return
	}
	for _, existing := range decision.ModelRefs {
		if existing.Model == m.Model && existing.LoRAName == m.LoRA {
			return
		}
	}
	c.appendModelRef(decision, m)
}

func (c *Compiler) compileCandidateIteration(iter *CandidateIterationDecl) config.CandidateIterationConfig {
	if iter == nil {
		return config.CandidateIterationConfig{}
	}
	compiled := config.CandidateIterationConfig{
		Variable: iter.Variable,
		Source:   iter.Source,
	}
	for _, model := range iter.Models {
		ref := config.ModelRef{
			Model:    model.Model,
			LoRAName: model.LoRA,
			Weight:   model.Weight,
		}
		if model.Reasoning != nil {
			ref.UseReasoning = model.Reasoning
		}
		if model.Effort != "" {
			ref.ReasoningEffort = model.Effort
		}
		compiled.Models = append(compiled.Models, ref)
	}
	for _, output := range iter.Outputs {
		compiled.Outputs = append(compiled.Outputs, config.CandidateIterationOutputConfig{
			Type:  output.Type,
			Value: output.Value,
		})
	}
	return compiled
}

func (c *Compiler) compileBoolExpr(expr BoolExpr) config.RuleCombination {
	switch e := expr.(type) {
	case *BoolAnd:
		// Flatten nested ANDs: (a AND b) AND c → AND(a, b, c)
		conditions := c.flattenBoolExpr(expr, func(ex BoolExpr) bool {
			_, ok := ex.(*BoolAnd)
			return ok
		}, func(ex BoolExpr) (BoolExpr, BoolExpr) {
			a := ex.(*BoolAnd)
			return a.Left, a.Right
		})
		return config.RuleCombination{
			Operator:   "AND",
			Conditions: conditions,
		}
	case *BoolOr:
		// Flatten nested ORs: (a OR b) OR c → OR(a, b, c)
		conditions := c.flattenBoolExpr(expr, func(ex BoolExpr) bool {
			_, ok := ex.(*BoolOr)
			return ok
		}, func(ex BoolExpr) (BoolExpr, BoolExpr) {
			o := ex.(*BoolOr)
			return o.Left, o.Right
		})
		return config.RuleCombination{
			Operator:   "OR",
			Conditions: conditions,
		}
	case *BoolNot:
		return config.RuleCombination{
			Operator: "NOT",
			Conditions: []config.RuleNode{
				c.compileBoolExpr(e.Expr),
			},
		}
	case *SignalRefExpr:
		return config.RuleCombination{
			Type: e.SignalType,
			Name: e.SignalName,
		}
	default:
		c.addError(Position{}, "unknown bool expression type %T", expr)
		return config.RuleCombination{}
	}
}

func (c *Compiler) flattenBoolExpr(
	expr BoolExpr,
	isSameType func(BoolExpr) bool,
	getChildren func(BoolExpr) (BoolExpr, BoolExpr),
) []config.RuleNode {
	if isSameType(expr) {
		left, right := getChildren(expr)
		var result []config.RuleNode
		result = append(result, c.flattenBoolExpr(left, isSameType, getChildren)...)
		result = append(result, c.flattenBoolExpr(right, isSameType, getChildren)...)
		return result
	}
	return []config.RuleNode{c.compileBoolExpr(expr)}
}

func compileComposerObj(ov ObjectValue) config.RuleCombination {
	rc := config.RuleCombination{}
	if t, ok := getStringField(ov.Fields, "type"); ok {
		rc.Type = t
	}
	if n, ok := getStringField(ov.Fields, "name"); ok {
		rc.Name = n
	}
	if op, ok := getStringField(ov.Fields, "operator"); ok {
		rc.Operator = op
	}
	if arr, ok := ov.Fields["conditions"]; ok {
		if av, ok := arr.(ArrayValue); ok {
			for _, item := range av.Items {
				if obj, ok := item.(ObjectValue); ok {
					rc.Conditions = append(rc.Conditions, compileComposerObj(obj))
				}
			}
		}
	}
	return rc
}
