package dsl

import (
	"fmt"
	"sort"
)

func applyRawRouting(prog *Program, raw *rawRoutingDecl) []error {
	fields := entriesToMap(raw.Fields)
	strategy, ok := getStringField(fields, "strategy")
	if !ok {
		return []error{fmt.Errorf("%s: ROUTING requires strategy", posFromLexer(raw.Pos))}
	}
	prog.Strategy = strategy
	return nil
}

func rawToEntrypoint(raw *rawEntrypointDecl) (*EntrypointDecl, []error) {
	decl := &EntrypointDecl{Pos: posFromLexer(raw.Pos)}
	fields := entriesToMap(raw.Fields)
	if unknown := unknownEntrypointObjectField(fields, "name", "aliases", "recipe", "assignments", "rules"); unknown != "" {
		return decl, []error{fmt.Errorf("%s: ENTRYPOINT contains unknown field %q", decl.Pos, unknown)}
	}
	if value, ok := getStringField(fields, "name"); ok {
		decl.Name = value
	}
	if aliases, ok := getStringArrayField(fields, "aliases"); ok {
		decl.Aliases = aliases
	} else if _, exists := fields["aliases"]; exists {
		return decl, []error{fmt.Errorf("%s: ENTRYPOINT aliases must be an array of strings", decl.Pos)}
	}
	decl.Recipe, _ = getStringField(fields, "recipe")
	var errs []error
	if rawAssignments, exists := fields["assignments"]; exists {
		assignments, err := parseEntrypointAssignments(rawAssignments, decl.Pos)
		if err != nil {
			errs = append(errs, fmt.Errorf("ENTRYPOINT assignments: %w", err))
		} else {
			decl.Assignments = assignments
		}
	}
	if rules, ok := fields["rules"].(ArrayValue); ok {
		for index, item := range rules.Items {
			rule, err := parseEntrypointRule(item, decl.Pos)
			if err != nil {
				errs = append(errs, fmt.Errorf("ENTRYPOINT rules[%d]: %w", index, err))
				continue
			}
			decl.Rules = append(decl.Rules, rule)
		}
	} else if _, exists := fields["rules"]; exists {
		errs = append(errs, fmt.Errorf("%s: ENTRYPOINT rules must be an array", decl.Pos))
	}
	if decl.Name == "" {
		errs = append(errs, fmt.Errorf("%s: ENTRYPOINT requires name", decl.Pos))
	}
	if len(decl.Rules) > 0 {
		if decl.Recipe != "" || decl.Assignments != nil {
			errs = append(errs, fmt.Errorf("%s: ENTRYPOINT cannot mix recipe/assignments with rules", decl.Pos))
		}
	} else {
		if decl.Recipe == "" {
			errs = append(errs, fmt.Errorf("%s: ENTRYPOINT requires recipe", decl.Pos))
		}
		if len(decl.Assignments) == 0 {
			errs = append(errs, fmt.Errorf("%s: ENTRYPOINT requires non-empty assignments", decl.Pos))
		}
	}
	return decl, errs
}

func parseEntrypointRule(value Value, pos Position) (*EntrypointRuleDecl, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("%s: rule must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "name", "matches", "recipe", "assignments"); unknown != "" {
		return nil, fmt.Errorf("%s: rule contains unknown field %q", pos, unknown)
	}
	rule := &EntrypointRuleDecl{Pos: pos}
	rule.Name, _ = getStringField(object.Fields, "name")
	rule.Recipe, _ = getStringField(object.Fields, "recipe")
	if rule.Name == "" || rule.Recipe == "" {
		return nil, fmt.Errorf("%s: rule requires name and recipe", pos)
	}
	if rawMatches, exists := object.Fields["matches"]; exists {
		matches, ok := rawMatches.(ArrayValue)
		if !ok {
			return nil, fmt.Errorf("%s: rule matches must be an array", pos)
		}
		for index, rawMatch := range matches.Items {
			match, err := parseEntrypointMatch(rawMatch, pos)
			if err != nil {
				return nil, fmt.Errorf("matches[%d]: %w", index, err)
			}
			rule.Matches = append(rule.Matches, match)
		}
	}
	rawAssignments, exists := object.Fields["assignments"]
	if !exists {
		return nil, fmt.Errorf("%s: rule requires assignments", pos)
	}
	assignments, err := parseEntrypointAssignments(rawAssignments, pos)
	if err != nil {
		return nil, err
	}
	rule.Assignments = assignments
	return rule, nil
}

func parseEntrypointMatch(value Value, pos Position) (*EntrypointMatchDecl, error) {
	object, isObject := value.(ObjectValue)
	if !isObject {
		return nil, fmt.Errorf("%s: match must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "claim", "path"); unknown != "" {
		return nil, fmt.Errorf("%s: match contains unknown field %q", pos, unknown)
	}
	claimValue, hasClaim := object.Fields["claim"]
	pathValue, hasPath := object.Fields["path"]
	if hasClaim == hasPath {
		return nil, fmt.Errorf("%s: match requires exactly one of claim or path", pos)
	}
	if hasClaim {
		claim, ok := claimValue.(ObjectValue)
		if !ok {
			return nil, fmt.Errorf("%s: claim match must be an object", pos)
		}
		if unknown := unknownEntrypointObjectField(claim.Fields, "name", "exact"); unknown != "" {
			return nil, fmt.Errorf("%s: claim match contains unknown field %q", pos, unknown)
		}
		name, _ := getStringField(claim.Fields, "name")
		exact, exists := claim.Fields["exact"]
		if name == "" || !exists {
			return nil, fmt.Errorf("%s: claim match requires name and exact", pos)
		}
		switch exact.(type) {
		case StringValue, BoolValue, IntValue:
		default:
			return nil, fmt.Errorf("%s: claim exact must be a string, boolean, or integer", pos)
		}
		return &EntrypointMatchDecl{Claim: &EntrypointClaimMatchDecl{Name: name, Exact: exact}}, nil
	}
	path, ok := pathValue.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("%s: path match must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(path.Fields, "exact", "prefix"); unknown != "" {
		return nil, fmt.Errorf("%s: path match contains unknown field %q", pos, unknown)
	}
	exact, hasExact := getStringField(path.Fields, "exact")
	prefix, hasPrefix := getStringField(path.Fields, "prefix")
	if hasExact == hasPrefix || (exact == "" && prefix == "") {
		return nil, fmt.Errorf("%s: path match requires exactly one non-empty exact or prefix", pos)
	}
	return &EntrypointMatchDecl{Path: &EntrypointPathMatchDecl{Exact: exact, Prefix: prefix}}, nil
}

func parseEntrypointAssignments(value Value, pos Position) (map[string]*EntrypointAssignmentSetDecl, error) {
	rawAssignments, ok := value.(ArrayValue)
	if !ok || len(rawAssignments.Items) == 0 {
		return nil, fmt.Errorf("%s: assignments must be a non-empty array", pos)
	}
	assignments := make(map[string]*EntrypointAssignmentSetDecl, len(rawAssignments.Items))
	for index, rawAssignment := range rawAssignments.Items {
		decisionName, assignmentSet, err := parseEntrypointDecisionAssignment(rawAssignment, pos)
		if err != nil {
			return nil, fmt.Errorf("assignments[%d]: %w", index, err)
		}
		if _, duplicate := assignments[decisionName]; duplicate {
			return nil, fmt.Errorf("%s: duplicate assignment for decision %q", pos, decisionName)
		}
		assignments[decisionName] = assignmentSet
	}
	return assignments, nil
}

func parseEntrypointDecisionAssignment(value Value, pos Position) (string, *EntrypointAssignmentSetDecl, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return "", nil, fmt.Errorf("%s: assignment must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "decision", "models", "fallback"); unknown != "" {
		return "", nil, fmt.Errorf("%s: assignment contains unknown field %q", pos, unknown)
	}
	decisionName, _ := getStringField(object.Fields, "decision")
	models, ok := object.Fields["models"].(ArrayValue)
	if decisionName == "" || !ok || len(models.Items) == 0 {
		return "", nil, fmt.Errorf("%s: assignment requires decision and non-empty models", pos)
	}
	result := &EntrypointAssignmentSetDecl{Models: make([]*EntrypointAssignmentDecl, 0, len(models.Items))}
	for index, modelValue := range models.Items {
		model, err := parseEntrypointAssignment(modelValue, pos)
		if err != nil {
			return "", nil, fmt.Errorf("models[%d]: %w", index, err)
		}
		result.Models = append(result.Models, model)
	}
	if rawFallback, exists := object.Fields["fallback"]; exists {
		fallback, err := parseEntrypointFallback(rawFallback, pos)
		if err != nil {
			return "", nil, err
		}
		result.Fallback = fallback
	}
	return decisionName, result, nil
}

func parseEntrypointFallback(value Value, pos Position) (*EntrypointFallbackDecl, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("%s: fallback must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "strategy", "on"); unknown != "" {
		return nil, fmt.Errorf("%s: fallback contains unknown field %q", pos, unknown)
	}
	strategy, _ := getStringField(object.Fields, "strategy")
	on, ok := getStringArrayField(object.Fields, "on")
	if strategy == "" || !ok || len(on) == 0 {
		return nil, fmt.Errorf("%s: fallback requires strategy and non-empty on", pos)
	}
	return &EntrypointFallbackDecl{Strategy: strategy, On: on}, nil
}

func parseEntrypointAssignment(value Value, pos Position) (*EntrypointAssignmentDecl, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("%s: model assignment must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "model", "priority", "weight", "lora", "reasoning"); unknown != "" {
		return nil, fmt.Errorf("%s: model assignment contains unknown field %q", pos, unknown)
	}
	modelName, _ := getStringField(object.Fields, "model")
	if modelName == "" {
		return nil, fmt.Errorf("%s: model assignment requires model", pos)
	}
	assignment := &EntrypointAssignmentDecl{Model: modelName, Pos: pos}
	if priority, ok := getIntField(object.Fields, "priority"); ok {
		assignment.Priority = priority
	} else if _, exists := object.Fields["priority"]; exists {
		return nil, fmt.Errorf("%s: model assignment priority must be an integer", pos)
	}
	if weight, ok := getStringField(object.Fields, "weight"); ok {
		assignment.Weight = weight
	} else if _, exists := object.Fields["weight"]; exists {
		return nil, fmt.Errorf("%s: model assignment weight must be a decimal string", pos)
	}
	assignment.LoRAName, _ = getStringField(object.Fields, "lora")
	if rawReasoning, exists := object.Fields["reasoning"]; exists {
		reasoning, err := parseEntrypointReasoning(rawReasoning, pos)
		if err != nil {
			return nil, err
		}
		assignment.Reasoning = reasoning
	}
	return assignment, nil
}

func parseEntrypointReasoning(value Value, pos Position) (*EntrypointReasoningDecl, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("%s: reasoning must be an object", pos)
	}
	if unknown := unknownEntrypointObjectField(object.Fields, "enabled", "effort", "description"); unknown != "" {
		return nil, fmt.Errorf("%s: reasoning contains unknown field %q", pos, unknown)
	}
	enabled, ok := getBoolField(object.Fields, "enabled")
	if !ok {
		return nil, fmt.Errorf("%s: reasoning requires boolean enabled", pos)
	}
	reasoning := &EntrypointReasoningDecl{Enabled: enabled}
	reasoning.Effort, _ = getStringField(object.Fields, "effort")
	reasoning.Description, _ = getStringField(object.Fields, "description")
	return reasoning, nil
}

func unknownEntrypointObjectField(fields map[string]Value, allowed ...string) string {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, field := range allowed {
		allowedSet[field] = struct{}{}
	}
	unknown := make([]string, 0)
	for field := range fields {
		if _, ok := allowedSet[field]; !ok {
			unknown = append(unknown, field)
		}
	}
	if len(unknown) == 0 {
		return ""
	}
	sort.Strings(unknown)
	return unknown[0]
}

func rawToRecipe(raw *rawRecipeDecl) (*RecipeDecl, []error) {
	decl := &RecipeDecl{
		Name:    unquoteIdent(raw.Name),
		Program: &Program{},
		Pos:     posFromLexer(raw.Pos),
	}
	optionErrs := applyRawRecipeOptions(decl, raw.Opts)

	state := recipeParseState{decl: decl}
	state.errs = append(state.errs, optionErrs...)
	for _, entry := range raw.Body {
		state.appendEntry(entry)
	}
	if state.hasDirectRoutes && state.treeCount > 0 {
		state.errs = append(state.errs, fmt.Errorf("RECIPE %q: DECISION_TREE and ROUTE declarations cannot coexist", decl.Name))
	}
	return decl, state.errs
}

func applyRawRecipeOptions(decl *RecipeDecl, opts []*RouteOpt) []error {
	var errs []error
	for _, opt := range opts {
		switch opt.Key {
		case "description":
			if opt.Value != nil && opt.Value.Str != nil {
				decl.Description = unquote(*opt.Value.Str)
			}
		case "strategy":
			if opt.Value != nil {
				if value, ok := valToValue(opt.Value).(StringValue); ok {
					decl.Program.Strategy = value.V
				}
			}
		default:
			errs = append(errs, fmt.Errorf("%s: RECIPE contains unknown option %q", posFromLexer(opt.Pos), opt.Key))
		}
	}
	return errs
}

type recipeParseState struct {
	decl            *RecipeDecl
	errs            []error
	hasDirectRoutes bool
	treeCount       int
}

func (state *recipeParseState) appendEntry(entry *rawRecipeEntry) {
	prog := state.decl.Program
	switch {
	case entry.Routing != nil:
		state.errs = append(state.errs, applyRawRouting(prog, entry.Routing)...)
	case entry.Signal != nil:
		prog.Signals = append(prog.Signals, rawToSignal(entry.Signal))
	case entry.Projection != nil:
		appendRawProjection(prog, entry.Projection)
	case entry.Route != nil:
		state.hasDirectRoutes = true
		route, errs := rawToRoute(entry.Route)
		prog.Routes = append(prog.Routes, route)
		state.errs = append(state.errs, errs...)
	case entry.DecisionTree != nil:
		routes, errs := rawDecisionTreeToRoutes(entry.DecisionTree, state.treeCount)
		state.treeCount++
		prog.Routes = append(prog.Routes, routes...)
		state.errs = append(state.errs, errs...)
	case entry.Plugin != nil:
		prog.Plugins = append(prog.Plugins, rawToPlugin(entry.Plugin))
	case entry.TestBlock != nil:
		prog.TestBlocks = append(prog.TestBlocks, rawToTestBlock(entry.TestBlock))
	}
}

func appendRawProjection(prog *Program, raw *rawProjectionDecl) {
	switch raw.Kind {
	case "partition":
		prog.ProjectionPartitions = append(prog.ProjectionPartitions, rawToProjectionPartition(raw))
	case "score":
		prog.ProjectionScores = append(prog.ProjectionScores, rawToProjectionScore(raw))
	case "mapping":
		prog.ProjectionMappings = append(prog.ProjectionMappings, rawToProjectionMapping(raw))
	}
}
