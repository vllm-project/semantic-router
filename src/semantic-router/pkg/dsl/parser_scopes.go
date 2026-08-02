package dsl

import "fmt"

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
	if recipe, ok := getStringField(fields, "recipe"); ok {
		decl.Recipe = recipe
	}
	if modelNames, ok := fields["model_names"].(ArrayValue); ok {
		for _, item := range modelNames.Items {
			if name, ok := item.(StringValue); ok {
				decl.ModelNames = append(decl.ModelNames, name.V)
			}
		}
	}
	var errs []error
	if len(decl.ModelNames) == 0 {
		errs = append(errs, fmt.Errorf("%s: ENTRYPOINT requires a non-empty model_names array", decl.Pos))
	}
	if decl.Recipe == "" {
		errs = append(errs, fmt.Errorf("%s: ENTRYPOINT requires recipe", decl.Pos))
	}
	return decl, errs
}

func rawToRecipe(raw *rawRecipeDecl) (*RecipeDecl, []error) {
	decl := &RecipeDecl{
		Name:    unquoteIdent(raw.Name),
		Program: &Program{},
		Pos:     posFromLexer(raw.Pos),
	}
	for _, opt := range raw.Opts {
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
		}
	}

	var errs []error
	hasDirectRoutes := false
	treeCount := 0
	for _, entry := range raw.Body {
		switch {
		case entry.Routing != nil:
			errs = append(errs, applyRawRouting(decl.Program, entry.Routing)...)
		case entry.Signal != nil:
			decl.Program.Signals = append(decl.Program.Signals, rawToSignal(entry.Signal))
		case entry.Projection != nil:
			appendRawProjection(decl.Program, entry.Projection)
		case entry.Route != nil:
			hasDirectRoutes = true
			decl.Program.Routes = append(decl.Program.Routes, rawToRoute(entry.Route))
		case entry.DecisionTree != nil:
			treeCount++
			routes, treeErrs := rawDecisionTreeToRoutes(entry.DecisionTree, treeCount-1)
			decl.Program.Routes = append(decl.Program.Routes, routes...)
			errs = append(errs, treeErrs...)
		case entry.Plugin != nil:
			decl.Program.Plugins = append(decl.Program.Plugins, rawToPlugin(entry.Plugin))
		case entry.TestBlock != nil:
			decl.Program.TestBlocks = append(decl.Program.TestBlocks, rawToTestBlock(entry.TestBlock))
		}
	}
	if hasDirectRoutes && treeCount > 0 {
		errs = append(errs, fmt.Errorf("RECIPE %q: DECISION_TREE and ROUTE declarations cannot coexist", decl.Name))
	}
	return decl, errs
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
