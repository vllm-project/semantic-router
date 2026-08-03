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
	applyRawRecipeOptions(decl, raw.Opts)

	state := recipeParseState{decl: decl}
	for _, entry := range raw.Body {
		state.appendEntry(entry)
	}
	if state.hasDirectRoutes && state.treeCount > 0 {
		state.errs = append(state.errs, fmt.Errorf("RECIPE %q: DECISION_TREE and ROUTE declarations cannot coexist", decl.Name))
	}
	return decl, state.errs
}

func applyRawRecipeOptions(decl *RecipeDecl, opts []*RouteOpt) {
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
		}
	}
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
		prog.Routes = append(prog.Routes, rawToRoute(entry.Route))
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
