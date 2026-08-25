package dsl

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/alecthomas/participle/v2"
	"github.com/alecthomas/participle/v2/lexer"
)

// dslLexer defines the lexical rules for the DSL.
var dslLexer = lexer.MustSimple([]lexer.SimpleRule{
	{Name: "Comment", Pattern: `#[^\n]*`},
	{Name: "Whitespace", Pattern: `[\s]+`},
	{Name: "Float", Pattern: `[+-]?[0-9]+\.[0-9]+`},
	{Name: "Int", Pattern: `[+-]?[0-9]+`},
	{Name: "String", Pattern: `"(?:[^"\\]|\\.)*"`},
	{Name: "Arrow", Pattern: `->|→`},
	{Name: "Ident", Pattern: `[a-zA-Z_][a-zA-Z0-9_\-\.\/]*`},
	{Name: "LBrace", Pattern: `\{`},
	{Name: "RBrace", Pattern: `\}`},
	{Name: "LParen", Pattern: `\(`},
	{Name: "RParen", Pattern: `\)`},
	{Name: "LBracket", Pattern: `\[`},
	{Name: "RBracket", Pattern: `\]`},
	{Name: "Colon", Pattern: `:`},
	{Name: "Comma", Pattern: `,`},
	{Name: "GreaterThan", Pattern: `>`},
	{Name: "Equals", Pattern: `=`},
})

// rawParser is the participle parser for the DSL.
var rawParser = participle.MustBuild[rawProgram](
	participle.Lexer(dslLexer),
	participle.Elide("Comment", "Whitespace"),
	participle.UseLookahead(3),
)

// Parse tokenizes and parses a DSL source string into a Program AST.
// If the input has syntax errors, Parse attempts error recovery by
// splitting the input into top-level blocks and parsing each independently.
func Parse(input string) (*Program, []error) {
	raw, err := rawParser.ParseString("", input)
	if err == nil {
		return rawToProgram(raw)
	}

	// Error recovery: split input into top-level blocks and parse each
	blocks := splitTopLevelBlocks(input)
	if len(blocks) <= 1 {
		return nil, []error{err}
	}

	prog := &Program{}
	var allErrors []error
	parsedAny := false
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		r, e := rawParser.ParseString("", block)
		if e != nil {
			allErrors = append(allErrors, e)
			continue
		}
		parsedAny = true
		resolved, lowerErrs := rawToProgram(r)
		mergeProgram(prog, resolved)
		allErrors = append(allErrors, lowerErrs...)
	}

	if !parsedAny {
		return nil, []error{err}
	}
	return prog, allErrors
}

// splitTopLevelBlocks splits DSL source into top-level blocks by finding
// top-level keywords (SIGNAL, ROUTE, MODEL, PLUGIN) that appear outside of
// braces.
func splitTopLevelBlocks(input string) []string {
	var blocks []string
	depth := 0
	start := 0
	keywords := []string{
		"DECISION_TREE", "ENTRYPOINT", "PROJECTION", "ROUTING", "RECIPE",
		"SIGNAL", "ROUTE", "MODEL", "PLUGIN", "TEST",
	}

	for i := 0; i < len(input); i++ {
		ch := input[i]
		if ch == '"' {
			// skip string literals
			i++
			for i < len(input) && input[i] != '"' {
				if input[i] == '\\' {
					i++
				}
				i++
			}
			continue
		}
		if ch == '#' {
			// skip comments
			for i < len(input) && input[i] != '\n' {
				i++
			}
			continue
		}
		if ch == '{' {
			depth++
			continue
		}
		if ch == '}' {
			depth--
			continue
		}
		if depth == 0 {
			for _, kw := range keywords {
				if i+len(kw) <= len(input) && input[i:i+len(kw)] == kw {
					// Check that it's at a word boundary
					if i > 0 && isIdentPart(rune(input[i-1])) {
						continue
					}
					if i+len(kw) < len(input) && isIdentPart(rune(input[i+len(kw)])) {
						continue
					}
					// Found a top-level keyword — split here
					if i > start {
						blocks = append(blocks, input[start:i])
					}
					start = i
					break
				}
			}
		}
	}
	if start < len(input) {
		blocks = append(blocks, input[start:])
	}
	return blocks
}

// ---------- Raw → Resolved Conversion ----------

func rawToProgram(raw *rawProgram) (*Program, []error) {
	prog := &Program{}
	var errs []error
	hasDirectRoutes := false
	treeCount := 0
	for _, entry := range raw.Entries {
		switch {
		case entry.Routing != nil:
			errs = append(errs, applyRawRouting(prog, entry.Routing)...)
		case entry.Entrypoint != nil:
			decl, entryErrs := rawToEntrypoint(entry.Entrypoint)
			prog.Entrypoints = append(prog.Entrypoints, decl)
			errs = append(errs, entryErrs...)
		case entry.Recipe != nil:
			decl, recipeErrs := rawToRecipe(entry.Recipe)
			prog.Recipes = append(prog.Recipes, decl)
			errs = append(errs, recipeErrs...)
		case entry.Signal != nil:
			prog.Signals = append(prog.Signals, rawToSignal(entry.Signal))
		case entry.Projection != nil:
			switch entry.Projection.Kind {
			case "partition":
				prog.ProjectionPartitions = append(prog.ProjectionPartitions, rawToProjectionPartition(entry.Projection))
			case "score":
				prog.ProjectionScores = append(prog.ProjectionScores, rawToProjectionScore(entry.Projection))
			case "mapping":
				prog.ProjectionMappings = append(prog.ProjectionMappings, rawToProjectionMapping(entry.Projection))
			}
		case entry.Route != nil:
			hasDirectRoutes = true
			route, routeErrs := rawToRoute(entry.Route)
			prog.Routes = append(prog.Routes, route)
			errs = append(errs, routeErrs...)
		case entry.DecisionTree != nil:
			treeCount++
			routes, treeErrs := rawDecisionTreeToRoutes(entry.DecisionTree, treeCount-1)
			prog.Routes = append(prog.Routes, routes...)
			errs = append(errs, treeErrs...)
		case entry.Model != nil:
			model, modelErrs := rawToModelDecl(entry.Model)
			prog.Models = append(prog.Models, model)
			errs = append(errs, modelErrs...)
		case entry.Plugin != nil:
			prog.Plugins = append(prog.Plugins, rawToPlugin(entry.Plugin))
		case entry.TestBlock != nil:
			prog.TestBlocks = append(prog.TestBlocks, rawToTestBlock(entry.TestBlock))
		}
	}
	if hasDirectRoutes && treeCount > 0 {
		errs = append(errs, fmt.Errorf("DECISION_TREE and ROUTE declarations cannot coexist in the same program. Use only DECISION_TREE (for if/else conditional logic) or only ROUTE (for priority-based routing with WHEN clauses), not both"))
	}
	return prog, errs
}

func mergeProgram(dst, src *Program) {
	if src.Strategy != "" {
		dst.Strategy = src.Strategy
	}
	dst.Entrypoints = append(dst.Entrypoints, src.Entrypoints...)
	dst.Recipes = append(dst.Recipes, src.Recipes...)
	dst.Signals = append(dst.Signals, src.Signals...)
	dst.ProjectionPartitions = append(dst.ProjectionPartitions, src.ProjectionPartitions...)
	dst.ProjectionScores = append(dst.ProjectionScores, src.ProjectionScores...)
	dst.ProjectionMappings = append(dst.ProjectionMappings, src.ProjectionMappings...)
	dst.Routes = append(dst.Routes, src.Routes...)
	dst.Models = append(dst.Models, src.Models...)
	dst.Plugins = append(dst.Plugins, src.Plugins...)
	dst.TestBlocks = append(dst.TestBlocks, src.TestBlocks...)
}

func rawToProjectionPartition(r *rawProjectionDecl) *ProjectionPartitionDecl {
	partition := &ProjectionPartitionDecl{
		Name: unquoteIdent(r.Name),
		Pos:  posFromLexer(r.Pos),
	}
	fields := entriesToMap(r.Fields)
	if v, ok := fields["semantics"]; ok {
		if sv, ok := v.(StringValue); ok {
			partition.Semantics = sv.V
		}
	}
	if v, ok := fields["temperature"]; ok {
		switch tv := v.(type) {
		case FloatValue:
			partition.Temperature = tv.V
		case IntValue:
			partition.Temperature = float64(tv.V)
		}
	}
	if v, ok := fields["members"]; ok {
		if av, ok := v.(ArrayValue); ok {
			for _, item := range av.Items {
				if sv, ok := item.(StringValue); ok {
					partition.Members = append(partition.Members, sv.V)
				}
			}
		}
	}
	if v, ok := fields["default"]; ok {
		if sv, ok := v.(StringValue); ok {
			partition.Default = sv.V
		}
	}
	return partition
}

func rawToProjectionScore(r *rawProjectionDecl) *ProjectionScoreDecl {
	score := &ProjectionScoreDecl{
		Name:   unquoteIdent(r.Name),
		Pos:    posFromLexer(r.Pos),
		Method: "weighted_sum",
	}
	fields := entriesToMap(r.Fields)
	if method, ok := getStringField(fields, "method"); ok {
		score.Method = method
	}
	if rawInputs, ok := fields["inputs"].(ArrayValue); ok {
		for _, item := range rawInputs.Items {
			ov, ok := item.(ObjectValue)
			if !ok {
				continue
			}
			input := &ProjectionScoreInputDecl{}
			if signalType, ok := getStringField(ov.Fields, "type"); ok {
				input.SignalType = signalType
			}
			if signalName, ok := getStringField(ov.Fields, "name"); ok {
				input.SignalName = signalName
			}
			if kb, ok := getStringField(ov.Fields, "kb"); ok {
				input.KB = kb
			}
			if metric, ok := getStringField(ov.Fields, "metric"); ok {
				input.Metric = metric
			}
			if weight, ok := getFloat64Field(ov.Fields, "weight"); ok {
				input.Weight = weight
			}
			if valueSource, ok := getStringField(ov.Fields, "value_source"); ok {
				input.ValueSource = valueSource
			}
			if match, ok := getFloat64Field(ov.Fields, "match"); ok {
				input.Match = match
			}
			if miss, ok := getFloat64Field(ov.Fields, "miss"); ok {
				input.Miss = miss
			}
			score.Inputs = append(score.Inputs, input)
		}
	}
	return score
}

func rawToProjectionMapping(r *rawProjectionDecl) *ProjectionMappingDecl {
	mapping := &ProjectionMappingDecl{
		Name:   unquoteIdent(r.Name),
		Pos:    posFromLexer(r.Pos),
		Method: "threshold_bands",
	}
	fields := entriesToMap(r.Fields)
	if source, ok := getStringField(fields, "source"); ok {
		mapping.Source = source
	}
	if method, ok := getStringField(fields, "method"); ok {
		mapping.Method = method
	}
	if calibrationObj, ok := fields["calibration"].(ObjectValue); ok {
		calibration := &ProjectionMappingCalibrationDecl{}
		if method, ok := getStringField(calibrationObj.Fields, "method"); ok {
			calibration.Method = method
		}
		if slope, ok := getFloat64Field(calibrationObj.Fields, "slope"); ok {
			calibration.Slope = slope
		}
		mapping.Calibration = calibration
	}
	if rawOutputs, ok := fields["outputs"].(ArrayValue); ok {
		for _, item := range rawOutputs.Items {
			ov, ok := item.(ObjectValue)
			if !ok {
				continue
			}
			output := &ProjectionMappingOutputDecl{}
			if name, ok := getStringField(ov.Fields, "name"); ok {
				output.Name = name
			}
			if v, ok := getFloat64Field(ov.Fields, "lt"); ok {
				output.LT = float64Ptr(v)
			}
			if v, ok := getFloat64Field(ov.Fields, "lte"); ok {
				output.LTE = float64Ptr(v)
			}
			if v, ok := getFloat64Field(ov.Fields, "gt"); ok {
				output.GT = float64Ptr(v)
			}
			if v, ok := getFloat64Field(ov.Fields, "gte"); ok {
				output.GTE = float64Ptr(v)
			}
			mapping.Outputs = append(mapping.Outputs, output)
		}
	}
	return mapping
}

func float64Ptr(v float64) *float64 {
	return &v
}

func rawToTestBlock(r *rawTestBlockDecl) *TestBlockDecl {
	tb := &TestBlockDecl{
		Name: unquoteIdent(r.Name),
		Pos:  posFromLexer(r.Pos),
	}
	for _, entry := range r.Entries {
		tb.Entries = append(tb.Entries, &TestEntry{
			Query:     unquote(entry.Query),
			RouteName: unquoteIdent(entry.RouteName),
			Pos:       posFromLexer(entry.Pos),
		})
	}
	return tb
}

func rawToSignal(r *rawSignalDecl) *SignalDecl {
	return &SignalDecl{
		SignalType: r.SignalType,
		Name:       unquoteIdent(r.Name),
		Fields:     entriesToMap(r.Fields),
		Pos:        posFromLexer(r.Pos),
	}
}

func rawToRoute(r *rawRouteDecl) (*RouteDecl, []error) {
	route := &RouteDecl{
		Name: unquoteIdent(r.Name),
		Pos:  posFromLexer(r.Pos),
	}
	var errs []error

	// Process options
	for _, opt := range r.Opts {
		switch opt.Key {
		case "description":
			if opt.Value != nil && opt.Value.Str != nil {
				route.Description = unquote(*opt.Value.Str)
			}
		default:
			errs = append(errs, fmt.Errorf("%s: ROUTE contains unknown option %q", posFromLexer(opt.Pos), opt.Key))
		}
	}

	// Process body items
	for _, item := range r.Body {
		switch {
		case item.Priority != nil:
			route.Priority = *item.Priority
		case item.Tier != nil:
			route.Tier = *item.Tier
		case item.When != nil:
			route.When = toBoolExpr(item.When)
		case item.Model != nil:
			for _, m := range item.Model.Models {
				route.Models = append(route.Models, rawToModelRef(m))
			}
		case item.Algorithm != nil:
			route.Algorithm = rawToAlgo(item.Algorithm)
		case item.Plugin != nil:
			route.Plugins = append(route.Plugins, rawToPluginRef(item.Plugin))
		case item.Description != nil:
			route.Description = unquote(*item.Description)
		case item.CandidateFor != nil:
			route.CandidateIterations = append(route.CandidateIterations, rawToCandidateIteration(item.CandidateFor))
		case item.Emit != nil:
			route.Emits = append(route.Emits, rawToEmitDecl(item.Emit))
		}
	}

	return route, errs
}

func rawToModelDecl(r *rawModelDecl) (*ModelDecl, []error) {
	decl := &ModelDecl{
		Name:   unquoteIdent(r.Name),
		Fields: entriesToMap(r.Fields),
		Pos:    posFromLexer(r.Pos),
	}
	allowed := []string{
		"aliases", "param_size", "context_window_size", "description",
		"capabilities", "reasoning", "loras", "quality_score", "modality", "tags",
	}
	if unknown := unknownEntrypointObjectField(decl.Fields, allowed...); unknown != "" {
		return decl, []error{fmt.Errorf("%s: MODEL contains unknown ModelCard field %q", decl.Pos, unknown)}
	}
	if rawReasoning, exists := decl.Fields["reasoning"]; exists {
		reasoning, ok := rawReasoning.(ObjectValue)
		if !ok {
			return decl, []error{fmt.Errorf("%s: MODEL reasoning must be an object", decl.Pos)}
		}
		if unknown := unknownEntrypointObjectField(reasoning.Fields, "type", "efforts"); unknown != "" {
			return decl, []error{fmt.Errorf("%s: MODEL reasoning contains unknown field %q", decl.Pos, unknown)}
		}
		if _, ok := getStringField(reasoning.Fields, "type"); !ok {
			return decl, []error{fmt.Errorf("%s: MODEL reasoning requires a string type", decl.Pos)}
		}
		if _, exists := reasoning.Fields["efforts"]; exists {
			if _, ok := getStringArrayField(reasoning.Fields, "efforts"); !ok {
				return decl, []error{fmt.Errorf("%s: MODEL reasoning efforts must be an array of strings", decl.Pos)}
			}
		}
	}
	return decl, nil
}

func rawToPlugin(r *rawPluginDecl) *PluginDecl {
	return &PluginDecl{
		Name:       unquoteIdent(r.Name),
		PluginType: normalizePluginName(r.PluginType),
		Fields:     entriesToMap(r.Fields),
		Pos:        posFromLexer(r.Pos),
	}
}

func rawToPluginRef(r *rawPluginRef) *PluginRef {
	ref := &PluginRef{
		Name: normalizePluginName(unquoteIdent(r.Name)),
		Pos:  posFromLexer(r.Pos),
	}
	if len(r.Fields) > 0 {
		ref.Fields = entriesToMap(r.Fields)
	}
	return ref
}

// knownInlinePluginAliases maps hyphenated plugin type names to their canonical
// underscore form. Only known inline types are normalized; template names pass
// through unchanged so "PLUGIN my-template system_prompt {}" keeps its name.
var knownInlinePluginAliases = map[string]string{
	"context-compression": "context_compression",
	"system-prompt":       "system_prompt",
	"header-mutation":     "header_mutation",
	"router-replay":       "router_replay",
	"image-gen":           "image_gen",
	"fast-response":       "fast_response",
	"request-params":      "request_params",
	"response-jailbreak":  "response_jailbreak",
}

// normalizePluginName converts known hyphenated plugin type aliases to their
// canonical underscore form. Unknown names (e.g. template references) are
// returned unchanged.
func normalizePluginName(name string) string {
	if canonical, ok := knownInlinePluginAliases[name]; ok {
		return canonical
	}
	return name
}

func rawToAlgo(r *rawAlgoSpec) *AlgoSpec {
	return &AlgoSpec{
		AlgoType: r.AlgoType,
		Fields:   entriesToMap(r.Fields),
		Pos:      posFromLexer(r.Pos),
	}
}

func rawToModelRef(r *rawModelRef) *ModelRef {
	m := &ModelRef{
		Model: unquote(r.Model),
		Pos:   posFromLexer(r.Pos),
	}
	for _, opt := range r.Options {
		if opt.Value == nil {
			continue
		}
		v := opt.Value
		switch opt.Key {
		case "reasoning":
			if v.Bool != nil {
				b := *v.Bool == "true"
				m.Reasoning = &b
			}
		case "effort":
			if v.Str != nil {
				m.Effort = unquote(*v.Str)
			}
		case "reasoning_description":
			if v.Str != nil {
				m.ReasoningDescription = unquote(*v.Str)
			}
		case "lora":
			if v.Str != nil {
				m.LoRA = unquote(*v.Str)
			}
		case "param_size":
			if v.Str != nil {
				m.ParamSize = unquote(*v.Str)
			}
		case "weight":
			if v.Int != nil {
				m.Weight = float64(*v.Int)
			} else if v.Float != nil {
				m.Weight = *v.Float
			}
		}
	}
	return m
}

// ---------- Boolean Expression Conversion ----------

func toBoolExpr(top *BoolExprTop) BoolExpr {
	if top == nil || len(top.Terms) == 0 {
		return nil
	}
	result := toAndExpr(top.Terms[0])
	for i := 1; i < len(top.Terms); i++ {
		right := toAndExpr(top.Terms[i])
		result = &BoolOr{Left: result, Right: right, Pos: posFromLexer(top.Pos)}
	}
	return result
}

func toAndExpr(term *BoolAndTerm) BoolExpr {
	if term == nil || len(term.Factors) == 0 {
		return nil
	}
	result := toFactorExpr(term.Factors[0])
	for i := 1; i < len(term.Factors); i++ {
		right := toFactorExpr(term.Factors[i])
		result = &BoolAnd{Left: result, Right: right, Pos: posFromLexer(term.Pos)}
	}
	return result
}

func toFactorExpr(f *BoolFactor) BoolExpr {
	if f == nil {
		return nil
	}
	pos := posFromLexer(f.Pos)
	switch {
	case f.Not != nil:
		return &BoolNot{Expr: toFactorExpr(f.Not), Pos: pos}
	case f.Paren != nil:
		return toBoolExpr(f.Paren)
	case f.SignalRef != nil:
		return &SignalRefExpr{
			SignalType: f.SignalRef.SignalType,
			SignalName: unquote(f.SignalRef.SignalName),
			Fields:     entriesToMap(f.SignalRef.Fields),
			Pos:        pos,
		}
	}
	return nil
}

// ---------- Field / Value Conversion ----------

func entriesToMap(entries []*FieldEntry) map[string]Value {
	result := make(map[string]Value, len(entries))
	for _, e := range entries {
		if e != nil && e.Value != nil {
			result[e.Key] = valToValue(e.Value)
		}
	}
	return result
}

func valToValue(v *Val) Value {
	if v == nil {
		return nil
	}
	switch {
	case v.Str != nil:
		return StringValue{V: unquote(*v.Str)}
	case v.Float != nil:
		return FloatValue{V: *v.Float}
	case v.Int != nil:
		return IntValue{V: *v.Int}
	case v.Bool != nil:
		return BoolValue{V: *v.Bool == "true"}
	case v.ArrayVal != nil:
		items := make([]Value, 0, len(v.ArrayVal.Items))
		for _, item := range v.ArrayVal.Items {
			items = append(items, valToValue(item))
		}
		return ArrayValue{Items: items}
	case v.Object != nil:
		return ObjectValue{Fields: entriesToMap(v.Object)}
	case v.BareStr != nil:
		return StringValue{V: *v.BareStr}
	}
	return nil
}

// ---------- String Helpers ----------

// unquote removes surrounding quotes and handles escapes.
func unquote(s string) string {
	if len(s) >= 2 && s[0] == '"' && s[len(s)-1] == '"' {
		if unq, err := strconv.Unquote(s); err == nil {
			return unq
		}
		return s[1 : len(s)-1]
	}
	return s
}

// unquoteIdent unquotes a name that may be either a bare ident or a quoted string.
func unquoteIdent(s string) string {
	return unquote(s)
}
