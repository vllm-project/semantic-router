package dsl

import "encoding/json"

// ---------- AST → JSON serialization ----------
//
// The AST uses Go interfaces (BoolExpr, Value) which require
// custom JSON marshaling to produce a typed, discriminated union
// that the frontend Visual Builder can consume.

// ProgramJSON is the JSON-serializable form of Program.
type ProgramJSON struct {
	Strategy             string                         `json:"strategy,omitempty"`
	Entrypoints          []*EntrypointDeclJSON          `json:"entrypoints,omitempty"`
	Recipes              []*RecipeDeclJSON              `json:"recipes,omitempty"`
	Signals              []*SignalDeclJSON              `json:"signals"`
	ProjectionPartitions []*ProjectionPartitionDeclJSON `json:"projectionPartitions,omitempty"`
	ProjectionScores     []*ProjectionScoreDeclJSON     `json:"projectionScores,omitempty"`
	ProjectionMappings   []*ProjectionMappingDeclJSON   `json:"projectionMappings,omitempty"`
	Routes               []*RouteDeclJSON               `json:"routes"`
	Models               []*ModelDeclJSON               `json:"models"`
	Plugins              []*PluginDeclJSON              `json:"plugins"`
	TestBlocks           []*TestBlockDeclJSON           `json:"testBlocks,omitempty"`
}

// EntrypointDeclJSON is the JSON form of a request-facing recipe binding.
type EntrypointDeclJSON struct {
	Name        string                                      `json:"name"`
	Aliases     []string                                    `json:"aliases,omitempty"`
	Recipe      string                                      `json:"recipe,omitempty"`
	Assignments map[string]*EntrypointAssignmentSetDeclJSON `json:"assignments,omitempty"`
	Rules       []*EntrypointRuleDeclJSON                   `json:"rules,omitempty"`
	Pos         Position                                    `json:"pos"`
}

type EntrypointRuleDeclJSON struct {
	Name        string                                      `json:"name"`
	Matches     []*EntrypointMatchDeclJSON                  `json:"matches,omitempty"`
	Recipe      string                                      `json:"recipe"`
	Assignments map[string]*EntrypointAssignmentSetDeclJSON `json:"assignments"`
	Pos         Position                                    `json:"pos"`
}

type EntrypointMatchDeclJSON struct {
	Claim *EntrypointClaimMatchDeclJSON `json:"claim,omitempty"`
	Path  *EntrypointPathMatchDeclJSON  `json:"path,omitempty"`
}

type EntrypointClaimMatchDeclJSON struct {
	Name  string    `json:"name"`
	Exact JSONValue `json:"exact"`
}

type EntrypointPathMatchDeclJSON struct {
	Exact  string `json:"exact,omitempty"`
	Prefix string `json:"prefix,omitempty"`
}

type EntrypointAssignmentSetDeclJSON struct {
	Models   []*EntrypointAssignmentDeclJSON `json:"models"`
	Fallback *EntrypointFallbackDeclJSON     `json:"fallback,omitempty"`
}

type EntrypointAssignmentDeclJSON struct {
	Model     string                       `json:"model"`
	Priority  int                          `json:"priority"`
	Weight    string                       `json:"weight,omitempty"`
	LoRAName  string                       `json:"loraName,omitempty"`
	Reasoning *EntrypointReasoningDeclJSON `json:"reasoning,omitempty"`
	Pos       Position                     `json:"pos"`
}

type EntrypointFallbackDeclJSON struct {
	Strategy string   `json:"strategy"`
	On       []string `json:"on"`
}

type EntrypointReasoningDeclJSON struct {
	Enabled     bool   `json:"enabled"`
	Effort      string `json:"effort,omitempty"`
	Description string `json:"description,omitempty"`
}

// RecipeDeclJSON is the JSON form of one isolated recipe program.
type RecipeDeclJSON struct {
	Name        string       `json:"name"`
	Description string       `json:"description,omitempty"`
	Program     *ProgramJSON `json:"program"`
	Pos         Position     `json:"pos"`
}

// ProjectionPartitionDeclJSON is the JSON form of ProjectionPartitionDecl.
type ProjectionPartitionDeclJSON struct {
	Name        string   `json:"name"`
	Semantics   string   `json:"semantics,omitempty"`
	Temperature float64  `json:"temperature,omitempty"`
	Members     []string `json:"members"`
	Default     string   `json:"default,omitempty"`
	Pos         Position `json:"pos"`
}

// ProjectionScoreDeclJSON is the JSON form of ProjectionScoreDecl.
type ProjectionScoreDeclJSON struct {
	Name   string                      `json:"name"`
	Method string                      `json:"method,omitempty"`
	Inputs []*ProjectionScoreInputJSON `json:"inputs,omitempty"`
	Pos    Position                    `json:"pos"`
}

// ProjectionScoreInputJSON is the JSON form of ProjectionScoreInputDecl.
type ProjectionScoreInputJSON struct {
	SignalType  string  `json:"signalType"`
	SignalName  string  `json:"signalName"`
	Weight      float64 `json:"weight"`
	ValueSource string  `json:"valueSource,omitempty"`
	Match       float64 `json:"match,omitempty"`
	Miss        float64 `json:"miss,omitempty"`
}

// ProjectionMappingDeclJSON is the JSON form of ProjectionMappingDecl.
type ProjectionMappingDeclJSON struct {
	Name        string                            `json:"name"`
	Source      string                            `json:"source,omitempty"`
	Method      string                            `json:"method,omitempty"`
	Calibration *ProjectionMappingCalibrationJSON `json:"calibration,omitempty"`
	Outputs     []*ProjectionMappingOutputJSON    `json:"outputs,omitempty"`
	Pos         Position                          `json:"pos"`
}

// ProjectionMappingCalibrationJSON is the JSON form of ProjectionMappingCalibrationDecl.
type ProjectionMappingCalibrationJSON struct {
	Method string  `json:"method,omitempty"`
	Slope  float64 `json:"slope,omitempty"`
}

// ProjectionMappingOutputJSON is the JSON form of ProjectionMappingOutputDecl.
type ProjectionMappingOutputJSON struct {
	Name string   `json:"name"`
	LT   *float64 `json:"lt,omitempty"`
	LTE  *float64 `json:"lte,omitempty"`
	GT   *float64 `json:"gt,omitempty"`
	GTE  *float64 `json:"gte,omitempty"`
}

// TestBlockDeclJSON is the JSON form of TestBlockDecl.
type TestBlockDeclJSON struct {
	Name    string               `json:"name"`
	Entries []*TestEntryDeclJSON `json:"entries"`
	Pos     Position             `json:"pos"`
}

// TestEntryDeclJSON is the JSON form of TestEntry.
type TestEntryDeclJSON struct {
	Query     string   `json:"query"`
	RouteName string   `json:"routeName"`
	Pos       Position `json:"pos"`
}

// SignalDeclJSON is the JSON form of SignalDecl.
type SignalDeclJSON struct {
	SignalType string     `json:"signalType"`
	Name       string     `json:"name"`
	Fields     JSONObject `json:"fields"`
	Pos        Position   `json:"pos"`
}

// RouteDeclJSON is the JSON form of RouteDecl.
type RouteDeclJSON struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Priority    int              `json:"priority"`
	Tier        int              `json:"tier,omitempty"`
	When        *BoolExprJSON    `json:"when,omitempty"`
	Models      []*ModelRefJSON  `json:"models"`
	Algorithm   *AlgoSpecJSON    `json:"algorithm,omitempty"`
	Plugins     []*PluginRefJSON `json:"plugins"`
	Emits       []*EmitDeclJSON  `json:"emits,omitempty"`
	Pos         Position         `json:"pos"`
}

// EmitDeclJSON is the JSON form of an EMIT directive on a route.
type EmitDeclJSON struct {
	Kind      string                  `json:"kind"`
	Retention *RetentionDirectiveJSON `json:"retention,omitempty"`
	Pos       Position                `json:"pos"`
}

// RetentionDirectiveJSON is the JSON form of RetentionDirective. All fields
// are tri-state pointers to preserve "unset" vs explicit zero.
type RetentionDirectiveJSON struct {
	Drop                  *bool `json:"drop,omitempty"`
	TTLTurns              *int  `json:"ttl_turns,omitempty"`
	KeepCurrentModel      *bool `json:"keep_current_model,omitempty"`
	PreferPrefixRetention *bool `json:"prefer_prefix_retention,omitempty"`
}

// ModelDeclJSON is the JSON form of ModelDecl.
type ModelDeclJSON struct {
	Name   string     `json:"name"`
	Fields JSONObject `json:"fields"`
	Pos    Position   `json:"pos"`
}

// ModelRefJSON is the JSON form of ModelRef.
type ModelRefJSON struct {
	Model                string   `json:"model"`
	Reasoning            *bool    `json:"reasoning,omitempty"`
	ReasoningDescription string   `json:"reasoningDescription,omitempty"`
	Effort               string   `json:"effort,omitempty"`
	LoRA                 string   `json:"lora,omitempty"`
	ParamSize            string   `json:"paramSize,omitempty"`
	Weight               float64  `json:"weight,omitempty"`
	Pos                  Position `json:"pos"`
}

// AlgoSpecJSON is the JSON form of AlgoSpec.
type AlgoSpecJSON struct {
	AlgoType string     `json:"algoType"`
	Fields   JSONObject `json:"fields"`
	Pos      Position   `json:"pos"`
}

// PluginDeclJSON is the JSON form of PluginDecl.
type PluginDeclJSON struct {
	Name       string     `json:"name"`
	PluginType string     `json:"pluginType"`
	Fields     JSONObject `json:"fields"`
	Pos        Position   `json:"pos"`
}

// PluginRefJSON is the JSON form of PluginRef.
type PluginRefJSON struct {
	Name   string      `json:"name"`
	Fields *JSONObject `json:"fields,omitempty"`
	Pos    Position    `json:"pos"`
}

// JSONObject preserves object-shaped DSL field trees without weak map types.
type JSONObject struct {
	Fields []JSONField
}

// JSONField represents one key/value entry in a JSON object.
type JSONField struct {
	Name  string
	Value JSONValue
}

// JSONValue is a typed recursive JSON value that still marshals to plain JSON.
type JSONValue struct {
	Kind   JSONValueKind
	String string
	Int    int
	Float  float64
	Bool   bool
	Array  []JSONValue
	Object *JSONObject
}

// JSONValueKind identifies which JSONValue branch is active.
type JSONValueKind string

const (
	JSONValueNull   JSONValueKind = "null"
	JSONValueString JSONValueKind = "string"
	JSONValueInt    JSONValueKind = "int"
	JSONValueFloat  JSONValueKind = "float"
	JSONValueBool   JSONValueKind = "bool"
	JSONValueArray  JSONValueKind = "array"
	JSONValueObject JSONValueKind = "object"
)

// BoolExprJSON is the discriminated JSON form of BoolExpr.
type BoolExprJSON struct {
	Type       string        `json:"type"`
	Left       *BoolExprJSON `json:"left,omitempty"`
	Right      *BoolExprJSON `json:"right,omitempty"`
	Expr       *BoolExprJSON `json:"expr,omitempty"`
	SignalType string        `json:"signalType,omitempty"`
	SignalName string        `json:"signalName,omitempty"`
	Fields     *JSONObject   `json:"fields,omitempty"`
	Pos        Position      `json:"pos"`
}

// ProgramToJSON converts a resolved AST Program to its JSON-serializable form.
func ProgramToJSON(prog *Program) *ProgramJSON {
	if prog == nil {
		return nil
	}

	result := &ProgramJSON{
		Strategy: prog.Strategy,
		Signals:  make([]*SignalDeclJSON, 0, len(prog.Signals)),
		Routes:   make([]*RouteDeclJSON, 0, len(prog.Routes)),
		Models:   make([]*ModelDeclJSON, 0, len(prog.Models)),
		Plugins:  make([]*PluginDeclJSON, 0, len(prog.Plugins)),
	}
	for _, entrypoint := range prog.Entrypoints {
		result.Entrypoints = append(result.Entrypoints, &EntrypointDeclJSON{
			Name: entrypoint.Name, Aliases: append([]string(nil), entrypoint.Aliases...),
			Recipe: entrypoint.Recipe, Assignments: entrypointAssignmentsToJSON(entrypoint.Assignments),
			Rules: entrypointRulesToJSON(entrypoint.Rules), Pos: entrypoint.Pos,
		})
	}
	for _, recipe := range prog.Recipes {
		result.Recipes = append(result.Recipes, &RecipeDeclJSON{
			Name:        recipe.Name,
			Description: recipe.Description,
			Program:     ProgramToJSON(recipe.Program),
			Pos:         recipe.Pos,
		})
	}
	appendSignalDecls(result, prog.Signals)
	appendProjectionPartitionDecls(result, prog.ProjectionPartitions)
	appendProjectionScoreDecls(result, prog.ProjectionScores)
	appendProjectionMappingDecls(result, prog.ProjectionMappings)
	appendRouteDecls(result, prog.Routes)
	appendModelDecls(result, prog.Models)
	appendPluginDecls(result, prog.Plugins)
	appendTestBlockDecls(result, prog.TestBlocks)
	return result
}

func appendSignalDecls(result *ProgramJSON, signals []*SignalDecl) {
	for _, signal := range signals {
		result.Signals = append(result.Signals, &SignalDeclJSON{
			SignalType: signal.SignalType,
			Name:       signal.Name,
			Fields:     marshalObjectFields(signal.Fields),
			Pos:        signal.Pos,
		})
	}
}

func appendProjectionPartitionDecls(
	result *ProgramJSON,
	partitions []*ProjectionPartitionDecl,
) {
	for _, partition := range partitions {
		result.ProjectionPartitions = append(
			result.ProjectionPartitions,
			&ProjectionPartitionDeclJSON{
				Name:        partition.Name,
				Semantics:   partition.Semantics,
				Temperature: partition.Temperature,
				Members:     partition.Members,
				Default:     partition.Default,
				Pos:         partition.Pos,
			},
		)
	}
}

func appendProjectionScoreDecls(result *ProgramJSON, scores []*ProjectionScoreDecl) {
	for _, score := range scores {
		scoreJSON := &ProjectionScoreDeclJSON{
			Name:   score.Name,
			Method: score.Method,
			Pos:    score.Pos,
		}
		for _, input := range score.Inputs {
			scoreJSON.Inputs = append(scoreJSON.Inputs, &ProjectionScoreInputJSON{
				SignalType:  input.SignalType,
				SignalName:  input.SignalName,
				Weight:      input.Weight,
				ValueSource: input.ValueSource,
				Match:       input.Match,
				Miss:        input.Miss,
			})
		}
		result.ProjectionScores = append(result.ProjectionScores, scoreJSON)
	}
}

func appendProjectionMappingDecls(result *ProgramJSON, mappings []*ProjectionMappingDecl) {
	for _, mapping := range mappings {
		mappingJSON := &ProjectionMappingDeclJSON{
			Name:   mapping.Name,
			Source: mapping.Source,
			Method: mapping.Method,
			Pos:    mapping.Pos,
		}
		if mapping.Calibration != nil {
			mappingJSON.Calibration = &ProjectionMappingCalibrationJSON{
				Method: mapping.Calibration.Method,
				Slope:  mapping.Calibration.Slope,
			}
		}
		for _, output := range mapping.Outputs {
			mappingJSON.Outputs = append(mappingJSON.Outputs, &ProjectionMappingOutputJSON{
				Name: output.Name,
				LT:   output.LT,
				LTE:  output.LTE,
				GT:   output.GT,
				GTE:  output.GTE,
			})
		}
		result.ProjectionMappings = append(result.ProjectionMappings, mappingJSON)
	}
}

func appendRouteDecls(result *ProgramJSON, routes []*RouteDecl) {
	for _, route := range routes {
		result.Routes = append(result.Routes, routeDeclToJSON(route))
	}
}

func appendModelDecls(result *ProgramJSON, models []*ModelDecl) {
	for _, model := range models {
		result.Models = append(result.Models, &ModelDeclJSON{
			Name:   model.Name,
			Fields: marshalObjectFields(model.Fields),
			Pos:    model.Pos,
		})
	}
}

func appendPluginDecls(result *ProgramJSON, plugins []*PluginDecl) {
	for _, plugin := range plugins {
		result.Plugins = append(result.Plugins, &PluginDeclJSON{
			Name:       plugin.Name,
			PluginType: plugin.PluginType,
			Fields:     marshalObjectFields(plugin.Fields),
			Pos:        plugin.Pos,
		})
	}
}

func appendTestBlockDecls(result *ProgramJSON, blocks []*TestBlockDecl) {
	for _, block := range blocks {
		testBlockJSON := &TestBlockDeclJSON{
			Name:    block.Name,
			Entries: make([]*TestEntryDeclJSON, 0, len(block.Entries)),
			Pos:     block.Pos,
		}
		for _, entry := range block.Entries {
			testBlockJSON.Entries = append(testBlockJSON.Entries, &TestEntryDeclJSON{
				Query:     entry.Query,
				RouteName: entry.RouteName,
				Pos:       entry.Pos,
			})
		}
		result.TestBlocks = append(result.TestBlocks, testBlockJSON)
	}
}

func entrypointRulesToJSON(rules []*EntrypointRuleDecl) []*EntrypointRuleDeclJSON {
	result := make([]*EntrypointRuleDeclJSON, 0, len(rules))
	for _, rule := range rules {
		if rule == nil {
			continue
		}
		converted := &EntrypointRuleDeclJSON{
			Name: rule.Name, Recipe: rule.Recipe,
			Assignments: entrypointAssignmentsToJSON(rule.Assignments), Pos: rule.Pos,
		}
		for _, match := range rule.Matches {
			if match == nil {
				continue
			}
			convertedMatch := &EntrypointMatchDeclJSON{}
			if match.Claim != nil {
				convertedMatch.Claim = &EntrypointClaimMatchDeclJSON{Name: match.Claim.Name, Exact: marshalValue(match.Claim.Exact)}
			}
			if match.Path != nil {
				convertedMatch.Path = &EntrypointPathMatchDeclJSON{Exact: match.Path.Exact, Prefix: match.Path.Prefix}
			}
			converted.Matches = append(converted.Matches, convertedMatch)
		}
		result = append(result, converted)
	}
	return result
}

func entrypointAssignmentsToJSON(input map[string]*EntrypointAssignmentSetDecl) map[string]*EntrypointAssignmentSetDeclJSON {
	if input == nil {
		return nil
	}
	output := make(map[string]*EntrypointAssignmentSetDeclJSON, len(input))
	for decisionName, assignmentSet := range input {
		if assignmentSet == nil {
			continue
		}
		convertedSet := &EntrypointAssignmentSetDeclJSON{Models: make([]*EntrypointAssignmentDeclJSON, 0, len(assignmentSet.Models))}
		for _, assignment := range assignmentSet.Models {
			if assignment == nil {
				continue
			}
			converted := &EntrypointAssignmentDeclJSON{
				Model: assignment.Model, Priority: assignment.Priority, Weight: assignment.Weight,
				LoRAName: assignment.LoRAName, Pos: assignment.Pos,
			}
			if assignment.Reasoning != nil {
				converted.Reasoning = &EntrypointReasoningDeclJSON{
					Enabled: assignment.Reasoning.Enabled, Effort: assignment.Reasoning.Effort,
					Description: assignment.Reasoning.Description,
				}
			}
			convertedSet.Models = append(convertedSet.Models, converted)
		}
		if assignmentSet.Fallback != nil {
			convertedSet.Fallback = &EntrypointFallbackDeclJSON{
				Strategy: assignmentSet.Fallback.Strategy,
				On:       append([]string(nil), assignmentSet.Fallback.On...),
			}
		}
		output[decisionName] = convertedSet
	}
	return output
}

func modelRefToJSON(model *ModelRef) *ModelRefJSON {
	if model == nil {
		return nil
	}
	return &ModelRefJSON{
		Model:                model.Model,
		Reasoning:            model.Reasoning,
		ReasoningDescription: model.ReasoningDescription,
		Effort:               model.Effort,
		LoRA:                 model.LoRA,
		ParamSize:            model.ParamSize,
		Weight:               model.Weight,
		Pos:                  model.Pos,
	}
}

func routeDeclToJSON(r *RouteDecl) *RouteDeclJSON {
	rj := &RouteDeclJSON{
		Name:        r.Name,
		Description: r.Description,
		Priority:    r.Priority,
		Tier:        r.Tier,
		When:        marshalBoolExpr(r.When),
		Models:      make([]*ModelRefJSON, 0, len(r.Models)),
		Plugins:     make([]*PluginRefJSON, 0, len(r.Plugins)),
		Pos:         r.Pos,
	}
	for _, m := range r.Models {
		rj.Models = append(rj.Models, modelRefToJSON(m))
	}
	if r.Algorithm != nil {
		rj.Algorithm = &AlgoSpecJSON{
			AlgoType: r.Algorithm.AlgoType,
			Fields:   marshalObjectFields(r.Algorithm.Fields),
			Pos:      r.Algorithm.Pos,
		}
	}
	for _, p := range r.Plugins {
		pj := &PluginRefJSON{
			Name: p.Name,
			Pos:  p.Pos,
		}
		if len(p.Fields) > 0 {
			fields := marshalObjectFields(p.Fields)
			pj.Fields = &fields
		}
		rj.Plugins = append(rj.Plugins, pj)
	}
	for _, e := range r.Emits {
		rj.Emits = append(rj.Emits, emitDeclToJSON(e))
	}
	return rj
}

func emitDeclToJSON(e *EmitDecl) *EmitDeclJSON {
	if e == nil {
		return nil
	}
	out := &EmitDeclJSON{
		Kind: e.Kind,
		Pos:  e.Pos,
	}
	if e.Retention != nil {
		out.Retention = &RetentionDirectiveJSON{
			Drop:                  clonePtrBoolDSL(e.Retention.Drop),
			TTLTurns:              clonePtrIntDSL(e.Retention.TTLTurns),
			KeepCurrentModel:      clonePtrBoolDSL(e.Retention.KeepCurrentModel),
			PreferPrefixRetention: clonePtrBoolDSL(e.Retention.PreferPrefixRetention),
		}
	}
	return out
}

// MarshalProgramJSON marshals a Program to JSON bytes.
func MarshalProgramJSON(prog *Program) ([]byte, error) {
	pj := ProgramToJSON(prog)
	return json.Marshal(pj)
}

func (o JSONObject) MarshalJSON() ([]byte, error) {
	raw := make(map[string]json.RawMessage, len(o.Fields))
	for _, field := range o.Fields {
		payload, err := json.Marshal(field.Value)
		if err != nil {
			return nil, err
		}
		raw[field.Name] = payload
	}
	return json.Marshal(raw)
}

func (v JSONValue) MarshalJSON() ([]byte, error) {
	switch v.Kind {
	case JSONValueNull:
		return []byte("null"), nil
	case JSONValueString:
		return json.Marshal(v.String)
	case JSONValueInt:
		return json.Marshal(v.Int)
	case JSONValueFloat:
		return json.Marshal(v.Float)
	case JSONValueBool:
		return json.Marshal(v.Bool)
	case JSONValueArray:
		return json.Marshal(v.Array)
	case JSONValueObject:
		if v.Object == nil {
			return []byte("{}"), nil
		}
		return json.Marshal(v.Object)
	default:
		return []byte("null"), nil
	}
}

func marshalObjectFields(fields map[string]Value) JSONObject {
	if fields == nil {
		return JSONObject{Fields: []JSONField{}}
	}
	result := JSONObject{Fields: make([]JSONField, 0, len(fields))}
	for key, value := range fields {
		result.Fields = append(result.Fields, JSONField{
			Name:  key,
			Value: marshalValue(value),
		})
	}
	return result
}

func marshalValue(v Value) JSONValue {
	if v == nil {
		return JSONValue{Kind: JSONValueNull}
	}
	switch val := v.(type) {
	case StringValue:
		return JSONValue{Kind: JSONValueString, String: val.V}
	case IntValue:
		return JSONValue{Kind: JSONValueInt, Int: val.V}
	case FloatValue:
		return JSONValue{Kind: JSONValueFloat, Float: val.V}
	case BoolValue:
		return JSONValue{Kind: JSONValueBool, Bool: val.V}
	case ArrayValue:
		items := make([]JSONValue, len(val.Items))
		for i, item := range val.Items {
			items[i] = marshalValue(item)
		}
		return JSONValue{Kind: JSONValueArray, Array: items}
	case ObjectValue:
		obj := marshalObjectFields(val.Fields)
		return JSONValue{Kind: JSONValueObject, Object: &obj}
	default:
		return JSONValue{Kind: JSONValueNull}
	}
}

// marshalBoolExpr converts a BoolExpr to a discriminated JSON node.
func marshalBoolExpr(expr BoolExpr) *BoolExprJSON {
	if expr == nil {
		return nil
	}
	switch e := expr.(type) {
	case *BoolAnd:
		return &BoolExprJSON{
			Type:  "and",
			Left:  marshalBoolExpr(e.Left),
			Right: marshalBoolExpr(e.Right),
			Pos:   e.Pos,
		}
	case *BoolOr:
		return &BoolExprJSON{
			Type:  "or",
			Left:  marshalBoolExpr(e.Left),
			Right: marshalBoolExpr(e.Right),
			Pos:   e.Pos,
		}
	case *BoolNot:
		return &BoolExprJSON{
			Type: "not",
			Expr: marshalBoolExpr(e.Expr),
			Pos:  e.Pos,
		}
	case *SignalRefExpr:
		return &BoolExprJSON{
			Type:       "signal_ref",
			SignalType: e.SignalType,
			SignalName: e.SignalName,
			Fields:     marshalOptionalObjectFields(e.Fields),
			Pos:        e.Pos,
		}
	default:
		return nil
	}
}

func marshalOptionalObjectFields(
	fields map[string]Value,
) *JSONObject {
	if len(fields) == 0 {
		return nil
	}
	result := marshalObjectFields(fields)
	return &result
}
