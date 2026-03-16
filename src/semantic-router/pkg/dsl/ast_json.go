package dsl

import "encoding/json"

// ---------- AST → JSON serialization ----------
//
// The AST uses Go interfaces (BoolExpr, Value) which require
// custom JSON marshaling to produce a typed, discriminated union
// that the frontend Visual Builder can consume.

// ProgramJSON is the JSON-serializable form of Program.
type ProgramJSON struct {
	Signals []*SignalDeclJSON `json:"signals"`
	Routes  []*RouteDeclJSON  `json:"routes"`
	Models  []*ModelDeclJSON  `json:"models"`
	Plugins []*PluginDeclJSON `json:"plugins"`
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
	When        *BoolExprJSON    `json:"when,omitempty"`
	Models      []*ModelRefJSON  `json:"models"`
	Algorithm   *AlgoSpecJSON    `json:"algorithm,omitempty"`
	Plugins     []*PluginRefJSON `json:"plugins"`
	Pos         Position         `json:"pos"`
}

// ModelDeclJSON is the JSON form of ModelDecl.
type ModelDeclJSON struct {
	Name   string     `json:"name"`
	Fields JSONObject `json:"fields"`
	Pos    Position   `json:"pos"`
}

// ModelRefJSON is the JSON form of ModelRef.
type ModelRefJSON struct {
	Model     string   `json:"model"`
	Reasoning *bool    `json:"reasoning,omitempty"`
	Effort    string   `json:"effort,omitempty"`
	LoRA      string   `json:"lora,omitempty"`
	ParamSize string   `json:"paramSize,omitempty"`
	Weight    float64  `json:"weight,omitempty"`
	Pos       Position `json:"pos"`
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
	Pos        Position      `json:"pos"`
}

// ProgramToJSON converts a resolved AST Program to its JSON-serializable form.
func ProgramToJSON(prog *Program) *ProgramJSON {
	if prog == nil {
		return nil
	}

	result := &ProgramJSON{
		Signals: make([]*SignalDeclJSON, 0, len(prog.Signals)),
		Routes:  make([]*RouteDeclJSON, 0, len(prog.Routes)),
		Models:  make([]*ModelDeclJSON, 0, len(prog.Models)),
		Plugins: make([]*PluginDeclJSON, 0, len(prog.Plugins)),
	}

	for _, s := range prog.Signals {
		result.Signals = append(result.Signals, &SignalDeclJSON{
			SignalType: s.SignalType,
			Name:       s.Name,
			Fields:     marshalObjectFields(s.Fields),
			Pos:        s.Pos,
		})
	}

	for _, r := range prog.Routes {
		rj := &RouteDeclJSON{
			Name:        r.Name,
			Description: r.Description,
			Priority:    r.Priority,
			When:        marshalBoolExpr(r.When),
			Models:      make([]*ModelRefJSON, 0, len(r.Models)),
			Plugins:     make([]*PluginRefJSON, 0, len(r.Plugins)),
			Pos:         r.Pos,
		}
		for _, m := range r.Models {
			rj.Models = append(rj.Models, &ModelRefJSON{
				Model:     m.Model,
				Reasoning: m.Reasoning,
				Effort:    m.Effort,
				LoRA:      m.LoRA,
				ParamSize: m.ParamSize,
				Weight:    m.Weight,
				Pos:       m.Pos,
			})
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
		result.Routes = append(result.Routes, rj)
	}

	for _, m := range prog.Models {
		result.Models = append(result.Models, &ModelDeclJSON{
			Name:   m.Name,
			Fields: marshalObjectFields(m.Fields),
			Pos:    m.Pos,
		})
	}

	for _, p := range prog.Plugins {
		result.Plugins = append(result.Plugins, &PluginDeclJSON{
			Name:       p.Name,
			PluginType: p.PluginType,
			Fields:     marshalObjectFields(p.Fields),
			Pos:        p.Pos,
		})
	}

	return result
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
			Pos:        e.Pos,
		}
	default:
		return nil
	}
}
