package dsl

import "encoding/json"

// ---------- AST → JSON serialization ----------
//
// The AST uses Go interfaces (BoolExpr, Value) which require
// custom JSON marshaling to produce a typed, discriminated union
// that the frontend Visual Builder can consume.

// ProgramJSON is the JSON-serializable form of Program.
type ProgramJSON struct {
	Signals  []*SignalDeclJSON  `json:"signals"`
	Routes   []*RouteDeclJSON   `json:"routes"`
	Plugins  []*PluginDeclJSON  `json:"plugins"`
	Backends []*BackendDeclJSON `json:"backends"`
	Global   *GlobalDeclJSON    `json:"global,omitempty"`
}

// SignalDeclJSON is the JSON form of SignalDecl.
type SignalDeclJSON struct {
	SignalType string                 `json:"signalType"`
	Name       string                 `json:"name"`
	Fields     map[string]interface{} `json:"fields"`
	Pos        Position               `json:"pos"`
}

// RouteDeclJSON is the JSON form of RouteDecl.
type RouteDeclJSON struct {
	Name        string           `json:"name"`
	Description string           `json:"description,omitempty"`
	Priority    int              `json:"priority"`
	When        interface{}      `json:"when"`
	Models      []*ModelRefJSON  `json:"models"`
	Algorithm   *AlgoSpecJSON    `json:"algorithm,omitempty"`
	Plugins     []*PluginRefJSON `json:"plugins"`
	Pos         Position         `json:"pos"`
}

// ModelRefJSON is the JSON form of ModelRef.
type ModelRefJSON struct {
	Model           string   `json:"model"`
	Reasoning       *bool    `json:"reasoning,omitempty"`
	Effort          string   `json:"effort,omitempty"`
	LoRA            string   `json:"lora,omitempty"`
	ParamSize       string   `json:"paramSize,omitempty"`
	Weight          float64  `json:"weight,omitempty"`
	ReasoningFamily string   `json:"reasoningFamily,omitempty"`
	Pos             Position `json:"pos"`
}

// AlgoSpecJSON is the JSON form of AlgoSpec.
type AlgoSpecJSON struct {
	AlgoType string                 `json:"algoType"`
	Fields   map[string]interface{} `json:"fields"`
	Pos      Position               `json:"pos"`
}

// PluginDeclJSON is the JSON form of PluginDecl.
type PluginDeclJSON struct {
	Name       string                 `json:"name"`
	PluginType string                 `json:"pluginType"`
	Fields     map[string]interface{} `json:"fields"`
	Pos        Position               `json:"pos"`
}

// PluginRefJSON is the JSON form of PluginRef.
type PluginRefJSON struct {
	Name   string                 `json:"name"`
	Fields map[string]interface{} `json:"fields,omitempty"`
	Pos    Position               `json:"pos"`
}

// BackendDeclJSON is the JSON form of BackendDecl.
type BackendDeclJSON struct {
	BackendType string                 `json:"backendType"`
	Name        string                 `json:"name"`
	Fields      map[string]interface{} `json:"fields"`
	Pos         Position               `json:"pos"`
}

// GlobalDeclJSON is the JSON form of GlobalDecl.
type GlobalDeclJSON struct {
	Fields map[string]interface{} `json:"fields"`
	Pos    Position               `json:"pos"`
}

// ProgramToJSON converts a resolved AST Program to its JSON-serializable form.
func ProgramToJSON(prog *Program) *ProgramJSON {
	if prog == nil {
		return nil
	}

	result := &ProgramJSON{
		Signals:  make([]*SignalDeclJSON, 0, len(prog.Signals)),
		Routes:   make([]*RouteDeclJSON, 0, len(prog.Routes)),
		Plugins:  make([]*PluginDeclJSON, 0, len(prog.Plugins)),
		Backends: make([]*BackendDeclJSON, 0, len(prog.Backends)),
	}

	for _, s := range prog.Signals {
		result.Signals = append(result.Signals, &SignalDeclJSON{
			SignalType: s.SignalType,
			Name:       s.Name,
			Fields:     marshalFields(s.Fields),
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
				Model:           m.Model,
				Reasoning:       m.Reasoning,
				Effort:          m.Effort,
				LoRA:            m.LoRA,
				ParamSize:       m.ParamSize,
				Weight:          m.Weight,
				ReasoningFamily: m.ReasoningFamily,
				Pos:             m.Pos,
			})
		}
		if r.Algorithm != nil {
			rj.Algorithm = &AlgoSpecJSON{
				AlgoType: r.Algorithm.AlgoType,
				Fields:   marshalFields(r.Algorithm.Fields),
				Pos:      r.Algorithm.Pos,
			}
		}
		for _, p := range r.Plugins {
			pj := &PluginRefJSON{
				Name: p.Name,
				Pos:  p.Pos,
			}
			if len(p.Fields) > 0 {
				pj.Fields = marshalFields(p.Fields)
			}
			rj.Plugins = append(rj.Plugins, pj)
		}
		result.Routes = append(result.Routes, rj)
	}

	for _, p := range prog.Plugins {
		result.Plugins = append(result.Plugins, &PluginDeclJSON{
			Name:       p.Name,
			PluginType: p.PluginType,
			Fields:     marshalFields(p.Fields),
			Pos:        p.Pos,
		})
	}

	for _, b := range prog.Backends {
		result.Backends = append(result.Backends, &BackendDeclJSON{
			BackendType: b.BackendType,
			Name:        b.Name,
			Fields:      marshalFields(b.Fields),
			Pos:         b.Pos,
		})
	}

	if prog.Global != nil {
		result.Global = &GlobalDeclJSON{
			Fields: marshalFields(prog.Global.Fields),
			Pos:    prog.Global.Pos,
		}
	}

	return result
}

// MarshalProgramJSON marshals a Program to JSON bytes.
func MarshalProgramJSON(prog *Program) ([]byte, error) {
	pj := ProgramToJSON(prog)
	return json.Marshal(pj)
}

// ---------- Helper: Value → interface{} ----------

func marshalFields(fields map[string]Value) map[string]interface{} {
	if fields == nil {
		return map[string]interface{}{}
	}
	result := make(map[string]interface{}, len(fields))
	for k, v := range fields {
		result[k] = marshalValue(v)
	}
	return result
}

func marshalValue(v Value) interface{} {
	if v == nil {
		return nil
	}
	switch val := v.(type) {
	case StringValue:
		return val.V
	case IntValue:
		return val.V
	case FloatValue:
		return val.V
	case BoolValue:
		return val.V
	case ArrayValue:
		items := make([]interface{}, len(val.Items))
		for i, item := range val.Items {
			items[i] = marshalValue(item)
		}
		return items
	case ObjectValue:
		return marshalFields(val.Fields)
	default:
		return nil
	}
}

// ---------- Helper: BoolExpr → discriminated JSON ----------

// marshalBoolExpr converts a BoolExpr to a JSON-friendly map with a "type" discriminator.
//
// Output shapes:
//
//	{ "type": "and", "left": ..., "right": ..., "pos": ... }
//	{ "type": "or",  "left": ..., "right": ..., "pos": ... }
//	{ "type": "not", "expr": ..., "pos": ... }
//	{ "type": "signal_ref", "signalType": "domain", "signalName": "math", "pos": ... }
func marshalBoolExpr(expr BoolExpr) interface{} {
	if expr == nil {
		return nil
	}
	switch e := expr.(type) {
	case *BoolAnd:
		return map[string]interface{}{
			"type":  "and",
			"left":  marshalBoolExpr(e.Left),
			"right": marshalBoolExpr(e.Right),
			"pos":   e.Pos,
		}
	case *BoolOr:
		return map[string]interface{}{
			"type":  "or",
			"left":  marshalBoolExpr(e.Left),
			"right": marshalBoolExpr(e.Right),
			"pos":   e.Pos,
		}
	case *BoolNot:
		return map[string]interface{}{
			"type": "not",
			"expr": marshalBoolExpr(e.Expr),
			"pos":  e.Pos,
		}
	case *SignalRefExpr:
		return map[string]interface{}{
			"type":       "signal_ref",
			"signalType": e.SignalType,
			"signalName": e.SignalName,
			"pos":        e.Pos,
		}
	default:
		return nil
	}
}
