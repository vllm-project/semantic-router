package dsl

import (
	"fmt"

	"github.com/alecthomas/participle/v2/lexer"
)

// Position represents a source location.
type Position struct {
	Line   int
	Column int
}

func (p Position) String() string {
	return fmt.Sprintf("line %d, column %d", p.Line, p.Column)
}

// posFromLexer converts a participle lexer.Position to our Position.
func posFromLexer(p lexer.Position) Position {
	return Position{Line: p.Line, Column: p.Column}
}

// ---------- Participle Grammar AST (raw parse tree) ----------

// rawProgram is the root parse node.
type rawProgram struct {
	Pos     lexer.Position
	Entries []*rawTopLevel `parser:"@@*"`
}

// rawTopLevel is a union for top-level declarations.
type rawTopLevel struct {
	Pos     lexer.Position
	Signal  *rawSignalDecl  `parser:"  @@"`
	Route   *rawRouteDecl   `parser:"| @@"`
	Plugin  *rawPluginDecl  `parser:"| @@"`
	Backend *rawBackendDecl `parser:"| @@"`
	Global  *rawGlobalDecl  `parser:"| @@"`
}

// rawSignalDecl: SIGNAL <type> <name> { fields... }
type rawSignalDecl struct {
	Pos        lexer.Position
	SignalType string        `parser:"'SIGNAL' @Ident"`
	Name       string        `parser:"@(Ident | String)"`
	Fields     []*FieldEntry `parser:"'{' @@* '}'"`
}

// rawRouteDecl: ROUTE <name> (opts...) { body... }
type rawRouteDecl struct {
	Pos  lexer.Position
	Name string          `parser:"'ROUTE' @(Ident | String)"`
	Opts []*RouteOpt     `parser:"( '(' @@* ')' )?"`
	Body []*rawRouteItem `parser:"'{' @@* '}'"`
}

// RouteOpt: key = value inside route header parens
type RouteOpt struct {
	Pos   lexer.Position
	Key   string `parser:"@Ident '='"`
	Value *Val   `parser:"@@"`
}

// rawRouteItem: a single element inside a route body
type rawRouteItem struct {
	Pos       lexer.Position
	Priority  *int          `parser:"  'PRIORITY' @Int"`
	When      *BoolExprTop  `parser:"| 'WHEN' @@"`
	Model     *rawModelList `parser:"| 'MODEL' @@"`
	Algorithm *rawAlgoSpec  `parser:"| 'ALGORITHM' @@"`
	Plugin    *rawPluginRef `parser:"| 'PLUGIN' @@"`
}

// rawPluginDecl: PLUGIN <name> <type> { fields... }
type rawPluginDecl struct {
	Pos        lexer.Position
	Name       string        `parser:"'PLUGIN' @(Ident | String)"`
	PluginType string        `parser:"@Ident"`
	Fields     []*FieldEntry `parser:"'{' @@* '}'"`
}

// rawPluginRef: <name> { fields... }? inside a route
type rawPluginRef struct {
	Pos    lexer.Position
	Name   string        `parser:"@(Ident | String)"`
	Fields []*FieldEntry `parser:"( '{' @@* '}' )?"`
}

// rawBackendDecl: BACKEND <type> <name> { fields... }
type rawBackendDecl struct {
	Pos         lexer.Position
	BackendType string        `parser:"'BACKEND' @Ident"`
	Name        string        `parser:"@(Ident | String)"`
	Fields      []*FieldEntry `parser:"'{' @@* '}'"`
}

// rawGlobalDecl: GLOBAL { fields... }
type rawGlobalDecl struct {
	Pos    lexer.Position
	Fields []*FieldEntry `parser:"'GLOBAL' '{' @@* '}'"`
}

// rawModelList: comma-separated model refs
type rawModelList struct {
	Pos    lexer.Position
	Models []*rawModelRef `parser:"@@ ( ',' @@ )*"`
}

// rawModelRef: "model_name" (options...)?
type rawModelRef struct {
	Pos     lexer.Position
	Model   string      `parser:"@String"`
	Options []*ModelOpt `parser:"( '(' @@ ( ',' @@ )* ')' )?"`
}

// ModelOpt: key = value inside model options
type ModelOpt struct {
	Key   string `parser:"@Ident '='"`
	Value *Val   `parser:"@@"`
}

// rawAlgoSpec: <type> { fields... }?
type rawAlgoSpec struct {
	Pos      lexer.Position
	AlgoType string        `parser:"@Ident"`
	Fields   []*FieldEntry `parser:"( '{' @@* '}' )?"`
}

// ---------- Boolean Expressions (parsed by participle) ----------

// BoolExprTop: OR-separated terms
type BoolExprTop struct {
	Pos   lexer.Position
	Terms []*BoolAndTerm `parser:"@@ ( 'OR' @@ )*"`
}

// BoolAndTerm: AND-separated factors
type BoolAndTerm struct {
	Pos     lexer.Position
	Factors []*BoolFactor `parser:"@@ ( 'AND' @@ )*"`
}

// BoolFactor: NOT, parenthesized expr, or signal ref
type BoolFactor struct {
	Pos       lexer.Position
	Not       *BoolFactor       `parser:"  'NOT' @@"`
	Paren     *BoolExprTop      `parser:"| '(' @@ ')'"`
	SignalRef *rawSignalRefExpr `parser:"| @@"`
}

// rawSignalRefExpr: signal_type("signal_name")
type rawSignalRefExpr struct {
	Pos        lexer.Position
	SignalType string `parser:"@Ident"`
	SignalName string `parser:"'(' @String ')'"`
}

// ---------- Field / Value (parsed by participle) ----------

// FieldEntry: key: value ,?
type FieldEntry struct {
	Pos   lexer.Position
	Key   string `parser:"','? @Ident ':'"`
	Value *Val   `parser:"@@"`
}

// Val: value union (string, int, float, bool, array, object, bare ident)
type Val struct {
	Pos      lexer.Position
	Str      *string       `parser:"  @String"`
	Float    *float64      `parser:"| @Float"`
	Int      *int          `parser:"| @Int"`
	Bool     *string       `parser:"| @('true' | 'false')"`
	ArrayVal *ArrayVal     `parser:"| @@"`
	Object   []*FieldEntry `parser:"| '{' @@* '}'"`
	BareStr  *string       `parser:"| @Ident"`
}

// ArrayVal wraps array parsing to handle empty arrays.
type ArrayVal struct {
	Pos   lexer.Position
	Items []*Val `parser:"'[' ( @@ ( ',' @@ )* ','? )? ']'"`
}

// ---------- Resolved AST (used by compiler/decompiler/validator) ----------

// Program is the root AST node, representing a complete DSL file.
type Program struct {
	Signals  []*SignalDecl
	Routes   []*RouteDecl
	Plugins  []*PluginDecl
	Backends []*BackendDecl
	Global   *GlobalDecl
}

// SignalDecl represents a SIGNAL declaration.
type SignalDecl struct {
	SignalType string
	Name       string
	Fields     map[string]Value
	Pos        Position
}

// RouteDecl represents a ROUTE declaration.
type RouteDecl struct {
	Name        string
	Description string
	Priority    int
	When        BoolExpr
	Models      []*ModelRef
	Algorithm   *AlgoSpec
	Plugins     []*PluginRef
	Pos         Position
}

// PluginDecl represents a top-level PLUGIN template declaration.
type PluginDecl struct {
	Name       string
	PluginType string
	Fields     map[string]Value
	Pos        Position
}

// PluginRef represents a plugin reference inside a ROUTE.
type PluginRef struct {
	Name   string
	Fields map[string]Value
	Pos    Position
}

// BackendDecl represents a BACKEND declaration.
type BackendDecl struct {
	BackendType string
	Name        string
	Fields      map[string]Value
	Pos         Position
}

// GlobalDecl represents the GLOBAL settings block.
type GlobalDecl struct {
	Fields map[string]Value
	Pos    Position
}

// ---------- Boolean Expressions (resolved) ----------

// BoolExpr is the interface for all boolean expression AST nodes.
type BoolExpr interface {
	boolExpr() // marker method
	GetPos() Position
}

// BoolAnd represents: <left> AND <right>
type BoolAnd struct {
	Left  BoolExpr
	Right BoolExpr
	Pos   Position
}

func (b *BoolAnd) boolExpr()        {}
func (b *BoolAnd) GetPos() Position { return b.Pos }

// BoolOr represents: <left> OR <right>
type BoolOr struct {
	Left  BoolExpr
	Right BoolExpr
	Pos   Position
}

func (b *BoolOr) boolExpr()        {}
func (b *BoolOr) GetPos() Position { return b.Pos }

// BoolNot represents: NOT <expr>
type BoolNot struct {
	Expr BoolExpr
	Pos  Position
}

func (b *BoolNot) boolExpr()        {}
func (b *BoolNot) GetPos() Position { return b.Pos }

// SignalRefExpr represents a signal reference in a boolean expression.
type SignalRefExpr struct {
	SignalType string
	SignalName string
	Pos        Position
}

func (s *SignalRefExpr) boolExpr()        {}
func (s *SignalRefExpr) GetPos() Position { return s.Pos }

// ---------- Model References ----------

// ModelRef represents a model reference in a ROUTE.
type ModelRef struct {
	Model           string
	Reasoning       *bool
	Effort          string
	LoRA            string
	ParamSize       string
	Weight          float64
	ReasoningFamily string
	Pos             Position
}

// ---------- Algorithm Specification ----------

// AlgoSpec represents an ALGORITHM block in a ROUTE.
type AlgoSpec struct {
	AlgoType string
	Fields   map[string]Value
	Pos      Position
}

// ---------- Value Types ----------

// Value is the interface for all DSL value types.
type Value interface {
	value() // marker method
}

// StringValue represents a string literal.
type StringValue struct{ V string }

func (v StringValue) value() {}

// IntValue represents an integer literal.
type IntValue struct{ V int }

func (v IntValue) value() {}

// FloatValue represents a floating-point literal.
type FloatValue struct{ V float64 }

func (v FloatValue) value() {}

// BoolValue represents a boolean literal.
type BoolValue struct{ V bool }

func (v BoolValue) value() {}

// ArrayValue represents an array: [v1, v2, ...]
type ArrayValue struct{ Items []Value }

func (v ArrayValue) value() {}

// ObjectValue represents a nested object: { key: value, ... }
type ObjectValue struct{ Fields map[string]Value }

func (v ObjectValue) value() {}
