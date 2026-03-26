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
	Pos          lexer.Position
	Signal       *rawSignalDecl       `parser:"  @@"`
	Projection   *rawProjectionDecl   `parser:"| @@"`
	Route        *rawRouteDecl        `parser:"| @@"`
	DecisionTree *rawDecisionTreeDecl `parser:"| @@"`
	Model        *rawModelDecl        `parser:"| @@"`
	Plugin       *rawPluginDecl       `parser:"| @@"`
	TestBlock    *rawTestBlockDecl    `parser:"| @@"`
}

// rawTestBlockDecl: TEST <name> { entries... }
type rawTestBlockDecl struct {
	Pos     lexer.Position
	Name    string              `parser:"'TEST' @(Ident | String)"`
	Entries []*rawTestEntryDecl `parser:"'{' @@* '}'"`
}

// rawTestEntryDecl: "query" -> route_name
type rawTestEntryDecl struct {
	Pos       lexer.Position
	Query     string `parser:"@String"`
	RouteName string `parser:"Arrow @(Ident | String)"`
}

// rawSignalDecl: SIGNAL <type> <name> { fields... }
type rawSignalDecl struct {
	Pos        lexer.Position
	SignalType string        `parser:"'SIGNAL' @Ident"`
	Name       string        `parser:"@(Ident | String)"`
	Fields     []*FieldEntry `parser:"'{' @@* '}'"`
}

// rawProjectionDecl: PROJECTION <partition|score|mapping> <name> { fields... }
type rawProjectionDecl struct {
	Pos    lexer.Position
	Kind   string        `parser:"'PROJECTION' @('partition' | 'score' | 'mapping')"`
	Name   string        `parser:"@(Ident | String)"`
	Fields []*FieldEntry `parser:"'{' @@* '}'"`
}

// rawRouteDecl: ROUTE <name> (opts...) { body... }
type rawRouteDecl struct {
	Pos  lexer.Position
	Name string          `parser:"'ROUTE' @(Ident | String)"`
	Opts []*RouteOpt     `parser:"( '(' @@* ')' )?"`
	Body []*rawRouteItem `parser:"'{' @@* '}'"`
}

// rawDecisionTreeDecl: DECISION_TREE <name> { IF ... ELSE IF ... ELSE ... }
type rawDecisionTreeDecl struct {
	Pos     lexer.Position
	Name    string                     `parser:"'DECISION_TREE' @(Ident | String)"`
	If      *rawDecisionTreeIfBranch   `parser:"'{' @@"`
	ElseIfs []*rawDecisionTreeIfBranch `parser:"@@*"`
	Else    *rawDecisionTreeElseBranch `parser:"@@ '}'"`
}

// rawDecisionTreeIfBranch represents IF/ELSE IF branches inside DECISION_TREE.
type rawDecisionTreeIfBranch struct {
	Pos       lexer.Position
	Condition *BoolExprTop           `parser:"( 'IF' | 'ELSE' 'IF' ) @@"`
	Body      []*rawDecisionTreeItem `parser:"'{' @@* '}'"`
}

// rawDecisionTreeElseBranch represents the terminal ELSE branch.
type rawDecisionTreeElseBranch struct {
	Pos  lexer.Position
	Body []*rawDecisionTreeItem `parser:"'ELSE' '{' @@* '}'"`
}

// rawDecisionTreeItem is a route-like statement inside one DECISION_TREE branch.
type rawDecisionTreeItem struct {
	Pos         lexer.Position
	Name        *string       `parser:"  'NAME' @(Ident | String)"`
	Description *string       `parser:"| 'DESCRIPTION' @String"`
	Tier        *int          `parser:"| 'TIER' @Int"`
	Model       *rawModelList `parser:"| 'MODEL' @@"`
	Algorithm   *rawAlgoSpec  `parser:"| 'ALGORITHM' @@"`
	Plugin      *rawPluginRef `parser:"| 'PLUGIN' @@"`
}

// rawModelDecl: MODEL <name> { fields... }
type rawModelDecl struct {
	Pos    lexer.Position
	Name   string        `parser:"'MODEL' @(Ident | String)"`
	Fields []*FieldEntry `parser:"'{' @@* '}'"`
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
	Tier      *int          `parser:"| 'TIER' @Int"`
	ToolScope *string       `parser:"| 'TOOL_SCOPE' @String"`
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
	Signals              []*SignalDecl
	ProjectionPartitions []*ProjectionPartitionDecl
	ProjectionScores     []*ProjectionScoreDecl
	ProjectionMappings   []*ProjectionMappingDecl
	Routes               []*RouteDecl
	Models               []*ModelDecl
	Plugins              []*PluginDecl
	TestBlocks           []*TestBlockDecl
}

// ProjectionPartitionDecl declares a mutually exclusive partition of signals.
// When Semantics is "softmax_exclusive", the runtime applies Voronoi
// normalization so that at most one member fires per query.
type ProjectionPartitionDecl struct {
	Name        string
	Semantics   string
	Temperature float64
	Members     []string
	Default     string
	Pos         Position
}

// ProjectionScoreDecl defines a weighted derived score over base signals.
type ProjectionScoreDecl struct {
	Name   string
	Method string
	Inputs []*ProjectionScoreInputDecl
	Pos    Position
}

// ProjectionScoreInputDecl is one weighted contribution to a projection score.
type ProjectionScoreInputDecl struct {
	SignalType  string
	SignalName  string
	KB          string
	Metric      string
	Weight      float64
	ValueSource string
	Match       float64
	Miss        float64
}

// ProjectionMappingDecl maps one score into named routing outputs.
type ProjectionMappingDecl struct {
	Name        string
	Source      string
	Method      string
	Calibration *ProjectionMappingCalibrationDecl
	Outputs     []*ProjectionMappingOutputDecl
	Pos         Position
}

// ProjectionMappingCalibrationDecl controls confidence derived from the mapped band.
type ProjectionMappingCalibrationDecl struct {
	Method string
	Slope  float64
}

// ProjectionMappingOutputDecl defines one threshold band emitted by a mapping.
type ProjectionMappingOutputDecl struct {
	Name string
	LT   *float64
	LTE  *float64
	GT   *float64
	GTE  *float64
}

// TestBlockDecl declares expected routing outcomes for specific queries.
// Native validation paths can run them through the routing signal pipeline
// to catch conflicts that static checks cannot find.
type TestBlockDecl struct {
	Name    string
	Entries []*TestEntry
	Pos     Position
}

// TestEntry is a single query → expected route mapping inside a TEST block.
type TestEntry struct {
	Query     string
	RouteName string
	Pos       Position
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
	Tier        int
	ToolScope   string
	When        BoolExpr
	Models      []*ModelRef
	Algorithm   *AlgoSpec
	Plugins     []*PluginRef
	Pos         Position
}

// ModelDecl represents a top-level model catalog entry.
type ModelDecl struct {
	Name   string
	Fields map[string]Value
	Pos    Position
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
	Model     string
	Reasoning *bool
	Effort    string
	LoRA      string
	ParamSize string
	Weight    float64
	Pos       Position
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
