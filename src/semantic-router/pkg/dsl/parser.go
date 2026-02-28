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
	{Name: "Ident", Pattern: `[a-zA-Z_][a-zA-Z0-9_\-\.\/]*`},
	{Name: "LBrace", Pattern: `\{`},
	{Name: "RBrace", Pattern: `\}`},
	{Name: "LParen", Pattern: `\(`},
	{Name: "RParen", Pattern: `\)`},
	{Name: "LBracket", Pattern: `\[`},
	{Name: "RBracket", Pattern: `\]`},
	{Name: "Colon", Pattern: `:`},
	{Name: "Comma", Pattern: `,`},
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
		return rawToProgram(raw), nil
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
		resolved := rawToProgram(r)
		prog.Signals = append(prog.Signals, resolved.Signals...)
		prog.Routes = append(prog.Routes, resolved.Routes...)
		prog.Plugins = append(prog.Plugins, resolved.Plugins...)
		prog.Backends = append(prog.Backends, resolved.Backends...)
		if resolved.Global != nil {
			prog.Global = resolved.Global
		}
	}

	if !parsedAny {
		return nil, []error{err}
	}
	return prog, allErrors
}

// splitTopLevelBlocks splits DSL source into top-level blocks by finding
// top-level keywords (SIGNAL, ROUTE, PLUGIN, BACKEND, GLOBAL) that appear
// outside of braces.
func splitTopLevelBlocks(input string) []string {
	var blocks []string
	depth := 0
	start := 0
	keywords := []string{"SIGNAL", "ROUTE", "PLUGIN", "BACKEND", "GLOBAL"}

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

func rawToProgram(raw *rawProgram) *Program {
	prog := &Program{}
	for _, entry := range raw.Entries {
		switch {
		case entry.Signal != nil:
			prog.Signals = append(prog.Signals, rawToSignal(entry.Signal))
		case entry.Route != nil:
			prog.Routes = append(prog.Routes, rawToRoute(entry.Route))
		case entry.Plugin != nil:
			prog.Plugins = append(prog.Plugins, rawToPlugin(entry.Plugin))
		case entry.Backend != nil:
			prog.Backends = append(prog.Backends, rawToBackend(entry.Backend))
		case entry.Global != nil:
			prog.Global = rawToGlobal(entry.Global)
		}
	}
	return prog
}

func rawToSignal(r *rawSignalDecl) *SignalDecl {
	return &SignalDecl{
		SignalType: r.SignalType,
		Name:       unquoteIdent(r.Name),
		Fields:     entriesToMap(r.Fields),
		Pos:        posFromLexer(r.Pos),
	}
}

func rawToRoute(r *rawRouteDecl) *RouteDecl {
	route := &RouteDecl{
		Name: unquoteIdent(r.Name),
		Pos:  posFromLexer(r.Pos),
	}

	// Process options
	for _, opt := range r.Opts {
		if opt.Key == "description" && opt.Value != nil && opt.Value.Str != nil {
			route.Description = unquote(*opt.Value.Str)
		}
	}

	// Process body items
	for _, item := range r.Body {
		switch {
		case item.Priority != nil:
			route.Priority = *item.Priority
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
		}
	}

	return route
}

func rawToPlugin(r *rawPluginDecl) *PluginDecl {
	return &PluginDecl{
		Name:       unquoteIdent(r.Name),
		PluginType: r.PluginType,
		Fields:     entriesToMap(r.Fields),
		Pos:        posFromLexer(r.Pos),
	}
}

func rawToPluginRef(r *rawPluginRef) *PluginRef {
	ref := &PluginRef{
		Name: unquoteIdent(r.Name),
		Pos:  posFromLexer(r.Pos),
	}
	if len(r.Fields) > 0 {
		ref.Fields = entriesToMap(r.Fields)
	}
	return ref
}

func rawToBackend(r *rawBackendDecl) *BackendDecl {
	return &BackendDecl{
		BackendType: r.BackendType,
		Name:        unquoteIdent(r.Name),
		Fields:      entriesToMap(r.Fields),
		Pos:         posFromLexer(r.Pos),
	}
}

func rawToGlobal(r *rawGlobalDecl) *GlobalDecl {
	return &GlobalDecl{
		Fields: entriesToMap(r.Fields),
		Pos:    posFromLexer(r.Pos),
	}
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
		case "reasoning_family":
			if v.Str != nil {
				m.ReasoningFamily = unquote(*v.Str)
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

// ---------- Legacy Token/Lex Compatibility ----------

// TokenType represents the type of a lexical token.
type TokenType int

const (
	TOKEN_EOF TokenType = iota
	TOKEN_ILLEGAL
	TOKEN_IDENT
	TOKEN_STRING
	TOKEN_INT
	TOKEN_FLOAT
	TOKEN_BOOL
	TOKEN_COMMENT
	TOKEN_SIGNAL
	TOKEN_ROUTE
	TOKEN_PLUGIN
	TOKEN_BACKEND
	TOKEN_GLOBAL
	TOKEN_PRIORITY
	TOKEN_WHEN
	TOKEN_MODEL
	TOKEN_ALGORITHM
	TOKEN_AND
	TOKEN_OR
	TOKEN_NOT
	TOKEN_LBRACE
	TOKEN_RBRACE
	TOKEN_LPAREN
	TOKEN_RPAREN
	TOKEN_LBRACKET
	TOKEN_RBRACKET
	TOKEN_COLON
	TOKEN_COMMA
	TOKEN_EQUALS
)

func (t TokenType) String() string {
	names := map[TokenType]string{
		TOKEN_EOF: "EOF", TOKEN_ILLEGAL: "ILLEGAL", TOKEN_IDENT: "IDENT",
		TOKEN_STRING: "STRING", TOKEN_INT: "INT", TOKEN_FLOAT: "FLOAT",
		TOKEN_BOOL: "BOOL", TOKEN_COMMENT: "COMMENT", TOKEN_SIGNAL: "SIGNAL",
		TOKEN_ROUTE: "ROUTE", TOKEN_PLUGIN: "PLUGIN", TOKEN_BACKEND: "BACKEND",
		TOKEN_GLOBAL: "GLOBAL", TOKEN_PRIORITY: "PRIORITY", TOKEN_WHEN: "WHEN",
		TOKEN_MODEL: "MODEL", TOKEN_ALGORITHM: "ALGORITHM", TOKEN_AND: "AND",
		TOKEN_OR: "OR", TOKEN_NOT: "NOT", TOKEN_LBRACE: "{", TOKEN_RBRACE: "}",
		TOKEN_LPAREN: "(", TOKEN_RPAREN: ")", TOKEN_LBRACKET: "[",
		TOKEN_RBRACKET: "]", TOKEN_COLON: ":", TOKEN_COMMA: ",", TOKEN_EQUALS: "=",
	}
	if s, ok := names[t]; ok {
		return s
	}
	return fmt.Sprintf("TokenType(%d)", t)
}

// Token represents a single lexical token.
type Token struct {
	Type    TokenType
	Literal string
	Pos     Position
}

func (t Token) String() string {
	return fmt.Sprintf("Token(%s, %q, %s)", t.Type, t.Literal, t.Pos)
}

// Lex tokenizes the DSL source (backward compatibility for validator).
func Lex(input string) ([]Token, []error) {
	lex, err := dslLexer.Lex("", strings.NewReader(input))
	if err != nil {
		return nil, []error{err}
	}

	keywordMap := map[string]TokenType{
		"SIGNAL": TOKEN_SIGNAL, "ROUTE": TOKEN_ROUTE, "PLUGIN": TOKEN_PLUGIN,
		"BACKEND": TOKEN_BACKEND, "GLOBAL": TOKEN_GLOBAL, "PRIORITY": TOKEN_PRIORITY,
		"WHEN": TOKEN_WHEN, "MODEL": TOKEN_MODEL, "ALGORITHM": TOKEN_ALGORITHM,
		"AND": TOKEN_AND, "OR": TOKEN_OR, "NOT": TOKEN_NOT,
		"true": TOKEN_BOOL, "false": TOKEN_BOOL,
	}
	punctMap := map[lexer.TokenType]TokenType{}

	symMap := dslLexer.Symbols()
	identSym := symMap["Ident"]
	stringSym := symMap["String"]
	intSym := symMap["Int"]
	floatSym := symMap["Float"]
	eofSym := symMap["EOF"]
	lbraceSym := symMap["LBrace"]
	rbraceSym := symMap["RBrace"]
	lparenSym := symMap["LParen"]
	rparenSym := symMap["RParen"]
	lbracketSym := symMap["LBracket"]
	rbracketSym := symMap["RBracket"]
	colonSym := symMap["Colon"]
	commaSym := symMap["Comma"]
	equalsSym := symMap["Equals"]
	commentSym := symMap["Comment"]
	wsSym := symMap["Whitespace"]

	punctMap[lbraceSym] = TOKEN_LBRACE
	punctMap[rbraceSym] = TOKEN_RBRACE
	punctMap[lparenSym] = TOKEN_LPAREN
	punctMap[rparenSym] = TOKEN_RPAREN
	punctMap[lbracketSym] = TOKEN_LBRACKET
	punctMap[rbracketSym] = TOKEN_RBRACKET
	punctMap[colonSym] = TOKEN_COLON
	punctMap[commaSym] = TOKEN_COMMA
	punctMap[equalsSym] = TOKEN_EQUALS

	var tokens []Token
	for {
		t, err := lex.Next()
		if err != nil {
			return nil, []error{err}
		}

		pos := Position{Line: t.Pos.Line, Column: t.Pos.Column}

		switch t.Type {
		case eofSym:
			tokens = append(tokens, Token{Type: TOKEN_EOF, Literal: "", Pos: pos})
			return tokens, nil
		case wsSym, commentSym:
			continue
		case identSym:
			if kwType, ok := keywordMap[t.Value]; ok {
				tokens = append(tokens, Token{Type: kwType, Literal: t.Value, Pos: pos})
			} else {
				tokens = append(tokens, Token{Type: TOKEN_IDENT, Literal: t.Value, Pos: pos})
			}
		case stringSym:
			tokens = append(tokens, Token{Type: TOKEN_STRING, Literal: unquote(t.Value), Pos: pos})
		case intSym:
			tokens = append(tokens, Token{Type: TOKEN_INT, Literal: t.Value, Pos: pos})
		case floatSym:
			tokens = append(tokens, Token{Type: TOKEN_FLOAT, Literal: t.Value, Pos: pos})
		default:
			if pType, ok := punctMap[t.Type]; ok {
				tokens = append(tokens, Token{Type: pType, Literal: t.Value, Pos: pos})
			}
		}
	}
}

// LookupIdent returns the token type for an identifier string.
func LookupIdent(ident string) TokenType {
	keywords := map[string]TokenType{
		"SIGNAL": TOKEN_SIGNAL, "ROUTE": TOKEN_ROUTE, "PLUGIN": TOKEN_PLUGIN,
		"BACKEND": TOKEN_BACKEND, "GLOBAL": TOKEN_GLOBAL, "PRIORITY": TOKEN_PRIORITY,
		"WHEN": TOKEN_WHEN, "MODEL": TOKEN_MODEL, "ALGORITHM": TOKEN_ALGORITHM,
		"AND": TOKEN_AND, "OR": TOKEN_OR, "NOT": TOKEN_NOT,
		"true": TOKEN_BOOL, "false": TOKEN_BOOL,
	}
	if tok, ok := keywords[ident]; ok {
		return tok
	}
	return TOKEN_IDENT
}

// isIdentPart returns true if ch is valid in a DSL identifier.
func isIdentPart(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' || ch == '/'
}
