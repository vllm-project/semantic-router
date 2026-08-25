package dsl

import (
	"fmt"
	"strings"

	"github.com/alecthomas/participle/v2/lexer"
)

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
	TOKEN_PRIORITY
	TOKEN_WHEN
	TOKEN_MODEL
	TOKEN_ALGORITHM
	TOKEN_FOR
	TOKEN_IN
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
		TOKEN_ROUTE: "ROUTE", TOKEN_PLUGIN: "PLUGIN", TOKEN_PRIORITY: "PRIORITY",
		TOKEN_WHEN: "WHEN", TOKEN_MODEL: "MODEL", TOKEN_ALGORITHM: "ALGORITHM",
		TOKEN_FOR: "FOR", TOKEN_IN: "IN", TOKEN_AND: "AND", TOKEN_OR: "OR", TOKEN_NOT: "NOT",
		TOKEN_LBRACE: "{", TOKEN_RBRACE: "}",
		TOKEN_LPAREN: "(", TOKEN_RPAREN: ")", TOKEN_LBRACKET: "[",
		TOKEN_RBRACKET: "]", TOKEN_COLON: ":", TOKEN_COMMA: ",", TOKEN_EQUALS: "=",
	}
	if name, ok := names[t]; ok {
		return name
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

// Lex tokenizes DSL source for syntax tooling and diagnostics.
func Lex(input string) ([]Token, []error) {
	stream, err := dslLexer.Lex("", strings.NewReader(input))
	if err != nil {
		return nil, []error{err}
	}
	punctuation, symbols := tokenPunctuationTypes()
	var tokens []Token
	for {
		token, nextErr := stream.Next()
		if nextErr != nil {
			return nil, []error{nextErr}
		}
		pos := Position{Line: token.Pos.Line, Column: token.Pos.Column}
		switch token.Type {
		case symbols["EOF"]:
			return append(tokens, Token{Type: TOKEN_EOF, Pos: pos}), nil
		case symbols["Whitespace"], symbols["Comment"]:
			continue
		case symbols["Ident"]:
			tokens = append(tokens, Token{Type: LookupIdent(token.Value), Literal: token.Value, Pos: pos})
		case symbols["String"]:
			tokens = append(tokens, Token{Type: TOKEN_STRING, Literal: unquote(token.Value), Pos: pos})
		case symbols["Int"]:
			tokens = append(tokens, Token{Type: TOKEN_INT, Literal: token.Value, Pos: pos})
		case symbols["Float"]:
			tokens = append(tokens, Token{Type: TOKEN_FLOAT, Literal: token.Value, Pos: pos})
		default:
			if tokenType, ok := punctuation[token.Type]; ok {
				tokens = append(tokens, Token{Type: tokenType, Literal: token.Value, Pos: pos})
			}
		}
	}
}

func tokenPunctuationTypes() (map[lexer.TokenType]TokenType, map[string]lexer.TokenType) {
	symbols := dslLexer.Symbols()
	return map[lexer.TokenType]TokenType{
		symbols["LBrace"]: TOKEN_LBRACE, symbols["RBrace"]: TOKEN_RBRACE,
		symbols["LParen"]: TOKEN_LPAREN, symbols["RParen"]: TOKEN_RPAREN,
		symbols["LBracket"]: TOKEN_LBRACKET, symbols["RBracket"]: TOKEN_RBRACKET,
		symbols["Colon"]: TOKEN_COLON, symbols["Comma"]: TOKEN_COMMA,
		symbols["Equals"]: TOKEN_EQUALS,
	}, symbols
}

func tokenKeywordTypes() map[string]TokenType {
	return map[string]TokenType{
		"SIGNAL": TOKEN_SIGNAL, "ROUTE": TOKEN_ROUTE,
		"PLUGIN": TOKEN_PLUGIN, "TEST": TOKEN_IDENT,
		"PRIORITY": TOKEN_PRIORITY, "TIER": TOKEN_IDENT, "WHEN": TOKEN_WHEN,
		"MODEL": TOKEN_MODEL, "ALGORITHM": TOKEN_ALGORITHM,
		"FOR": TOKEN_FOR, "IN": TOKEN_IN,
		"AND": TOKEN_AND, "OR": TOKEN_OR, "NOT": TOKEN_NOT,
		"true": TOKEN_BOOL, "false": TOKEN_BOOL,
	}
}

// LookupIdent returns the token type for an identifier string.
func LookupIdent(ident string) TokenType {
	if tokenType, ok := tokenKeywordTypes()[ident]; ok {
		return tokenType
	}
	return TOKEN_IDENT
}

// isIdentPart reports whether ch is valid in a DSL identifier.
func isIdentPart(ch rune) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
		(ch >= '0' && ch <= '9') || ch == '_' || ch == '-' || ch == '.' || ch == '/'
}
