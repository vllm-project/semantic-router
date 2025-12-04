/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fusion

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// Parser parses boolean expressions into Abstract Syntax Trees
type Parser struct {
	tokens []Token
	pos    int
}

// NewParser creates a new expression parser
func NewParser() *Parser {
	return &Parser{}
}

// Parse parses an expression string into an AST
func (p *Parser) Parse(expression string) (*ASTNode, error) {
	// Tokenize
	tokens, err := p.tokenize(expression)
	if err != nil {
		return nil, err
	}

	p.tokens = tokens
	p.pos = 0

	// Parse into AST
	ast, err := p.parseOr()
	if err != nil {
		return nil, err
	}

	// Ensure we consumed all tokens
	if p.pos < len(p.tokens) && p.tokens[p.pos].Type != TokenEOF {
		return nil, fmt.Errorf("unexpected token at position %d: %s", p.pos, p.tokens[p.pos].Value)
	}

	return ast, nil
}

// tokenize converts expression string into tokens
func (p *Parser) tokenize(expression string) ([]Token, error) {
	var tokens []Token
	i := 0

	for i < len(expression) {
		// Skip whitespace
		if unicode.IsSpace(rune(expression[i])) {
			i++
			continue
		}

		// Two-character operators
		if i+1 < len(expression) {
			twoChar := expression[i : i+2]
			switch twoChar {
			case "&&":
				tokens = append(tokens, Token{Type: TokenAnd, Value: "&&"})
				i += 2
				continue
			case "||":
				tokens = append(tokens, Token{Type: TokenOr, Value: "||"})
				i += 2
				continue
			case "==":
				tokens = append(tokens, Token{Type: TokenEQ, Value: "=="})
				i += 2
				continue
			case "!=":
				tokens = append(tokens, Token{Type: TokenNE, Value: "!="})
				i += 2
				continue
			case ">=":
				tokens = append(tokens, Token{Type: TokenGE, Value: ">="})
				i += 2
				continue
			case "<=":
				tokens = append(tokens, Token{Type: TokenLE, Value: "<="})
				i += 2
				continue
			}
		}

		// Single-character operators
		switch expression[i] {
		case '(':
			tokens = append(tokens, Token{Type: TokenLParen, Value: "("})
			i++
			continue
		case ')':
			tokens = append(tokens, Token{Type: TokenRParen, Value: ")"})
			i++
			continue
		case '!':
			tokens = append(tokens, Token{Type: TokenNot, Value: "!"})
			i++
			continue
		case '>':
			tokens = append(tokens, Token{Type: TokenGT, Value: ">"})
			i++
			continue
		case '<':
			tokens = append(tokens, Token{Type: TokenLT, Value: "<"})
			i++
			continue
		}

		// String literals
		if expression[i] == '"' || expression[i] == '\'' {
			quote := expression[i]
			i++
			start := i
			for i < len(expression) && expression[i] != quote {
				i++
			}
			if i >= len(expression) {
				return nil, fmt.Errorf("unterminated string at position %d", start-1)
			}
			tokens = append(tokens, Token{Type: TokenString, Value: expression[start:i]})
			i++ // Skip closing quote
			continue
		}

		// Numbers
		if unicode.IsDigit(rune(expression[i])) {
			start := i
			for i < len(expression) && (unicode.IsDigit(rune(expression[i])) || expression[i] == '.') {
				i++
			}
			tokens = append(tokens, Token{Type: TokenNumber, Value: expression[start:i]})
			continue
		}

		// Identifiers and keywords
		if unicode.IsLetter(rune(expression[i])) || expression[i] == '_' {
			start := i
			for i < len(expression) && (unicode.IsLetter(rune(expression[i])) || unicode.IsDigit(rune(expression[i])) || expression[i] == '_' || expression[i] == '.') {
				i++
			}
			value := expression[start:i]

			// Check for boolean keywords
			if value == "true" {
				tokens = append(tokens, Token{Type: TokenTrue, Value: "true"})
			} else if value == "false" {
				tokens = append(tokens, Token{Type: TokenFalse, Value: "false"})
			} else {
				tokens = append(tokens, Token{Type: TokenIdentifier, Value: value})
			}
			continue
		}

		return nil, fmt.Errorf("unexpected character at position %d: %c", i, expression[i])
	}

	tokens = append(tokens, Token{Type: TokenEOF, Value: ""})
	return tokens, nil
}

// parseOr parses OR expressions (lowest precedence)
func (p *Parser) parseOr() (*ASTNode, error) {
	left, err := p.parseAnd()
	if err != nil {
		return nil, err
	}

	for p.match(TokenOr) {
		right, err := p.parseAnd()
		if err != nil {
			return nil, err
		}
		left = &ASTNode{
			Type:  NodeOr,
			Left:  left,
			Right: right,
		}
	}

	return left, nil
}

// parseAnd parses AND expressions
func (p *Parser) parseAnd() (*ASTNode, error) {
	left, err := p.parseNot()
	if err != nil {
		return nil, err
	}

	for p.match(TokenAnd) {
		right, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		left = &ASTNode{
			Type:  NodeAnd,
			Left:  left,
			Right: right,
		}
	}

	return left, nil
}

// parseNot parses NOT expressions
func (p *Parser) parseNot() (*ASTNode, error) {
	if p.match(TokenNot) {
		operand, err := p.parseNot()
		if err != nil {
			return nil, err
		}
		return &ASTNode{
			Type: NodeNot,
			Left: operand,
		}, nil
	}

	return p.parseComparison()
}

// parseComparison parses comparison expressions
func (p *Parser) parseComparison() (*ASTNode, error) {
	left, err := p.parsePrimary()
	if err != nil {
		return nil, err
	}

	// Check for comparison operators
	if p.match(TokenEQ, TokenNE, TokenGT, TokenLT, TokenGE, TokenLE) {
		operator := p.previous().Value
		right, err := p.parsePrimary()
		if err != nil {
			return nil, err
		}
		return &ASTNode{
			Type:     NodeComparison,
			Operator: operator,
			Left:     left,
			Right:    right,
		}, nil
	}

	return left, nil
}

// parsePrimary parses primary expressions (literals, identifiers, parentheses)
func (p *Parser) parsePrimary() (*ASTNode, error) {
	// Parentheses
	if p.match(TokenLParen) {
		expr, err := p.parseOr()
		if err != nil {
			return nil, err
		}
		if !p.match(TokenRParen) {
			return nil, fmt.Errorf("expected ')' at position %d", p.pos)
		}
		return expr, nil
	}

	// Boolean literals
	if p.match(TokenTrue) {
		return &ASTNode{Type: NodeLiteral, Value: true}, nil
	}
	if p.match(TokenFalse) {
		return &ASTNode{Type: NodeLiteral, Value: false}, nil
	}

	// String literals
	if p.match(TokenString) {
		return &ASTNode{Type: NodeLiteral, Value: p.previous().Value}, nil
	}

	// Number literals
	if p.match(TokenNumber) {
		numStr := p.previous().Value
		if strings.Contains(numStr, ".") {
			num, err := strconv.ParseFloat(numStr, 64)
			if err != nil {
				return nil, fmt.Errorf("invalid number: %s", numStr)
			}
			return &ASTNode{Type: NodeLiteral, Value: num}, nil
		}
		num, err := strconv.ParseInt(numStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number: %s", numStr)
		}
		return &ASTNode{Type: NodeLiteral, Value: num}, nil
	}

	// Identifiers
	if p.match(TokenIdentifier) {
		return &ASTNode{Type: NodeIdentifier, Value: p.previous().Value}, nil
	}

	if p.pos < len(p.tokens) {
		return nil, fmt.Errorf("unexpected token at position %d: %s", p.pos, p.tokens[p.pos].Value)
	}
	return nil, fmt.Errorf("unexpected end of expression")
}

// match checks if current token matches any of the given types
func (p *Parser) match(types ...TokenType) bool {
	for _, t := range types {
		if p.check(t) {
			p.advance()
			return true
		}
	}
	return false
}

// check returns true if current token matches the given type
func (p *Parser) check(t TokenType) bool {
	if p.pos >= len(p.tokens) {
		return false
	}
	return p.tokens[p.pos].Type == t
}

// advance moves to the next token
func (p *Parser) advance() Token {
	if p.pos < len(p.tokens) {
		p.pos++
	}
	return p.previous()
}

// previous returns the previous token
func (p *Parser) previous() Token {
	return p.tokens[p.pos-1]
}
