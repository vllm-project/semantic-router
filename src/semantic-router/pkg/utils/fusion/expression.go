package fusion

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// ExpressionEvaluator evaluates boolean expressions against a SignalContext
type ExpressionEvaluator struct {
	context *SignalContext
}

// NewExpressionEvaluator creates a new expression evaluator
func NewExpressionEvaluator(context *SignalContext) *ExpressionEvaluator {
	return &ExpressionEvaluator{context: context}
}

// Evaluate parses and evaluates a boolean expression
// Supported operations:
// - Boolean operators: && (AND), || (OR), ! (NOT)
// - Comparisons: ==, !=, >, <, >=, <=
// - Signal references: provider.name.field (e.g., "keyword.kubernetes.matched", "similarity.reasoning.score")
func (e *ExpressionEvaluator) Evaluate(expression string) (bool, error) {
	expression = strings.TrimSpace(expression)
	if expression == "" {
		return false, fmt.Errorf("empty expression")
	}

	// Parse and evaluate the expression
	tokens, err := tokenize(expression)
	if err != nil {
		return false, err
	}

	return e.evaluateTokens(tokens)
}

// Token types
type tokenType int

const (
	tokenIdentifier tokenType = iota
	tokenNumber
	tokenString
	tokenOperator
	tokenLeftParen
	tokenRightParen
)

type token struct {
	typ   tokenType
	value string
}

// tokenize converts an expression string into tokens
func tokenize(expression string) ([]token, error) {
	var tokens []token
	var current strings.Builder
	inString := false
	stringDelim := rune(0)

	i := 0
	for i < len(expression) {
		ch := rune(expression[i])

		// Handle string literals
		if ch == '\'' || ch == '"' {
			if !inString {
				inString = true
				stringDelim = ch
				i++
				continue
			} else if ch == stringDelim {
				tokens = append(tokens, token{typ: tokenString, value: current.String()})
				current.Reset()
				inString = false
				stringDelim = 0
				i++
				continue
			}
		}

		if inString {
			current.WriteRune(ch)
			i++
			continue
		}

		// Skip whitespace
		if unicode.IsSpace(ch) {
			if current.Len() > 0 {
				tokens = append(tokens, parseToken(current.String()))
				current.Reset()
			}
			i++
			continue
		}

		// Handle operators
		if i+1 < len(expression) {
			twoChar := expression[i : i+2]
			if twoChar == "&&" || twoChar == "||" || twoChar == "==" || twoChar == "!=" || twoChar == ">=" || twoChar == "<=" {
				if current.Len() > 0 {
					tokens = append(tokens, parseToken(current.String()))
					current.Reset()
				}
				tokens = append(tokens, token{typ: tokenOperator, value: twoChar})
				i += 2
				continue
			}
		}

		// Handle single character operators
		if ch == '>' || ch == '<' || ch == '!' {
			if current.Len() > 0 {
				tokens = append(tokens, parseToken(current.String()))
				current.Reset()
			}
			tokens = append(tokens, token{typ: tokenOperator, value: string(ch)})
			i++
			continue
		}

		// Handle parentheses
		if ch == '(' {
			if current.Len() > 0 {
				tokens = append(tokens, parseToken(current.String()))
				current.Reset()
			}
			tokens = append(tokens, token{typ: tokenLeftParen, value: "("})
			i++
			continue
		}

		if ch == ')' {
			if current.Len() > 0 {
				tokens = append(tokens, parseToken(current.String()))
				current.Reset()
			}
			tokens = append(tokens, token{typ: tokenRightParen, value: ")"})
			i++
			continue
		}

		current.WriteRune(ch)
		i++
	}

	if inString {
		return nil, fmt.Errorf("unterminated string literal")
	}

	if current.Len() > 0 {
		tokens = append(tokens, parseToken(current.String()))
	}

	return tokens, nil
}

func parseToken(s string) token {
	// Check if it's a number
	if _, err := strconv.ParseFloat(s, 64); err == nil {
		return token{typ: tokenNumber, value: s}
	}
	// Otherwise it's an identifier
	return token{typ: tokenIdentifier, value: s}
}

// evaluateTokens evaluates a list of tokens
func (e *ExpressionEvaluator) evaluateTokens(tokens []token) (bool, error) {
	if len(tokens) == 0 {
		return false, fmt.Errorf("no tokens to evaluate")
	}

	// Handle parentheses first (strip outer parentheses if present)
	if len(tokens) > 0 && tokens[0].typ == tokenLeftParen && tokens[len(tokens)-1].typ == tokenRightParen {
		// Check if these are matching outer parentheses
		depth := 0
		allMatching := true
		for i := 0; i < len(tokens); i++ {
			if tokens[i].typ == tokenLeftParen {
				depth++
			} else if tokens[i].typ == tokenRightParen {
				depth--
			}
			// If depth goes to 0 before the end, these aren't outer parentheses
			if depth == 0 && i < len(tokens)-1 {
				allMatching = false
				break
			}
		}
		if allMatching {
			return e.evaluateTokens(tokens[1 : len(tokens)-1])
		}
	}

	// Handle OR operator (lowest precedence) - skip parentheses
	depth := 0
	for i := 0; i < len(tokens); i++ {
		if tokens[i].typ == tokenLeftParen {
			depth++
		} else if tokens[i].typ == tokenRightParen {
			depth--
		} else if depth == 0 && tokens[i].typ == tokenOperator && tokens[i].value == "||" {
			left, err := e.evaluateTokens(tokens[:i])
			if err != nil {
				return false, err
			}
			right, err := e.evaluateTokens(tokens[i+1:])
			if err != nil {
				return false, err
			}
			return left || right, nil
		}
	}

	// Handle AND operator (higher precedence than OR) - skip parentheses
	depth = 0
	for i := 0; i < len(tokens); i++ {
		if tokens[i].typ == tokenLeftParen {
			depth++
		} else if tokens[i].typ == tokenRightParen {
			depth--
		} else if depth == 0 && tokens[i].typ == tokenOperator && tokens[i].value == "&&" {
			left, err := e.evaluateTokens(tokens[:i])
			if err != nil {
				return false, err
			}
			right, err := e.evaluateTokens(tokens[i+1:])
			if err != nil {
				return false, err
			}
			return left && right, nil
		}
	}

	// Handle NOT operator
	if len(tokens) > 0 && tokens[0].typ == tokenOperator && tokens[0].value == "!" {
		result, err := e.evaluateTokens(tokens[1:])
		if err != nil {
			return false, err
		}
		return !result, nil
	}

	// Handle comparison operators - skip parentheses
	depth = 0
	for i := 0; i < len(tokens); i++ {
		if tokens[i].typ == tokenLeftParen {
			depth++
		} else if tokens[i].typ == tokenRightParen {
			depth--
		} else if depth == 0 && tokens[i].typ == tokenOperator {
			op := tokens[i].value
			if op == "==" || op == "!=" || op == ">" || op == "<" || op == ">=" || op == "<=" {
				if i == 0 || i == len(tokens)-1 {
					return false, fmt.Errorf("invalid comparison expression")
				}
				return e.evaluateComparison(tokens[:i], op, tokens[i+1:])
			}
		}
	}

	// Handle single boolean value (signal reference)
	if len(tokens) == 1 && tokens[0].typ == tokenIdentifier {
		return e.evaluateSignalReference(tokens[0].value)
	}

	return false, fmt.Errorf("unable to evaluate expression")
}

// evaluateComparison evaluates a comparison between two values
func (e *ExpressionEvaluator) evaluateComparison(leftTokens []token, op string, rightTokens []token) (bool, error) {
	leftVal, err := e.getComparisonValue(leftTokens)
	if err != nil {
		return false, err
	}

	rightVal, err := e.getComparisonValue(rightTokens)
	if err != nil {
		return false, err
	}

	// Try numeric comparison first
	leftNum, leftIsNum := leftVal.(float64)
	rightNum, rightIsNum := rightVal.(float64)

	if leftIsNum && rightIsNum {
		switch op {
		case "==":
			return leftNum == rightNum, nil
		case "!=":
			return leftNum != rightNum, nil
		case ">":
			return leftNum > rightNum, nil
		case "<":
			return leftNum < rightNum, nil
		case ">=":
			return leftNum >= rightNum, nil
		case "<=":
			return leftNum <= rightNum, nil
		}
	}

	// Fall back to string comparison
	leftStr := fmt.Sprint(leftVal)
	rightStr := fmt.Sprint(rightVal)

	switch op {
	case "==":
		return leftStr == rightStr, nil
	case "!=":
		return leftStr != rightStr, nil
	default:
		return false, fmt.Errorf("operator %s not supported for string comparison", op)
	}
}

// getComparisonValue extracts a value from tokens for comparison
func (e *ExpressionEvaluator) getComparisonValue(tokens []token) (interface{}, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty comparison value")
	}

	if len(tokens) == 1 {
		switch tokens[0].typ {
		case tokenNumber:
			return strconv.ParseFloat(tokens[0].value, 64)
		case tokenString:
			return tokens[0].value, nil
		case tokenIdentifier:
			return e.getSignalValue(tokens[0].value)
		}
	}

	return nil, fmt.Errorf("complex comparison values not supported")
}

// evaluateSignalReference evaluates a signal reference as a boolean
func (e *ExpressionEvaluator) evaluateSignalReference(ref string) (bool, error) {
	val, err := e.getSignalValue(ref)
	if err != nil {
		return false, err
	}

	// Convert to boolean
	if b, ok := val.(bool); ok {
		return b, nil
	}

	// Treat non-zero numbers as true
	if f, ok := val.(float64); ok {
		return f != 0, nil
	}

	// Treat non-empty strings as true
	if s, ok := val.(string); ok {
		return s != "", nil
	}

	return false, fmt.Errorf("cannot convert signal value to boolean")
}

// getSignalValue retrieves a value from a signal reference
// Format: provider.name.field (e.g., "keyword.kubernetes.matched", "similarity.reasoning.score")
func (e *ExpressionEvaluator) getSignalValue(ref string) (interface{}, error) {
	parts := strings.Split(ref, ".")
	if len(parts) < 3 {
		return nil, fmt.Errorf("invalid signal reference: %s (expected format: provider.name.field)", ref)
	}

	provider := parts[0]
	name := parts[1]
	field := parts[2]

	// Create signal key
	key := provider + "." + name

	signal, exists := e.context.Signals[key]
	if !exists {
		// Signal not found - treat as false/zero
		switch field {
		case "matched":
			return false, nil
		case "score":
			return 0.0, nil
		case "value":
			return "", nil
		default:
			return nil, fmt.Errorf("unknown field: %s", field)
		}
	}

	// Extract the requested field
	switch field {
	case "matched":
		return signal.Matched, nil
	case "score":
		return signal.Score, nil
	case "value":
		return signal.Value, nil
	default:
		return nil, fmt.Errorf("unknown field: %s", field)
	}
}
