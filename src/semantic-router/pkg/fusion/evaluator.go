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
	"strings"
)

// Evaluator evaluates AST nodes against a signal context
type Evaluator struct {
	ast     *ASTNode
	context *SignalContext
}

// NewEvaluator creates a new evaluator
func NewEvaluator(ast *ASTNode, ctx *SignalContext) *Evaluator {
	return &Evaluator{
		ast:     ast,
		context: ctx,
	}
}

// Evaluate evaluates the AST and returns the boolean result
func (e *Evaluator) Evaluate() (bool, error) {
	return e.evaluateNode(e.ast)
}

// evaluateNode recursively evaluates an AST node
func (e *Evaluator) evaluateNode(node *ASTNode) (bool, error) {
	if node == nil {
		return false, fmt.Errorf("nil node")
	}

	switch node.Type {
	case NodeAnd:
		return e.evaluateAnd(node)
	case NodeOr:
		return e.evaluateOr(node)
	case NodeNot:
		return e.evaluateNot(node)
	case NodeComparison:
		return e.evaluateComparison(node)
	case NodeIdentifier:
		return e.evaluateIdentifier(node)
	case NodeLiteral:
		return e.evaluateLiteral(node)
	default:
		return false, fmt.Errorf("unknown node type: %v", node.Type)
	}
}

// evaluateAnd evaluates AND with short-circuit (stops on first false)
func (e *Evaluator) evaluateAnd(node *ASTNode) (bool, error) {
	// Evaluate left side
	left, err := e.evaluateNode(node.Left)
	if err != nil {
		return false, err
	}

	// SHORT-CIRCUIT: If left is false, don't evaluate right
	if !left {
		return false, nil
	}

	// Left is true, evaluate right
	return e.evaluateNode(node.Right)
}

// evaluateOr evaluates OR with short-circuit (stops on first true)
func (e *Evaluator) evaluateOr(node *ASTNode) (bool, error) {
	// Evaluate left side
	left, err := e.evaluateNode(node.Left)
	if err != nil {
		return false, err
	}

	// SHORT-CIRCUIT: If left is true, don't evaluate right
	if left {
		return true, nil
	}

	// Left is false, evaluate right
	return e.evaluateNode(node.Right)
}

// evaluateNot evaluates NOT
func (e *Evaluator) evaluateNot(node *ASTNode) (bool, error) {
	result, err := e.evaluateNode(node.Left)
	if err != nil {
		return false, err
	}
	return !result, nil
}

// evaluateComparison evaluates comparison operators
func (e *Evaluator) evaluateComparison(node *ASTNode) (bool, error) {
	// Get left value
	leftVal, err := e.getValue(node.Left)
	if err != nil {
		return false, fmt.Errorf("left side of comparison: %w", err)
	}

	// Get right value
	rightVal, err := e.getValue(node.Right)
	if err != nil {
		return false, fmt.Errorf("right side of comparison: %w", err)
	}

	// Perform comparison based on operator
	switch node.Operator {
	case "==":
		return e.equals(leftVal, rightVal), nil
	case "!=":
		return !e.equals(leftVal, rightVal), nil
	case ">":
		return e.greaterThan(leftVal, rightVal)
	case "<":
		return e.lessThan(leftVal, rightVal)
	case ">=":
		return e.greaterThanOrEqual(leftVal, rightVal)
	case "<=":
		return e.lessThanOrEqual(leftVal, rightVal)
	default:
		return false, fmt.Errorf("unknown comparison operator: %s", node.Operator)
	}
}

// evaluateIdentifier evaluates an identifier (signal path)
func (e *Evaluator) evaluateIdentifier(node *ASTNode) (bool, error) {
	path, ok := node.Value.(string)
	if !ok {
		return false, fmt.Errorf("identifier value is not a string: %v", node.Value)
	}

	// For boolean identifiers (like keyword.k8s.matched)
	// we need to look up the value and check if it's true
	val, err := e.lookupSignal(path)
	if err != nil {
		return false, err
	}

	// If the value is a boolean, return it
	if boolVal, ok := val.(bool); ok {
		return boolVal, nil
	}

	// Otherwise, the identifier should be used in a comparison
	return false, fmt.Errorf("identifier %s is not a boolean and should be used in a comparison", path)
}

// evaluateLiteral evaluates a literal value
func (e *Evaluator) evaluateLiteral(node *ASTNode) (bool, error) {
	boolVal, ok := node.Value.(bool)
	if !ok {
		return false, fmt.Errorf("literal is not a boolean: %v", node.Value)
	}
	return boolVal, nil
}

// getValue extracts the value from a node (for use in comparisons)
func (e *Evaluator) getValue(node *ASTNode) (interface{}, error) {
	switch node.Type {
	case NodeLiteral:
		return node.Value, nil
	case NodeIdentifier:
		path, ok := node.Value.(string)
		if !ok {
			return nil, fmt.Errorf("identifier value is not a string")
		}
		return e.lookupSignal(path)
	default:
		return nil, fmt.Errorf("cannot get value from node type: %v", node.Type)
	}
}

// lookupSignal looks up a signal value from the context
// Signal paths have the format: {signal_type}.{signal_name}.{field}
// Examples:
//   - keyword.k8s.matched
//   - regex.ssn.matched
//   - similarity.reasoning.score
//   - bert.category
//   - bert.confidence
func (e *Evaluator) lookupSignal(path string) (interface{}, error) {
	parts := strings.Split(path, ".")
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid signal path (expected at least 2 parts): %s", path)
	}

	signalType := parts[0]
	signalName := parts[1]

	switch signalType {
	case "keyword":
		if len(parts) != 3 || parts[2] != "matched" {
			return nil, fmt.Errorf("keyword signal should have format 'keyword.{name}.matched': %s", path)
		}
		// Return the matched status (bool)
		matched, exists := e.context.KeywordMatches[signalName]
		if !exists {
			// If keyword rule doesn't exist in context, assume it didn't match
			return false, nil
		}
		return matched, nil

	case "regex":
		if len(parts) != 3 || parts[2] != "matched" {
			return nil, fmt.Errorf("regex signal should have format 'regex.{name}.matched': %s", path)
		}
		// Return the matched status (bool)
		matched, exists := e.context.RegexMatches[signalName]
		if !exists {
			// If regex pattern doesn't exist in context, assume it didn't match
			return false, nil
		}
		return matched, nil

	case "similarity":
		if len(parts) != 3 || parts[2] != "score" {
			return nil, fmt.Errorf("similarity signal should have format 'similarity.{name}.score': %s", path)
		}
		// Return the similarity score (float64)
		score, exists := e.context.SimilarityScores[signalName]
		if !exists {
			// If similarity concept doesn't exist in context, return 0.0
			return 0.0, nil
		}
		return score, nil

	case "bert":
		if len(parts) != 2 {
			return nil, fmt.Errorf("bert signal should have format 'bert.{field}': %s", path)
		}
		field := signalName // In this case, signalName is actually the field name

		switch field {
		case "category":
			return e.context.BERTCategory, nil
		case "confidence":
			return e.context.BERTConfidence, nil
		default:
			return nil, fmt.Errorf("unknown bert field: %s (expected 'category' or 'confidence')", field)
		}

	default:
		return nil, fmt.Errorf("unknown signal type: %s", signalType)
	}
}

// equals checks if two values are equal
func (e *Evaluator) equals(left, right interface{}) bool {
	// Handle string comparison
	leftStr, leftIsStr := left.(string)
	rightStr, rightIsStr := right.(string)
	if leftIsStr && rightIsStr {
		return leftStr == rightStr
	}

	// Handle numeric comparison
	leftNum, leftOk := e.toFloat64(left)
	rightNum, rightOk := e.toFloat64(right)
	if leftOk && rightOk {
		return leftNum == rightNum
	}

	// Handle boolean comparison
	leftBool, leftIsBool := left.(bool)
	rightBool, rightIsBool := right.(bool)
	if leftIsBool && rightIsBool {
		return leftBool == rightBool
	}

	// Types don't match or are incomparable
	return false
}

// greaterThan checks if left > right
func (e *Evaluator) greaterThan(left, right interface{}) (bool, error) {
	leftNum, leftOk := e.toFloat64(left)
	rightNum, rightOk := e.toFloat64(right)

	if !leftOk || !rightOk {
		return false, fmt.Errorf("cannot compare non-numeric values with >")
	}

	return leftNum > rightNum, nil
}

// lessThan checks if left < right
func (e *Evaluator) lessThan(left, right interface{}) (bool, error) {
	leftNum, leftOk := e.toFloat64(left)
	rightNum, rightOk := e.toFloat64(right)

	if !leftOk || !rightOk {
		return false, fmt.Errorf("cannot compare non-numeric values with <")
	}

	return leftNum < rightNum, nil
}

// greaterThanOrEqual checks if left >= right
func (e *Evaluator) greaterThanOrEqual(left, right interface{}) (bool, error) {
	leftNum, leftOk := e.toFloat64(left)
	rightNum, rightOk := e.toFloat64(right)

	if !leftOk || !rightOk {
		return false, fmt.Errorf("cannot compare non-numeric values with >=")
	}

	return leftNum >= rightNum, nil
}

// lessThanOrEqual checks if left <= right
func (e *Evaluator) lessThanOrEqual(left, right interface{}) (bool, error) {
	leftNum, leftOk := e.toFloat64(left)
	rightNum, rightOk := e.toFloat64(right)

	if !leftOk || !rightOk {
		return false, fmt.Errorf("cannot compare non-numeric values with <=")
	}

	return leftNum <= rightNum, nil
}

// toFloat64 converts various numeric types to float64
func (e *Evaluator) toFloat64(val interface{}) (float64, bool) {
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case int32:
		return float64(v), true
	default:
		return 0, false
	}
}
