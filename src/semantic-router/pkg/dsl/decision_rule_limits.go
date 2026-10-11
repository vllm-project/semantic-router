package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ValidateProgramRuleLimits counts the lowered rule shape before the compiler's
// recursive expression lowering. Adjacent ANDs/ORs flatten into a single node.
func ValidateProgramRuleLimits(prog *Program, limits config.DecisionRuleLimits) error {
	depth, nodes, err := limits.Effective()
	if err != nil {
		return err
	}
	if prog == nil {
		return fmt.Errorf("cannot compile a nil program")
	}
	validate := func(program *Program, recipe string) error {
		for _, route := range program.Routes {
			if err := validateRouteRuleLimits(route.When, depth, nodes); err != nil {
				return fmt.Errorf("routing recipe %q: decision %q: %w", recipe, route.Name, err)
			}
		}
		return nil
	}
	if err := validate(prog, "default"); err != nil {
		return err
	}
	for _, recipe := range prog.Recipes {
		if err := validate(recipe.Program, recipe.Name); err != nil {
			return err
		}
	}
	return nil
}

type ruleExpressionCursor struct {
	pending []BoolExpr
	flatten string
	path    string
	next    int
}

func newRuleExpressionCursor(expr BoolExpr, path string) ruleExpressionCursor {
	cursor := ruleExpressionCursor{path: path}
	switch node := expr.(type) {
	case *BoolAnd:
		cursor.flatten, cursor.pending = "AND", []BoolExpr{node.Right, node.Left}
	case *BoolOr:
		cursor.flatten, cursor.pending = "OR", []BoolExpr{node.Right, node.Left}
	case *BoolNot:
		cursor.pending = []BoolExpr{node.Expr}
	}
	return cursor
}

// nextChild expands only same-operator AST nodes; those disappear on lowering.
func (cursor *ruleExpressionCursor) nextChild() (BoolExpr, bool) {
	for len(cursor.pending) > 0 {
		last := len(cursor.pending) - 1
		expr := cursor.pending[last]
		cursor.pending = cursor.pending[:last]
		switch node := expr.(type) {
		case *BoolAnd:
			if cursor.flatten == "AND" {
				cursor.pending = append(cursor.pending, node.Right, node.Left)
				continue
			}
		case *BoolOr:
			if cursor.flatten == "OR" {
				cursor.pending = append(cursor.pending, node.Right, node.Left)
				continue
			}
		}
		return expr, true
	}
	return nil, false
}

func validateRouteRuleLimits(expr BoolExpr, maxDepth, maxNodes int) error {
	root := newRuleExpressionCursor(expr, "rules")
	stack := []ruleExpressionCursor{root}
	count := 1
	for len(stack) > 0 {
		parent := &stack[len(stack)-1]
		child, ok := parent.nextChild()
		if !ok {
			stack = stack[:len(stack)-1]
			continue
		}
		path := fmt.Sprintf("%s.conditions[%d]", parent.path, parent.next)
		parent.next++
		depth := len(stack) + 1
		count++
		if depth > maxDepth {
			return fmt.Errorf("%s: depth %d exceeds max_depth=%d", path, depth, maxDepth)
		}
		if count > maxNodes {
			return fmt.Errorf("%s: node count %d exceeds max_nodes=%d", path, count, maxNodes)
		}
		stack = append(stack, newRuleExpressionCursor(child, path))
	}
	return nil
}
