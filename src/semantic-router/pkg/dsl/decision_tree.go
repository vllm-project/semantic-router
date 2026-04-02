package dsl

import "fmt"

const (
	decisionTreePriorityBase = 1_000_000
	decisionTreePriorityStep = 1_000
)

type decisionTreeBranch struct {
	pos       Position
	condition BoolExpr
	body      []*rawDecisionTreeItem
}

func rawDecisionTreeToRoutes(tree *rawDecisionTreeDecl, treeIndex int) ([]*RouteDecl, []error) {
	if tree == nil {
		return nil, nil
	}

	branches := make([]decisionTreeBranch, 0, 1+len(tree.ElseIfs)+1)
	branches = append(branches, decisionTreeBranch{
		pos:       posFromLexer(tree.If.Pos),
		condition: toBoolExpr(tree.If.Condition),
		body:      tree.If.Body,
	})
	for _, elseIf := range tree.ElseIfs {
		branches = append(branches, decisionTreeBranch{
			pos:       posFromLexer(elseIf.Pos),
			condition: toBoolExpr(elseIf.Condition),
			body:      elseIf.Body,
		})
	}
	branches = append(branches, decisionTreeBranch{
		pos:  posFromLexer(tree.Else.Pos),
		body: tree.Else.Body,
	})

	basePriority := decisionTreePriorityBase - (treeIndex * decisionTreePriorityStep)
	priorConditions := make([]BoolExpr, 0, len(branches)-1)
	routes := make([]*RouteDecl, 0, len(branches))
	errs := make([]error, 0)

	for i, branch := range branches {
		route, err := decisionTreeBranchToRoute(tree, branch, i, basePriority-i, priorConditions)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		routes = append(routes, route)
		if branch.condition != nil {
			priorConditions = append(priorConditions, branch.condition)
		}
	}

	return routes, errs
}

func decisionTreeBranchToRoute(
	tree *rawDecisionTreeDecl,
	branch decisionTreeBranch,
	branchIndex int,
	priority int,
	priorConditions []BoolExpr,
) (*RouteDecl, error) {
	route := &RouteDecl{
		Name:     generatedDecisionTreeBranchName(unquoteIdent(tree.Name), branchIndex),
		Priority: priority,
		Pos:      branch.pos,
	}

	for _, item := range branch.body {
		switch {
		case item.Name != nil:
			route.Name = unquoteIdent(*item.Name)
		case item.Description != nil:
			route.Description = unquote(*item.Description)
		case item.Tier != nil:
			route.Tier = *item.Tier
		case item.Model != nil:
			for _, model := range item.Model.Models {
				route.Models = append(route.Models, rawToModelRef(model))
			}
		case item.Algorithm != nil:
			route.Algorithm = rawToAlgo(item.Algorithm)
		case item.Plugin != nil:
			route.Plugins = append(route.Plugins, rawToPluginRef(item.Plugin))
		}
	}

	if len(route.Models) == 0 {
		return nil, fmt.Errorf("DECISION_TREE %q branch %d must declare at least one MODEL. Add MODEL \"<model_name>\" inside the branch body", unquoteIdent(tree.Name), branchIndex+1)
	}

	conditions := make([]BoolExpr, 0, len(priorConditions)+1)
	if branch.condition != nil {
		conditions = append(conditions, branch.condition)
	}
	for _, prior := range priorConditions {
		conditions = append(conditions, &BoolNot{
			Expr: prior,
			Pos:  prior.GetPos(),
		})
	}
	route.When = andExprs(conditions)

	return route, nil
}

func generatedDecisionTreeBranchName(treeName string, branchIndex int) string {
	return fmt.Sprintf("%s__branch_%02d", treeName, branchIndex+1)
}

func andExprs(exprs []BoolExpr) BoolExpr {
	if len(exprs) == 0 {
		return nil
	}
	result := exprs[0]
	for _, expr := range exprs[1:] {
		result = &BoolAnd{
			Left:  result,
			Right: expr,
			Pos:   result.GetPos(),
		}
	}
	return result
}
