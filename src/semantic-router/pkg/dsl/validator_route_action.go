package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// checkRouteAction mirrors the config loader's route-action contract: only
// the route type, a known destination model, and an explicit jailbreak
// condition are accepted.
func (v *Validator) checkRouteAction(r *RouteDecl, context string) {
	if r.Action == nil {
		return
	}
	if r.Action.Type != "route" {
		v.addDiag(DiagConstraint, r.Action.Pos,
			fmt.Sprintf("%s: action type must be \"route\", got %q", context, r.Action.Type),
			nil,
		)
		return
	}
	if strings.TrimSpace(r.Action.Destination) == "" {
		v.addDiag(DiagConstraint, r.Action.Pos,
			fmt.Sprintf("%s: action destination is required", context),
			nil,
		)
		return
	}
	if len(v.modelNames) > 0 && !v.modelNames[r.Action.Destination] {
		v.addDiag(DiagWarning, r.Action.Pos,
			fmt.Sprintf("%s: action destination %q is not a declared model", context, r.Action.Destination),
			nil,
		)
	}
	if !boolExprReferencesSignal(r.When, config.SignalTypeJailbreak) {
		v.addDiag(DiagConstraint, r.Action.Pos,
			fmt.Sprintf("%s: a route action requires an explicit jailbreak condition in WHEN", context),
			nil,
		)
	}
}

func boolExprReferencesSignal(expr BoolExpr, signalType string) bool {
	switch e := expr.(type) {
	case *BoolAnd:
		return boolExprReferencesSignal(e.Left, signalType) ||
			boolExprReferencesSignal(e.Right, signalType)
	case *BoolOr:
		return boolExprReferencesSignal(e.Left, signalType) ||
			boolExprReferencesSignal(e.Right, signalType)
	case *BoolNot:
		return boolExprReferencesSignal(e.Expr, signalType)
	case *SignalRefExpr:
		return strings.EqualFold(e.SignalType, signalType)
	}
	return false
}
