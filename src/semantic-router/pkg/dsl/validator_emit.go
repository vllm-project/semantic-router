package dsl

import "fmt"

// emitKindRetention is the only EMIT kind currently recognized by the
// contract layer. Future kinds (e.g. "log", "metric") will be added here.
const emitKindRetention = "retention"

var supportedEmitKinds = map[string]bool{
	emitKindRetention: true,
}

// checkRouteEmits validates EMIT directives attached to a route. It reports:
//   - error: unknown Kind
//   - error: duplicate Kind within the same route
//   - error: ttl_turns < 0
//   - error: drop=true together with ttl_turns > 0 (hard conflict)
//   - warn:  retention block with no fields set
//   - warn:  ttl_turns == 0 and drop is unset (likely no-op)
func (v *Validator) checkRouteEmits(r *RouteDecl) {
	if r == nil || len(r.Emits) == 0 {
		return
	}
	context := fmt.Sprintf("ROUTE %s EMIT", r.Name)
	seen := make(map[string]bool, len(r.Emits))
	for _, e := range r.Emits {
		if e == nil {
			continue
		}
		if !supportedEmitKinds[e.Kind] {
			v.addDiag(DiagError, e.Pos,
				fmt.Sprintf("%s: unknown EMIT kind %q. Supported kinds: retention", context, e.Kind),
				nil,
			)
			continue
		}
		if seen[e.Kind] {
			v.addDiag(DiagError, e.Pos,
				fmt.Sprintf("%s: duplicate EMIT kind %q in the same route. Each kind may appear at most once", context, e.Kind),
				nil,
			)
			continue
		}
		seen[e.Kind] = true

		if e.Kind == emitKindRetention {
			v.checkRetentionDirective(e, context)
		}
	}
}

func (v *Validator) checkRetentionDirective(e *EmitDecl, context string) {
	r := e.Retention
	if r == nil || (r.Drop == nil && r.TTLTurns == nil && r.KeepCurrentModel == nil && r.PreferPrefixRetention == nil) {
		v.addDiag(DiagWarning, e.Pos,
			fmt.Sprintf("%s retention: empty block has no effect. Set drop, ttl_turns, keep_current_model, or prefer_prefix_retention", context),
			nil,
		)
		return
	}

	if r.TTLTurns != nil && *r.TTLTurns < 0 {
		v.addDiag(DiagError, e.Pos,
			fmt.Sprintf("%s retention: ttl_turns must be >= 0, got %d", context, *r.TTLTurns),
			nil,
		)
	}

	dropTrue := r.Drop != nil && *r.Drop
	if dropTrue && r.TTLTurns != nil && *r.TTLTurns > 0 {
		v.addDiag(DiagError, e.Pos,
			fmt.Sprintf("%s retention: drop=true conflicts with ttl_turns=%d. Use one or the other", context, *r.TTLTurns),
			nil,
		)
	}

	if r.TTLTurns != nil && *r.TTLTurns == 0 && r.Drop == nil {
		v.addDiag(DiagWarning, e.Pos,
			fmt.Sprintf("%s retention: ttl_turns=0 without drop is likely a no-op. Add drop: true to evict, or remove ttl_turns", context),
			nil,
		)
	}
}
