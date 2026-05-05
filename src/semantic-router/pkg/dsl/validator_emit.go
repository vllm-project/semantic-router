package dsl

import "fmt"

// emitKindRetention is the only EMIT kind currently recognized by the
// contract layer. Future kinds (e.g. "log", "metric") will be added here.
const emitKindRetention = "retention"

var supportedEmitKinds = map[string]bool{
	emitKindRetention: true,
}

var retentionFieldTypes = map[string]string{
	"drop":                    "bool",
	"ttl_turns":               "int",
	"keep_current_model":      "bool",
	"prefer_prefix_retention": "bool",
}

type retentionFieldDiagnostic struct {
	pos     Position
	message string
}

// checkRouteEmits validates EMIT directives attached to a route. It reports:
//   - error: unknown Kind
//   - error: unknown retention field
//   - error: wrong retention field type
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
	v.addRetentionRawFieldDiagnostics(e, context)

	if retentionDirectiveEmpty(e) {
		v.warnEmptyRetentionDirective(e, context)
		return
	}

	r := e.Retention
	v.checkRetentionTTLTurns(r, e.Pos, context)
	v.checkRetentionDropConflict(r, e.Pos, context)
	v.checkRetentionZeroTurnsNoDrop(r, e.Pos, context)
}

func (v *Validator) addRetentionRawFieldDiagnostics(e *EmitDecl, context string) {
	for _, issue := range retentionRawFieldDiagnostics(e, context) {
		v.addDiag(DiagError, issue.pos, issue.message, nil)
	}
}

func retentionDirectiveEmpty(e *EmitDecl) bool {
	if e == nil || e.Retention == nil {
		return true
	}
	r := e.Retention
	return r.Drop == nil && r.TTLTurns == nil && r.KeepCurrentModel == nil && r.PreferPrefixRetention == nil
}

func (v *Validator) warnEmptyRetentionDirective(e *EmitDecl, context string) {
	if e == nil || len(e.RawFields) > 0 {
		return
	}
	v.addDiag(DiagWarning, e.Pos,
		fmt.Sprintf("%s retention: empty block has no effect. Set drop, ttl_turns, keep_current_model, or prefer_prefix_retention", context),
		nil,
	)
}

func (v *Validator) checkRetentionTTLTurns(r *RetentionDirective, pos Position, context string) {
	if r.TTLTurns != nil && *r.TTLTurns < 0 {
		v.addDiag(DiagError, pos,
			fmt.Sprintf("%s retention: ttl_turns must be >= 0, got %d", context, *r.TTLTurns),
			nil,
		)
	}
}

func (v *Validator) checkRetentionDropConflict(r *RetentionDirective, pos Position, context string) {
	if retentionDropTrue(r) && r.TTLTurns != nil && *r.TTLTurns > 0 {
		v.addDiag(DiagError, pos,
			fmt.Sprintf("%s retention: drop=true conflicts with ttl_turns=%d. Use one or the other", context, *r.TTLTurns),
			nil,
		)
	}
}

func (v *Validator) checkRetentionZeroTurnsNoDrop(r *RetentionDirective, pos Position, context string) {
	if r.TTLTurns != nil && *r.TTLTurns == 0 && r.Drop == nil {
		v.addDiag(DiagWarning, pos,
			fmt.Sprintf("%s retention: ttl_turns=0 without drop is likely a no-op. Add drop: true to evict, or remove ttl_turns", context),
			nil,
		)
	}
}

func retentionDropTrue(r *RetentionDirective) bool {
	return r.Drop != nil && *r.Drop
}

func retentionRawFieldDiagnostics(e *EmitDecl, context string) []retentionFieldDiagnostic {
	if e == nil || e.Kind != emitKindRetention {
		return nil
	}
	issues := make([]retentionFieldDiagnostic, 0)
	for _, field := range e.RawFields {
		if field == nil {
			continue
		}
		want, ok := retentionFieldTypes[field.Key]
		if !ok {
			issues = append(issues, retentionFieldDiagnostic{
				pos:     retentionFieldPosition(field, e.Pos),
				message: fmt.Sprintf("%s retention: unknown field %q. Supported fields: drop, ttl_turns, keep_current_model, prefer_prefix_retention", context, field.Key),
			})
			continue
		}
		if !retentionFieldHasType(field, want) {
			issues = append(issues, retentionFieldDiagnostic{
				pos:     retentionFieldPosition(field, e.Pos),
				message: fmt.Sprintf("%s retention: field %q must be %s, got %s", context, field.Key, want, retentionFieldTypeName(field)),
			})
		}
	}
	return issues
}

func retentionFieldHasType(field *FieldEntry, want string) bool {
	if field == nil || field.Value == nil {
		return false
	}
	switch want {
	case "bool":
		return field.Value.Bool != nil
	case "int":
		return field.Value.Int != nil
	default:
		return false
	}
}

func retentionFieldTypeName(field *FieldEntry) string {
	if field == nil || field.Value == nil {
		return "missing"
	}
	switch {
	case field.Value.Str != nil:
		return "string"
	case field.Value.Float != nil:
		return "float"
	case field.Value.Int != nil:
		return "int"
	case field.Value.Bool != nil:
		return "bool"
	case field.Value.ArrayVal != nil:
		return "array"
	case field.Value.Object != nil:
		return "object"
	case field.Value.BareStr != nil:
		return "identifier"
	default:
		return "unknown"
	}
}

func retentionFieldPosition(field *FieldEntry, fallback Position) Position {
	if field == nil || field.Value == nil {
		return fallback
	}
	return posFromLexer(field.Value.Pos)
}
