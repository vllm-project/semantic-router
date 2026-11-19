package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// retentionFieldTypes are the fields a RETENTION block takes, by type.
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

// rawToRetentionDecl converts a parsed RETENTION block into the resolved AST
// form. Field-level coercion is centralized here so the validator and the
// compiler check one typed structure.
func rawToRetentionDecl(r *rawRetentionDecl) *RetentionDecl {
	if r == nil {
		return nil
	}
	decl := &RetentionDecl{RawFields: r.Fields, Pos: posFromLexer(r.Pos)}
	for _, e := range r.Fields {
		if e == nil || e.Value == nil {
			continue
		}
		switch e.Key {
		case "drop":
			decl.Drop = boolEntry(e)
		case "ttl_turns":
			if e.Value.Int != nil {
				n := *e.Value.Int
				decl.TTLTurns = &n
			}
		case "keep_current_model":
			decl.KeepCurrentModel = boolEntry(e)
		case "prefer_prefix_retention":
			decl.PreferPrefixRetention = boolEntry(e)
		}
	}
	return decl
}

func boolEntry(e *FieldEntry) *bool {
	if e.Value.Bool == nil {
		return nil
	}
	b := *e.Value.Bool == "true"
	return &b
}

func (r *RetentionDecl) empty() bool {
	return r.Drop == nil && r.TTLTurns == nil && r.KeepCurrentModel == nil && r.PreferPrefixRetention == nil
}

// retentionErrors are a RETENTION block's hard errors: unknown or
// wrong-typed fields, a negative ttl_turns, and drop with a positive
// ttl_turns.
func retentionErrors(r *RetentionDecl, context string) []retentionFieldDiagnostic {
	issues := retentionRawFieldDiagnostics(r, context)
	if r.TTLTurns != nil && *r.TTLTurns < 0 {
		issues = append(issues, retentionFieldDiagnostic{r.Pos, fmt.Sprintf("%s: ttl_turns must be >= 0, got %d", context, *r.TTLTurns)})
	}
	if r.Drop != nil && *r.Drop && r.TTLTurns != nil && *r.TTLTurns > 0 {
		issues = append(issues, retentionFieldDiagnostic{r.Pos, fmt.Sprintf("%s: drop=true conflicts with ttl_turns=%d. Use one or the other", context, *r.TTLTurns)})
	}
	return issues
}

// checkRouteRetention reports a route's RETENTION errors, and warns about a
// block that has no effect: one with no fields, and ttl_turns: 0 without drop.
func (v *Validator) checkRouteRetention(r *RouteDecl) {
	if r == nil || r.Retention == nil {
		return
	}
	retention := r.Retention
	context := fmt.Sprintf("ROUTE %s RETENTION", r.Name)
	for _, issue := range retentionErrors(retention, context) {
		v.addDiag(DiagError, issue.pos, issue.message, nil)
	}
	switch {
	case retention.empty() && len(retention.RawFields) == 0:
		v.addDiag(DiagWarning, retention.Pos,
			fmt.Sprintf("%s: empty block has no effect. Set drop, ttl_turns, keep_current_model, or prefer_prefix_retention", context), nil)
	case retention.TTLTurns != nil && *retention.TTLTurns == 0 && retention.Drop == nil:
		v.addDiag(DiagWarning, retention.Pos,
			fmt.Sprintf("%s: ttl_turns=0 without drop is likely a no-op. Add drop: true to evict, or remove ttl_turns", context), nil)
	}
}

// compileRouteRetention writes a route's RETENTION block as the decision's
// retention attribute, or reports why it cannot.
func (c *Compiler) compileRouteRetention(r *RouteDecl, decision *config.Decision) {
	retention := r.Retention
	if retention == nil {
		return
	}
	issues := retentionErrors(retention, fmt.Sprintf("ROUTE %s RETENTION", r.Name))
	for _, issue := range issues {
		c.addError(issue.pos, "%s", issue.message)
	}
	if len(issues) > 0 {
		return
	}
	decision.Retention = &config.DecisionRetention{
		Drop:                  clonePtrBool(retention.Drop),
		TTLTurns:              clonePtrInt(retention.TTLTurns),
		KeepCurrentModel:      clonePtrBool(retention.KeepCurrentModel),
		PreferPrefixRetention: clonePtrBool(retention.PreferPrefixRetention),
	}
}

func retentionRawFieldDiagnostics(r *RetentionDecl, context string) []retentionFieldDiagnostic {
	issues := make([]retentionFieldDiagnostic, 0)
	for _, field := range r.RawFields {
		if field == nil {
			continue
		}
		want, ok := retentionFieldTypes[field.Key]
		if !ok {
			issues = append(issues, retentionFieldDiagnostic{
				pos:     retentionFieldPosition(field, r.Pos),
				message: fmt.Sprintf("%s: unknown field %q. Supported fields: drop, ttl_turns, keep_current_model, prefer_prefix_retention", context, field.Key),
			})
			continue
		}
		if !retentionFieldHasType(field, want) {
			issues = append(issues, retentionFieldDiagnostic{
				pos:     retentionFieldPosition(field, r.Pos),
				message: fmt.Sprintf("%s: field %q must be %s, got %s", context, field.Key, want, retentionFieldTypeName(field)),
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

// decompileRetention writes a decision's retention as a RETENTION block, in
// the field order of the schema.
func (d *decompiler) decompileRetention(retention *config.DecisionRetention) {
	if retention == nil {
		return
	}
	var lines []string
	if retention.Drop != nil {
		lines = append(lines, fmt.Sprintf("    drop: %v", *retention.Drop))
	}
	if retention.TTLTurns != nil {
		lines = append(lines, fmt.Sprintf("    ttl_turns: %d", *retention.TTLTurns))
	}
	if retention.KeepCurrentModel != nil {
		lines = append(lines, fmt.Sprintf("    keep_current_model: %v", *retention.KeepCurrentModel))
	}
	if retention.PreferPrefixRetention != nil {
		lines = append(lines, fmt.Sprintf("    prefer_prefix_retention: %v", *retention.PreferPrefixRetention))
	}
	if len(lines) == 0 {
		d.write("  RETENTION {}\n")
		return
	}
	d.write("  RETENTION {\n")
	for _, line := range lines {
		d.write("%s\n", line)
	}
	d.write("  }\n")
}

// retentionToDecl is a decision's retention as the AST node a RETENTION
// block parses into.
func retentionToDecl(retention *config.DecisionRetention) *RetentionDecl {
	if retention == nil {
		return nil
	}
	return &RetentionDecl{
		Drop:                  clonePtrBool(retention.Drop),
		TTLTurns:              clonePtrInt(retention.TTLTurns),
		KeepCurrentModel:      clonePtrBool(retention.KeepCurrentModel),
		PreferPrefixRetention: clonePtrBool(retention.PreferPrefixRetention),
	}
}
