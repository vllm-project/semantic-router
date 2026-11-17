package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type outputFieldDiagnostic struct {
	pos     Position
	message string
}

// rawToOutputDecl converts a parsed OUTPUT block into the resolved AST form;
// a kind is a name or a list of names, an action a name.
func rawToOutputDecl(r *rawOutputDecl) *OutputDecl {
	if r == nil {
		return nil
	}
	decl := &OutputDecl{RawFields: r.Fields, Pos: posFromLexer(r.Pos)}
	for _, e := range r.Fields {
		if e == nil || e.Value == nil {
			continue
		}
		switch e.Key {
		case "kind":
			if name, ok := outputName(e.Value); ok {
				decl.Kind = []string{name}
				continue
			}
			if e.Value.ArrayVal != nil {
				decl.Kind = []string{}
				for _, item := range e.Value.ArrayVal.Items {
					if name, ok := outputName(item); ok {
						decl.Kind = append(decl.Kind, name)
					}
				}
			}
		case "action":
			decl.Action, _ = outputName(e.Value)
		}
	}
	return decl
}

func outputName(value *Val) (string, bool) {
	switch {
	case value == nil:
		return "", false
	case value.Str != nil:
		return unquote(*value.Str), true
	case value.BareStr != nil:
		return *value.BareStr, true
	default:
		return "", false
	}
}

func (o *OutputDecl) contract() *config.DecisionOutput {
	output := &config.DecisionOutput{Action: o.Action}
	if o.Kind != nil {
		output.Kind = append(config.OutputKinds{}, o.Kind...)
	}
	return output
}

// outputErrors are an OUTPUT block's errors: an unknown field, a field that
// is not a name (or, for kind, a list of names), and a kind or action the
// output contract does not have.
func outputErrors(o *OutputDecl, context string) []outputFieldDiagnostic {
	var issues []outputFieldDiagnostic
	for _, field := range o.RawFields {
		if field == nil {
			continue
		}
		at := retentionFieldPosition(field, o.Pos)
		switch field.Key {
		case "kind":
			if !outputKindWritten(field.Value) {
				issues = append(issues, outputFieldDiagnostic{at, fmt.Sprintf("%s: field \"kind\" must be one of %s, or a list of them, got %s", context, strings.Join(config.OutputKindNames(), ", "), retentionFieldTypeName(field))})
			}
		case "action":
			if _, ok := outputName(field.Value); !ok {
				issues = append(issues, outputFieldDiagnostic{at, fmt.Sprintf("%s: field \"action\" must be one of %s, got %s", context, strings.Join(config.OutputActionNames(), ", "), retentionFieldTypeName(field))})
			}
		default:
			issues = append(issues, outputFieldDiagnostic{at, fmt.Sprintf("%s: unknown field %q. Supported fields: kind, action", context, field.Key)})
		}
	}
	if len(issues) > 0 {
		return issues
	}
	if err := o.contract().Validate(); err != nil {
		issues = append(issues, outputFieldDiagnostic{o.Pos, fmt.Sprintf("%s: %v", context, err)})
	}
	return issues
}

func outputKindWritten(value *Val) bool {
	if _, ok := outputName(value); ok {
		return true
	}
	if value == nil || value.ArrayVal == nil {
		return false
	}
	for _, item := range value.ArrayVal.Items {
		if _, ok := outputName(item); !ok {
			return false
		}
	}
	return true
}

// checkRouteOutput reports a route's OUTPUT errors.
func (v *Validator) checkRouteOutput(r *RouteDecl) {
	if r == nil || r.Output == nil {
		return
	}
	for _, issue := range outputErrors(r.Output, fmt.Sprintf("ROUTE %s OUTPUT", r.Name)) {
		v.addDiag(DiagError, issue.pos, issue.message, nil)
	}
}

// compileRouteOutput writes a route's OUTPUT block as the decision's output
// contract, or reports why it cannot.
func (c *Compiler) compileRouteOutput(r *RouteDecl, decision *config.Decision) {
	if r.Output == nil {
		return
	}
	issues := outputErrors(r.Output, fmt.Sprintf("ROUTE %s OUTPUT", r.Name))
	for _, issue := range issues {
		c.addError(issue.pos, "%s", issue.message)
	}
	if len(issues) == 0 {
		decision.Output = r.Output.contract()
	}
}

// decompileOutput writes a decision's output contract as an OUTPUT block.
func (d *decompiler) decompileOutput(output *config.DecisionOutput) {
	if output == nil {
		return
	}
	d.write("  OUTPUT {\n")
	if len(output.Kind) == 1 {
		d.write("    kind: %q\n", output.Kind[0])
	} else if output.Kind != nil {
		d.write("    kind: %s\n", quotedStringArray(output.Kind))
	}
	if output.Action != "" {
		d.write("    action: %q\n", output.Action)
	}
	d.write("  }\n")
}

// outputToDecl is a decision's output contract as the AST node an OUTPUT
// block parses into.
func outputToDecl(output *config.DecisionOutput) *OutputDecl {
	if output == nil {
		return nil
	}
	decl := &OutputDecl{Action: output.Action}
	if output.Kind != nil {
		decl.Kind = append([]string{}, output.Kind...)
	}
	return decl
}
