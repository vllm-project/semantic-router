package dsl

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
)

// authzReferenceProgram declares one binding, admins, that grants the role
// admin, and references it from a WHEN clause and a projection input.
const authzReferenceProgram = `
SIGNAL authz admins {
  role: "admin"
  subjects: [{ kind: "Group", name: "platform-admins" }]
}

ROUTE admin_route {
  PRIORITY 100
  WHEN authz(%[1]q)
  MODEL "model-admin"
}

PROJECTION score admin_score {
  method: "weighted_sum"
  inputs: [
    { type: "authz", name: %[1]q, weight: 1.0 }
  ]
}
`

func authzReferenceDiagnostics(t *testing.T, reference string) []Diagnostic {
	t.Helper()
	diags, errs := Validate(fmt.Sprintf(authzReferenceProgram, reference))
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	var found []Diagnostic
	for _, d := range diags {
		if strings.Contains(d.Message, "authz(") {
			found = append(found, d)
		}
	}
	return found
}

// The Router's authz signal emits the role a binding grants, so WHEN clauses
// and projection inputs name the role; the binding's own name never matches.
func TestValidateAuthzReferencesResolveRoles(t *testing.T) {
	if diags := authzReferenceDiagnostics(t, "admin"); len(diags) != 0 {
		t.Fatalf("role reference: unexpected diagnostics %v", diags)
	}

	diags := authzReferenceDiagnostics(t, "admins")
	if len(diags) != 2 {
		t.Fatalf("binding-name reference: want a WHEN and a projection warning, got %v", diags)
	}
	for _, d := range diags {
		if d.Level != DiagWarning || !strings.Contains(d.Message, "is not defined") {
			t.Errorf("binding-name reference: got %s", d)
		}
		if d.Fix == nil || d.Fix.NewText != "admin" {
			t.Errorf("binding-name reference: the fix should name the role, got %+v", d.Fix)
		}
	}
}

func TestValidateSymbolTableListsAuthzRoles(t *testing.T) {
	_, symbols, errs := ValidateWithSymbols(fmt.Sprintf(authzReferenceProgram, "admin"))
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	var authz []string
	for _, s := range symbols.Signals {
		if s.Type == "authz" {
			authz = append(authz, s.Name)
		}
	}
	if !reflect.DeepEqual(authz, []string{"admin"}) {
		t.Fatalf("authz completion symbols = %v, want [admin]", authz)
	}
}
