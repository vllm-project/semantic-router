package managementapi

import "testing"

func TestRoutingEntrypointAssignmentsHaveClosedTypedSchema(t *testing.T) {
	schemas := routingSchemas()
	for _, name := range []string{"RoutingEntrypointRuleWrite", "RoutingEntrypointRuleView"} {
		assignments := schemas[name].Properties["assignments"]
		if assignments.AdditionalProperties == nil || *assignments.AdditionalProperties {
			t.Fatalf("%s assignments allow untyped values", name)
		}
		if len(assignments.PatternProperties) != 1 {
			t.Fatalf("%s assignment schema = %#v", name, assignments)
		}
	}
	write := schemas["RoutingAssignmentSetWrite"]
	if _, ok := write.Properties["models"]; !ok {
		t.Fatal("assignment set omits models")
	}
	if _, ok := write.Properties["fallback"]; !ok {
		t.Fatal("assignment set omits fallback")
	}
	priority := schemas["RoutingAssignmentWrite"].Properties["priority"]
	if priority.Minimum == nil || *priority.Minimum != 0 || priority.Maximum == nil || *priority.Maximum != 31 {
		t.Fatalf("priority schema = %#v", priority)
	}
}
