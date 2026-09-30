package config

import (
	"testing"

	"gopkg.in/yaml.v3"
)

func TestModelRefSerializationPreservesReasoningInheritance(t *testing.T) {
	for _, tc := range []struct {
		name  string
		value *bool
	}{
		{name: "inherited"},
		{name: "disabled", value: boolPtr(false)},
		{name: "enabled", value: boolPtr(true)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := ModelRef{Model: "example", ModelReasoningControl: ModelReasoningControl{UseReasoning: tc.value}}
			data, err := yaml.Marshal(original)
			if err != nil {
				t.Fatal(err)
			}
			var document map[string]interface{}
			if err := yaml.Unmarshal(data, &document); err != nil {
				t.Fatal(err)
			}
			value, present := document["use_reasoning"]
			if tc.value == nil {
				if present {
					t.Fatalf("inherited reasoning must be omitted, got %q", data)
				}
			} else if !present || value != *tc.value {
				t.Fatalf("explicit reasoning must survive serialization, got %q", data)
			}
			var decoded ModelRef
			if err := yaml.Unmarshal(data, &decoded); err != nil {
				t.Fatal(err)
			}
			if (decoded.UseReasoning == nil) != (tc.value == nil) ||
				(tc.value != nil && *decoded.UseReasoning != *tc.value) {
				t.Fatalf("reasoning inheritance changed after round trip: %#v", decoded)
			}
		})
	}
}
