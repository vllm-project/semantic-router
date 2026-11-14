//go:build !windows

package apiserver

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestTypedClassificationSchemaIsolationBetweenStrictAndLegacyRoutes(t *testing.T) {
	plain := openAPIRequestSchemaFor[services.IntentRequest]()
	before, err := json.Marshal(plain)
	if err != nil {
		t.Fatal(err)
	}
	strict := strictJSONBodyFor[services.IntentRequest]().Schema
	if strict.Properties["options"].AdditionalProperties != false {
		t.Fatal("strict route did not close its nested options")
	}
	for name, schema := range map[string]*OpenAPISchema{
		"previous":   plain,
		"subsequent": openAPIRequestSchemaFor[services.IntentRequest](),
	} {
		encoded, err := json.Marshal(schema)
		if err != nil {
			t.Fatal(err)
		}
		if string(encoded) != string(before) {
			t.Fatalf("strict route mutated %s legacy schema: %s", name, encoded)
		}
	}
}

func TestTypedClassificationSchemaOverridesReachNestedInputs(t *testing.T) {
	want := *openAPIRequestSchemaFor[services.ClassificationInput]()
	for name, input := range map[string]OpenAPISchema{
		"pointer": *openAPIRequestSchemaFor[*services.ClassificationInput](),
		"intent":  openAPIRequestSchemaFor[services.IntentRequest]().Properties["input"],
		"batch":   *openAPIRequestSchemaFor[BatchClassificationRequest]().Properties["inputs"].Items,
	} {
		t.Run(name, func(t *testing.T) {
			if !reflect.DeepEqual(input, want) || input.Nullable || len(input.OneOf) != 2 {
				t.Fatalf("typed input lost its non-nullable variants: %+v", input)
			}
			for index, variant := range []struct{ kind, payload string }{{"text", "text"}, {"image", "image_url"}} {
				branch := input.OneOf[index]
				if !reflect.DeepEqual(branch.Required, []string{"type", variant.payload}) || branch.AdditionalProperties != false || len(branch.Properties) != 2 {
					t.Fatalf("variant permits missing or extra fields: %+v", branch)
				}
				if discriminator := branch.Properties["type"]; discriminator.Type != "string" || !reflect.DeepEqual(discriminator.Enum, []string{variant.kind}) {
					t.Fatalf("wrong discriminator: %+v", discriminator)
				}
				if payload := branch.Properties[variant.payload]; payload.Type != "string" || payload.Nullable {
					t.Fatalf("payload must be a non-null string: %+v", payload)
				}
			}
		})
	}
}
