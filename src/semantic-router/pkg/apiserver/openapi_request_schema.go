//go:build !windows

package apiserver

import (
	"maps"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openapischema"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

var outputTokenMinimum = int64(1)

// routerSchemaOverrides describes Router types whose JSON encoding reflection
// cannot see.
var routerSchemaOverrides = openapischema.Overrides{
	reflect.TypeFor[config.OutputTokenDefault](): {OneOf: []OpenAPISchema{
		{Type: "integer", Minimum: &outputTokenMinimum},
		{Type: "string", Enum: []string{"auto"}},
	}},
}

func newRouterSchemaOverrides() openapischema.Overrides {
	// Routes may close object schemas for strict decoding. Derive request
	// overrides afresh so that mutation cannot affect a different route.
	overrides := maps.Clone(routerSchemaOverrides)
	fields := openapischema.For[services.ClassificationInput](overrides).Properties
	variants := make([]OpenAPISchema, 0, 2)
	for _, variant := range []struct{ kind, payload string }{{"text", "text"}, {"image", "image_url"}} {
		payload := fields[variant.payload]
		payload.Nullable = false
		variants = append(variants, OpenAPISchema{
			Type: "object", Required: []string{"type", variant.payload}, AdditionalProperties: false,
			Properties: map[string]OpenAPISchema{"type": {Type: "string", Enum: []string{variant.kind}}, variant.payload: payload},
		})
	}
	input := OpenAPISchema{OneOf: variants, Description: "Exactly one typed text or inline base64 image input. Remote image URLs and file paths are rejected."}
	overrides[reflect.TypeFor[services.ClassificationInput]()] = input
	// A present typed input must be an object, including when the DTO uses a
	// pointer to distinguish omission. The handler rejects an explicit null.
	overrides[reflect.TypeFor[*services.ClassificationInput]()] = input

	batch := openapischema.For[BatchClassificationRequest](overrides)
	minimum := int64(1)
	for _, name := range []string{"texts", "inputs"} {
		field := batch.Properties[name]
		field.Nullable, field.MinItems = false, &minimum
		batch.Properties[name] = field
	}
	// Both branches matching is invalid: exactly one input form is required.
	batch.OneOf = []OpenAPISchema{{Required: []string{"texts"}}, {Required: []string{"inputs"}}}
	overrides[reflect.TypeFor[BatchClassificationRequest]()] = batch

	intent := openapischema.For[services.IntentRequest](overrides)
	intent.OneOf = []OpenAPISchema{
		{Required: []string{"input"}, Properties: map[string]OpenAPISchema{
			"text": {Not: &OpenAPISchema{}}, "messages": {Not: &OpenAPISchema{}},
		}},
		{Not: &OpenAPISchema{Required: []string{"input"}}},
	}
	overrides[reflect.TypeFor[services.IntentRequest]()] = intent
	return overrides
}

// openAPIRequestSchemaFor derives request fields from the same Go type decoded
// by a route handler. Request structs remain the field-level source of truth;
// the route catalog only associates that type with its operation.
func openAPIRequestSchemaFor[T any]() *OpenAPISchema {
	schema := openapischema.For[T](newRouterSchemaOverrides())
	return &schema
}
