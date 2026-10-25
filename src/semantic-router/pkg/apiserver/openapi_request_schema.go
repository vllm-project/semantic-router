//go:build !windows

package apiserver

import (
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/openapischema"
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

// openAPIRequestSchemaFor derives request fields from the same Go type decoded
// by a route handler. Request structs remain the field-level source of truth;
// the route catalog only associates that type with its operation.
func openAPIRequestSchemaFor[T any]() *OpenAPISchema {
	schema := openapischema.For[T](routerSchemaOverrides)
	return &schema
}
