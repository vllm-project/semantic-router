package apicontract

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/openapischema"

// SchemaFor derives a schema from the Go type a handler encodes or decodes, so
// the handler's type stays the field-level source of truth.
func SchemaFor[T any]() Schema {
	return openapischema.For[T](nil)
}
