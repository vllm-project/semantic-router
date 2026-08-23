package agentmanagement

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"

	jsonschema "github.com/santhosh-tekuri/jsonschema/v5"
)

const (
	maximumSchemaBytes = 256 << 10
	maximumSchemaDepth = 32
	maximumSchemaNodes = 4096
	maximumValueDepth  = 64
	maximumValueNodes  = 32768
)

// compiledSchema is deliberately a thin bound around a standards-compliant
// Draft 2020-12 validator. Compilation is offline: an unregistered URI can
// never cause filesystem or network I/O.
type compiledSchema struct {
	validator *jsonschema.Schema
}

func compileToolSchema(raw json.RawMessage) (*compiledSchema, json.RawMessage, error) {
	if len(raw) == 0 || len(raw) > maximumSchemaBytes {
		return nil, nil, fmt.Errorf("%w: tool schema size is invalid", ErrInvalid)
	}
	value, compileToolSchemaErr := decodeBoundedJSON(raw, maximumSchemaDepth, maximumSchemaNodes)
	if compileToolSchemaErr != nil {
		return nil, nil, fmt.Errorf("%w: tool schema is invalid: %w", ErrInvalid, compileToolSchemaErr)
	}
	if _, ok := value.(map[string]any); !ok {
		return nil, nil, fmt.Errorf("%w: tool schema must be an object", ErrInvalid)
	}
	canonical, compileToolSchemaErr := json.Marshal(value)
	if compileToolSchemaErr != nil {
		return nil, nil, fmt.Errorf("%w: canonicalize tool schema", ErrInvalid)
	}
	const resourceURL = "urn:vllm-sr:agent-tool-schema"
	compiler := jsonschema.NewCompiler()
	compiler.Draft = jsonschema.Draft2020
	compiler.AssertFormat = true
	compiler.LoadURL = func(resource string) (io.ReadCloser, error) {
		return nil, fmt.Errorf("external schema resource %q is disabled", resource)
	}
	if err := compiler.AddResource(resourceURL, bytes.NewReader(canonical)); err != nil {
		return nil, nil, fmt.Errorf("%w: register tool schema: %w", ErrInvalid, err)
	}
	validator, compileToolSchemaErr := compiler.Compile(resourceURL)
	if compileToolSchemaErr != nil {
		return nil, nil, fmt.Errorf("%w: compile tool schema: %w", ErrInvalid, compileToolSchemaErr)
	}
	return &compiledSchema{validator: validator}, canonical, nil
}

func (schema *compiledSchema) validateRaw(raw json.RawMessage, maximum int) error {
	if schema == nil || schema.validator == nil || len(raw) == 0 || len(raw) > maximum {
		return fmt.Errorf("%w: tool value exceeds its bound", ErrInvalid)
	}
	value, err := decodeBoundedJSON(raw, maximumValueDepth, maximumValueNodes)
	if err != nil {
		return fmt.Errorf("%w: tool value is invalid: %w", ErrInvalid, err)
	}
	if _, object := value.(map[string]any); !object {
		return fmt.Errorf("%w: top-level tool value must be an object", ErrInvalid)
	}
	if err := schema.validator.Validate(value); err != nil {
		return fmt.Errorf("%w: tool value does not match schema: %w", ErrInvalid, err)
	}
	return nil
}

// decodeBoundedJSON rejects duplicate object keys before the ordinary JSON
// decoder can collapse them and bounds both schema and result complexity.
func decodeBoundedJSON(raw []byte, maximumDepth, maximumNodes int) (any, error) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	nodes := 0
	value, err := decodeJSONValue(decoder, 0, maximumDepth, maximumNodes, &nodes)
	if err != nil {
		return nil, err
	}
	if token, err := decoder.Token(); !errors.Is(err, io.EOF) {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("unexpected trailing token %v", token)
	}
	return value, nil
}

func decodeJSONValue(
	decoder *json.Decoder,
	depth, maximumDepth, maximumNodes int,
	nodes *int,
) (any, error) {
	(*nodes)++
	if depth > maximumDepth || *nodes > maximumNodes {
		return nil, errors.New("JSON document exceeds structural limits")
	}
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	delimiter, compound := token.(json.Delim)
	if !compound {
		return token, nil
	}
	switch delimiter {
	case '{':
		object := make(map[string]any)
		for decoder.More() {
			nameToken, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			name, ok := nameToken.(string)
			if !ok {
				return nil, errors.New("object key is not a string")
			}
			if _, duplicate := object[name]; duplicate {
				return nil, fmt.Errorf("duplicate object key %q", name)
			}
			value, err := decodeJSONValue(decoder, depth+1, maximumDepth, maximumNodes, nodes)
			if err != nil {
				return nil, err
			}
			object[name] = value
		}
		if closing, err := decoder.Token(); err != nil || closing != json.Delim('}') {
			return nil, errors.New("unterminated JSON object")
		}
		return object, nil
	case '[':
		array := make([]any, 0)
		for decoder.More() {
			value, err := decodeJSONValue(decoder, depth+1, maximumDepth, maximumNodes, nodes)
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		if closing, err := decoder.Token(); err != nil || closing != json.Delim(']') {
			return nil, errors.New("unterminated JSON array")
		}
		return array, nil
	default:
		return nil, fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}
