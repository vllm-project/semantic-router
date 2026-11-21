package protocolcodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// requestFieldError is the client's error for a member its wire type does not
// define, or whose value is the wrong kind, named by its path from the top of
// the request as OpenAI names it. Only the request envelope's walk knows that
// path: a nested value decoded on its own knows its own members only, so it
// keeps the generic code.
func requestFieldError(err error, envelope bool) error {
	var unknown unknownJSONField
	var mistyped mistypedJSONField
	switch {
	case envelope && errors.As(err, &unknown):
		protocolError := llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "unknown_parameter",
			"Unrecognized request argument supplied: "+unknown.path, err)
		protocolError.Parameter = unknown.path
		return protocolError
	case envelope && errors.As(err, &mistyped):
		return invalidTypeError(mistyped.path, mistyped.want, mistyped.got, err)
	}
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_json", nonCanonicalFieldMessage(err), err)
}

func invalidTypeError(path, want, got string, cause error) error {
	protocolError := llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_type",
		fmt.Sprintf("Invalid type for '%s': expected %s, but got %s instead.", path, want, got), cause)
	protocolError.Parameter = path
	return protocolError
}

// jsonFieldPath writes encoding/json's dotted field path, which names no array,
// as the walk does: messages.role becomes messages[].role.
func jsonFieldPath(root reflect.Type, dotted string) string {
	var path strings.Builder
	shape := dereferenceJSONType(root)
	for index, name := range strings.Split(dotted, ".") {
		if index > 0 {
			path.WriteByte('.')
		}
		path.WriteString(name)
		var member reflect.Type
		if shape != nil && shape.Kind() == reflect.Struct {
			if field, ok := exactJSONStructFields(shape)[name]; ok {
				member = dereferenceJSONType(field.typeOf)
			}
		}
		for member != nil && (member.Kind() == reflect.Slice || member.Kind() == reflect.Array) && member.Elem().Kind() != reflect.Uint8 {
			path.WriteString("[]")
			member = dereferenceJSONType(member.Elem())
		}
		shape = member
	}
	return path.String()
}

// jsonTypeName is the JSON value a wire type holds, as an OpenAI error says it.
func jsonTypeName(target reflect.Type) string {
	switch dereferenceJSONType(target).Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "an integer"
	case reflect.Float32, reflect.Float64:
		return "a number"
	case reflect.String:
		return "a string"
	case reflect.Bool:
		return "a boolean"
	case reflect.Slice, reflect.Array:
		return "an array"
	case reflect.Struct, reflect.Map:
		return "an object"
	}
	return "another value"
}

// jsonValueName describes the value of a json.UnmarshalTypeError, which reads
// "number 5.5" for a number an integer field cannot hold.
func jsonValueName(value string) string {
	if number, ok := strings.CutPrefix(value, "number "); ok {
		if strings.ContainsAny(number, ".eE") {
			return "a decimal number"
		}
		return "a number out of range"
	}
	switch value {
	case "number":
		return "a number"
	case "string":
		return "a string"
	case "bool":
		return "a boolean"
	case "array":
		return "an array"
	case "object":
		return "an object"
	}
	return value
}

// jsonTokenName describes the first token of a value the walk read, whose
// numbers are json.Number.
func jsonTokenName(token json.Token) string {
	switch token := token.(type) {
	case json.Delim:
		if token == '[' {
			return "an array"
		}
		return "an object"
	case json.Number:
		if strings.ContainsAny(token.String(), ".eE") {
			return "a decimal number"
		}
		return "an integer"
	case string:
		return "a string"
	case bool:
		return "a boolean"
	}
	return "another value"
}
