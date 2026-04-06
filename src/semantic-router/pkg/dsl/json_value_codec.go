package dsl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func structuredPayloadFromFields(fields map[string]Value) (*config.StructuredPayload, error) {
	return structuredPayloadFromJSONObject(marshalObjectFields(fields))
}

func structuredPayloadFromJSONObject(obj JSONObject) (*config.StructuredPayload, error) {
	data, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("marshal structured payload: %w", err)
	}
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return nil, nil
	}
	return &config.StructuredPayload{Raw: json.RawMessage(data)}, nil
}

func structuredPayloadToJSONObject(payload *config.StructuredPayload) (JSONObject, bool) {
	if payload == nil || payload.IsEmpty() {
		return JSONObject{}, false
	}
	obj, err := unmarshalJSONObjectBytes(payload.Raw)
	if err != nil {
		return JSONObject{}, false
	}
	return obj, true
}

func unmarshalJSONObjectBytes(data []byte) (JSONObject, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return JSONObject{}, err
	}

	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	fields := make([]JSONField, 0, len(keys))
	for _, key := range keys {
		value, err := unmarshalJSONValueBytes(raw[key])
		if err != nil {
			return JSONObject{}, err
		}
		fields = append(fields, JSONField{Name: key, Value: value})
	}
	return JSONObject{Fields: fields}, nil
}

func unmarshalJSONValueBytes(data []byte) (JSONValue, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return JSONValue{Kind: JSONValueNull}, nil
	}

	switch trimmed[0] {
	case '{':
		obj, err := unmarshalJSONObjectBytes(trimmed)
		if err != nil {
			return JSONValue{}, err
		}
		return JSONValue{Kind: JSONValueObject, Object: &obj}, nil
	case '[':
		var rawItems []json.RawMessage
		if err := json.Unmarshal(trimmed, &rawItems); err != nil {
			return JSONValue{}, err
		}
		items := make([]JSONValue, 0, len(rawItems))
		for _, rawItem := range rawItems {
			item, err := unmarshalJSONValueBytes(rawItem)
			if err != nil {
				return JSONValue{}, err
			}
			items = append(items, item)
		}
		return JSONValue{Kind: JSONValueArray, Array: items}, nil
	case '"':
		var value string
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return JSONValue{}, err
		}
		return JSONValue{Kind: JSONValueString, String: value}, nil
	case 't', 'f':
		var value bool
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return JSONValue{}, err
		}
		return JSONValue{Kind: JSONValueBool, Bool: value}, nil
	default:
		number, err := decodeJSONNumber(trimmed)
		if err != nil {
			return JSONValue{}, err
		}
		if strings.ContainsAny(number, ".eE") {
			var value float64
			if err := json.Unmarshal(trimmed, &value); err != nil {
				return JSONValue{}, err
			}
			return JSONValue{Kind: JSONValueFloat, Float: value}, nil
		}

		var value int
		if err := json.Unmarshal(trimmed, &value); err != nil {
			return JSONValue{}, err
		}
		return JSONValue{Kind: JSONValueInt, Int: value}, nil
	}
}

func decodeJSONNumber(data []byte) (string, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value json.Number
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	return value.String(), nil
}

func formatJSONObjectValue(obj JSONObject) string {
	parts := make([]string, 0, len(obj.Fields))
	for _, field := range obj.Fields {
		parts = append(parts, fmt.Sprintf("%s: %s", field.Name, formatJSONValue(field.Value)))
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

func formatJSONValue(value JSONValue) string {
	switch value.Kind {
	case JSONValueNull:
		return "null"
	case JSONValueString:
		return fmt.Sprintf("%q", value.String)
	case JSONValueInt:
		return fmt.Sprintf("%d", value.Int)
	case JSONValueFloat:
		return fmt.Sprintf("%v", value.Float)
	case JSONValueBool:
		return fmt.Sprintf("%t", value.Bool)
	case JSONValueArray:
		parts := make([]string, 0, len(value.Array))
		for _, item := range value.Array {
			parts = append(parts, formatJSONValue(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case JSONValueObject:
		if value.Object == nil {
			return "{}"
		}
		return formatJSONObjectValue(*value.Object)
	default:
		return "null"
	}
}
