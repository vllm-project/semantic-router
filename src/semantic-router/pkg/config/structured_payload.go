package config

import (
	"bytes"
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v2"
)

// StructuredPayload stores an arbitrary YAML/JSON object as normalized JSON bytes.
// It removes weakly typed interface{} fields from public config structs while
// preserving round-trip compatibility for plugin/backend payloads whose exact
// schema depends on sibling discriminator fields.
type StructuredPayload struct {
	Raw json.RawMessage `json:"-" yaml:"-"`
}

func NewStructuredPayload(value interface{}) (*StructuredPayload, error) {
	normalized := normalizeStructuredPayloadValue(value)
	data, err := json.Marshal(normalized)
	if err != nil {
		return nil, fmt.Errorf("marshal structured payload: %w", err)
	}
	if isNullJSON(data) {
		return nil, nil
	}
	return &StructuredPayload{Raw: json.RawMessage(data)}, nil
}

func MustStructuredPayload(value interface{}) *StructuredPayload {
	payload, err := NewStructuredPayload(value)
	if err != nil {
		panic(err)
	}
	return payload
}

func (p *StructuredPayload) IsEmpty() bool {
	return p == nil || len(bytes.TrimSpace(p.Raw)) == 0 || isNullJSON(p.Raw)
}

func (p *StructuredPayload) DecodeInto(target interface{}) error {
	if p == nil || p.IsEmpty() {
		return fmt.Errorf("structured payload is empty")
	}
	if target == nil {
		return fmt.Errorf("structured payload target is nil")
	}
	if err := json.Unmarshal(p.Raw, target); err != nil {
		return fmt.Errorf("decode structured payload: %w", err)
	}
	return nil
}

func (p *StructuredPayload) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw interface{}
	if err := unmarshal(&raw); err != nil {
		return err
	}
	payload, err := NewStructuredPayload(raw)
	if err != nil {
		return err
	}
	if payload == nil {
		*p = StructuredPayload{}
		return nil
	}
	*p = *payload
	return nil
}

func (p StructuredPayload) MarshalYAML() (interface{}, error) {
	if p.IsEmpty() {
		return nil, nil
	}
	var raw interface{}
	if err := json.Unmarshal(p.Raw, &raw); err != nil {
		return nil, fmt.Errorf("unmarshal structured payload: %w", err)
	}
	return raw, nil
}

func (p *StructuredPayload) UnmarshalJSON(data []byte) error {
	if isNullJSON(data) {
		p.Raw = nil
		return nil
	}
	var raw interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	payload, err := NewStructuredPayload(raw)
	if err != nil {
		return err
	}
	if payload == nil {
		p.Raw = nil
		return nil
	}
	p.Raw = payload.Raw
	return nil
}

func (p StructuredPayload) MarshalJSON() ([]byte, error) {
	if p.IsEmpty() {
		return []byte("null"), nil
	}
	return p.Raw, nil
}

func (p *StructuredPayload) Clone() *StructuredPayload {
	if p == nil || p.IsEmpty() {
		return nil
	}
	cloned := make(json.RawMessage, len(p.Raw))
	copy(cloned, p.Raw)
	return &StructuredPayload{Raw: cloned}
}

func (p *StructuredPayload) AsStringMap() (map[string]interface{}, error) {
	if p == nil || p.IsEmpty() {
		return map[string]interface{}{}, nil
	}
	var value map[string]interface{}
	if err := json.Unmarshal(p.Raw, &value); err != nil {
		return nil, fmt.Errorf("decode structured payload as map: %w", err)
	}
	return value, nil
}

func normalizeStructuredPayloadValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			normalized[key] = normalizeStructuredPayloadValue(nested)
		}
		return normalized
	case map[interface{}]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			normalized[fmt.Sprintf("%v", key)] = normalizeStructuredPayloadValue(nested)
		}
		return normalized
	case yaml.MapSlice:
		normalized := make(map[string]interface{}, len(typed))
		for _, item := range typed {
			normalized[fmt.Sprintf("%v", item.Key)] = normalizeStructuredPayloadValue(item.Value)
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, len(typed))
		for index, nested := range typed {
			normalized[index] = normalizeStructuredPayloadValue(nested)
		}
		return normalized
	default:
		return value
	}
}

func isNullJSON(data []byte) bool {
	return bytes.Equal(bytes.TrimSpace(data), []byte("null"))
}
