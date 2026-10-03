package trainingcontract

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Parameters contains JSON values. Integer literals decode as int; decimal and
// exponent literals decode as float64, including inside objects and arrays.
// Out-of-range numbers are rejected instead of silently converting integers to
// float64. Use this type at both the management and worker boundaries.
type Parameters map[string]any

func (p *Parameters) UnmarshalJSON(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var values map[string]any
	if err := decoder.Decode(&values); err != nil {
		return err
	}
	if _, err := decodeParameterNumbers(values); err != nil {
		return fmt.Errorf("parameters: %w", err)
	}
	*p = values
	return nil
}

func decodeParameterNumbers(value any) (any, error) {
	switch v := value.(type) {
	case json.Number:
		if strings.ContainsAny(v.String(), ".eE") {
			return v.Float64()
		}
		return strconv.Atoi(v.String())
	case map[string]any:
		for key, item := range v {
			decoded, err := decodeParameterNumbers(item)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", key, err)
			}
			v[key] = decoded
		}
	case []any:
		for i, item := range v {
			decoded, err := decodeParameterNumbers(item)
			if err != nil {
				return nil, fmt.Errorf("[%d]: %w", i, err)
			}
			v[i] = decoded
		}
	}
	return value, nil
}

func (p Parameters) MarshalJSON() ([]byte, error) {
	if p == nil {
		return []byte("null"), nil
	}
	values := make(map[string]parameterJSON, len(p))
	for key, value := range p {
		values[key] = parameterJSON{value}
	}
	return json.Marshal(values)
}

type parameterJSON struct{ value any }

func (p parameterJSON) MarshalJSON() ([]byte, error) {
	switch v := p.value.(type) {
	case float64:
		data, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		// encoding/json emits some whole-valued floats (e.g. 1e20) without a
		// decimal point or exponent. Keep their float type on the next decode,
		// even when the value is outside the int range.
		if !bytes.ContainsAny(data, ".eE") {
			data = append(data, '.', '0')
		}
		return data, nil
	case map[string]any:
		return json.Marshal(Parameters(v))
	case []any:
		if v == nil {
			return []byte("null"), nil
		}
		values := make([]parameterJSON, len(v))
		for i, value := range v {
			values[i] = parameterJSON{value}
		}
		return json.Marshal(values)
	default:
		return json.Marshal(v)
	}
}
