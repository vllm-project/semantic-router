package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// rejectDuplicateJSONKeys prevents encoding/json's last-value-wins behavior
// from making an ambiguous object admissible at an evaluation trust boundary.
// It walks the token stream so duplicate keys are rejected in every nested
// object, including objects contained in arrays.
func rejectDuplicateJSONKeys(data []byte) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	if err := inspectJSONValueKeys(decoder); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}

// decodeExactJSON is the common evaluation JSON boundary: every object must
// have unique keys, every field must be declared by the destination contract,
// and no trailing JSON value is accepted.
func decodeExactJSON(data []byte, destination any) error {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	return ensureJSONEOF(decoder)
}

func inspectJSONValueKeys(decoder *json.Decoder) error {
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, structured := token.(json.Delim)
	if !structured {
		return nil
	}
	switch delimiter {
	case '{':
		seen := make(map[string]struct{})
		for decoder.More() {
			keyToken, keyErr := decoder.Token()
			if keyErr != nil {
				return keyErr
			}
			key, ok := keyToken.(string)
			if !ok {
				return fmt.Errorf("JSON object key must be a string")
			}
			if _, duplicate := seen[key]; duplicate {
				return fmt.Errorf("duplicate JSON object key %q", key)
			}
			seen[key] = struct{}{}
			if valueErr := inspectJSONValueKeys(decoder); valueErr != nil {
				return valueErr
			}
		}
		closing, closingErr := decoder.Token()
		if closingErr != nil {
			return closingErr
		}
		if closing != json.Delim('}') {
			return fmt.Errorf("JSON object has invalid closing delimiter")
		}
	case '[':
		for decoder.More() {
			if valueErr := inspectJSONValueKeys(decoder); valueErr != nil {
				return valueErr
			}
		}
		closing, closingErr := decoder.Token()
		if closingErr != nil {
			return closingErr
		}
		if closing != json.Delim(']') {
			return fmt.Errorf("JSON array has invalid closing delimiter")
		}
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
	return nil
}
