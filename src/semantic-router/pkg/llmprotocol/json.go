package llmprotocol

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"unicode/utf8"
)

// ValidateJSONObject validates protocol-neutral JSON values that are carried
// as strings or raw messages, such as tool arguments and JSON Schemas. The
// standard library accepts duplicate object members and replaces malformed
// Unicode surrogates; both behaviors would make translation non-deterministic.
func ValidateJSONObject(body []byte, maximumDepth int) error {
	if !utf8.Valid(body) {
		return fmt.Errorf("JSON is not valid UTF-8")
	}
	if err := validateJSONUnicodeEscapes(body); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok || delimiter != '{' {
		return fmt.Errorf("JSON value must be an object")
	}
	if err := consumeJSONObject(decoder, 0, maximumDepth); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("JSON contains an additional document")
		}
		return err
	}
	return nil
}

func consumeJSONValue(decoder *json.Decoder, depth, maximumDepth int) error {
	if maximumDepth <= 0 || depth > maximumDepth {
		return fmt.Errorf("JSON nesting exceeds the configured limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	delimiter, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	switch delimiter {
	case '{':
		return consumeJSONObject(decoder, depth, maximumDepth)
	case '[':
		return consumeJSONArray(decoder, depth, maximumDepth)
	default:
		return fmt.Errorf("unexpected JSON delimiter %q", delimiter)
	}
}

func consumeJSONObject(decoder *json.Decoder, depth, maximumDepth int) error {
	seen := make(map[string]struct{})
	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return err
		}
		key, ok := keyToken.(string)
		if !ok {
			return fmt.Errorf("JSON object key is not a string")
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("JSON object contains duplicate field %q", key)
		}
		seen[key] = struct{}{}
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim('}') {
		return fmt.Errorf("unterminated JSON object")
	}
	return nil
}

func consumeJSONArray(decoder *json.Decoder, depth, maximumDepth int) error {
	for decoder.More() {
		if err := consumeJSONValue(decoder, depth+1, maximumDepth); err != nil {
			return err
		}
	}
	closing, err := decoder.Token()
	if err != nil || closing != json.Delim(']') {
		return fmt.Errorf("unterminated JSON array")
	}
	return nil
}

// encoding/json replaces unpaired UTF-16 surrogates with U+FFFD. Reject them
// before decoding so identifiers, tool arguments, and schemas remain exact.
func validateJSONUnicodeEscapes(body []byte) error {
	insideString := false
	for index := 0; index < len(body); index++ {
		if body[index] == '"' {
			insideString = !insideString
			continue
		}
		if body[index] != '\\' || !insideString {
			continue
		}
		next, err := advanceJSONUnicodeEscape(body, index)
		if err != nil {
			return err
		}
		index = next
	}
	return nil
}

func advanceJSONUnicodeEscape(body []byte, index int) (int, error) {
	if index+1 >= len(body) {
		return index, nil
	}
	if body[index+1] != 'u' {
		return index + 1, nil
	}
	value, ok := decodeHexQuad(body, index+2)
	if !ok {
		return index, nil
	}
	if value >= 0xdc00 && value <= 0xdfff {
		return index, fmt.Errorf("JSON contains a lone low surrogate")
	}
	if value < 0xd800 || value > 0xdbff {
		return index + 5, nil
	}
	if !hasLowSurrogateEscape(body, index) {
		return index, fmt.Errorf("JSON high surrogate is not followed by a low surrogate")
	}
	return index + 11, nil
}

func hasLowSurrogateEscape(body []byte, index int) bool {
	if index+11 >= len(body) || body[index+6] != '\\' || body[index+7] != 'u' {
		return false
	}
	low, validLow := decodeHexQuad(body, index+8)
	return validLow && low >= 0xdc00 && low <= 0xdfff
}

func decodeHexQuad(body []byte, start int) (uint16, bool) {
	if start < 0 || start+4 > len(body) {
		return 0, false
	}
	var value uint16
	for _, character := range body[start : start+4] {
		value <<= 4
		switch {
		case character >= '0' && character <= '9':
			value += uint16(character - '0')
		case character >= 'a' && character <= 'f':
			value += uint16(character-'a') + 10
		case character >= 'A' && character <= 'F':
			value += uint16(character-'A') + 10
		default:
			return 0, false
		}
	}
	return value, true
}
