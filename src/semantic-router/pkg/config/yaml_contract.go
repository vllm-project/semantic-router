package config

import (
	"fmt"

	yamlv2 "gopkg.in/yaml.v2"
	yamlv3 "gopkg.in/yaml.v3"
)

// ParseYAML12Mapping decodes a user-authored YAML mapping with the canonical
// scalar and duplicate-key contract. The v3 decoder preserves YAML 1.2 boolean
// spellings such as the retry/fallback key "on".
func ParseYAML12Mapping(data []byte) (map[string]interface{}, error) {
	var raw map[string]interface{}
	if err := yamlv3.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	return raw, nil
}

// DecodeYAML12Strict decodes one user-authored YAML mapping into a typed
// contract. Normalizing the YAML 1.2 map before the established strict v2
// decoder keeps custom field decoding stable while making every authoring
// entrypoint agree on scalar resolution, duplicate keys, and unknown fields.
func DecodeYAML12Strict(data []byte, target interface{}) error {
	raw, err := ParseYAML12Mapping(data)
	if err != nil {
		return err
	}
	normalized, err := yamlv2.Marshal(raw)
	if err != nil {
		return fmt.Errorf("marshal normalized YAML input: %w", err)
	}
	if err := yamlv2.UnmarshalStrict(normalized, target); err != nil {
		return err
	}
	return nil
}
