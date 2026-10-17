package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
)

// Hash the complete saved configuration, not the execution projection in
// manifestConfig: provenance and scorer metadata also belong to its identity.
func validateFrozenManifest(data []byte, plan manifest) error {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	value, err := readManifestValue(decoder)
	if err != nil {
		return err
	}
	if _, err = decoder.Token(); !errors.Is(err, io.EOF) {
		return fmt.Errorf("unexpected trailing manifest data")
	}
	saved, ok := value.(map[string]interface{})
	if !ok || len(saved) != 9 {
		return fmt.Errorf("invalid plan fields")
	}
	for _, key := range []string{"schema_version", "experiment_id", "config_sha256", "config", "code_revision", "planner_sha256", "command", "status", "matrix"} {
		if _, exists := saved[key]; !exists {
			return fmt.Errorf("missing plan field %s", key)
		}
	}
	if plan.SchemaVersion != "looper-tts.v1" || saved["status"] != "planned" {
		return fmt.Errorf("invalid plan schema or status")
	}
	for _, key := range []string{"experiment_id", "config_sha256", "planner_sha256"} {
		text, valid := saved[key].(string)
		decoded, decodeErr := hex.DecodeString(text)
		if !valid || decodeErr != nil || len(decoded) != sha256.Size || text != strings.ToLower(text) {
			return fmt.Errorf("invalid %s", key)
		}
	}
	revision, ok := saved["code_revision"].(string)
	if !ok || strings.TrimSpace(revision) == "" {
		return fmt.Errorf("invalid code_revision")
	}
	command, ok := saved["command"].([]interface{})
	if !ok || len(command) == 0 {
		return fmt.Errorf("invalid command")
	}
	for _, argument := range command {
		text, valid := argument.(string)
		if !valid || strings.TrimSpace(text) == "" {
			return fmt.Errorf("invalid command argument")
		}
	}
	if plan.Config.Dataset.EvidenceKind != "synthetic" && plan.Config.Dataset.EvidenceKind != "benchmark" {
		return fmt.Errorf("invalid evidence_kind")
	}
	configValue, ok := saved["config"].(map[string]interface{})
	if !ok || configValue["schema_version"] != "looper-tts.v1" {
		return fmt.Errorf("invalid configuration")
	}
	if err = checkManifestDigest(configValue, plan.ConfigSHA256, "config_sha256"); err != nil {
		return err
	}
	identity := map[string]interface{}{"config": configValue, "code_revision": revision, "planner_sha256": saved["planner_sha256"]}
	if err = checkManifestDigest(identity, plan.ExperimentID, "experiment identity"); err != nil {
		return err
	}
	seeds, ok := configValue["seeds"].([]interface{})
	if !ok || len(seeds) == 0 || len(plan.Config.Arms) == 0 || len(plan.Config.Budgets) == 0 || len(plan.Config.Dataset.Items) == 0 {
		return fmt.Errorf("empty experiment matrix")
	}
	items := make([]interface{}, 0, len(plan.Config.Dataset.Items))
	for _, item := range plan.Config.Dataset.Items {
		items = append(items, item.ID)
	}
	expected := make([]interface{}, 0)
	for _, arm := range plan.Config.Arms {
		for _, budget := range plan.Config.Budgets {
			for _, seed := range seeds {
				number, valid := seed.(json.Number)
				if !valid || strings.ContainsAny(string(number), ".eE-") {
					return fmt.Errorf("invalid matrix seed")
				}
				coordinates := map[string]interface{}{"experiment_id": plan.ExperimentID, "arm_id": arm.ID, "budget_id": budget.ID, "seed": seed}
				id, digestErr := manifestDigest(coordinates)
				if digestErr != nil {
					return digestErr
				}
				delete(coordinates, "experiment_id")
				coordinates["id"] = id
				coordinates["algorithm"] = arm.Algorithm
				coordinates["item_ids"] = items
				expected = append(expected, coordinates)
			}
		}
	}
	expectedDigest, err := manifestDigest(expected)
	if err != nil {
		return err
	}
	return checkManifestDigest(saved["matrix"], expectedDigest, "plan matrix")
}

func checkManifestDigest(value interface{}, expected, field string) error {
	actual, err := manifestDigest(value)
	if err != nil {
		return err
	}
	if actual != expected {
		return fmt.Errorf("%s mismatch", field)
	}
	return nil
}

// Reject duplicate keys before decoding into maps, just like load_json.
func readManifestValue(decoder *json.Decoder) (interface{}, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch token {
	case json.Delim('{'):
		object := map[string]interface{}{}
		for decoder.More() {
			keyToken, keyErr := decoder.Token()
			if keyErr != nil {
				return nil, keyErr
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("invalid object key")
			}
			if _, exists := object[key]; exists {
				return nil, fmt.Errorf("duplicate JSON key: %s", key)
			}
			value, valueErr := readManifestValue(decoder)
			if valueErr != nil {
				return nil, valueErr
			}
			object[key] = value
		}
		_, err = decoder.Token()
		return object, err
	case json.Delim('['):
		array := []interface{}{}
		for decoder.More() {
			value, valueErr := readManifestValue(decoder)
			if valueErr != nil {
				return nil, valueErr
			}
			array = append(array, value)
		}
		_, err = decoder.Token()
		return array, err
	default:
		return token, nil
	}
}

func manifestDigest(value interface{}) (string, error) {
	var buffer bytes.Buffer
	if err := writeManifestJSON(&buffer, value); err != nil {
		return "", err
	}
	sum := sha256.Sum256(buffer.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

// Match validation.digest's Python JSON encoding, including integer vs float
// spelling, Unicode and the scientific-notation thresholds. Go's default JSON
// encoding loses these distinctions and escapes HTML and Unicode separators.
func writeManifestJSON(buffer *bytes.Buffer, value interface{}) error {
	switch typed := value.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		buffer.WriteByte('{')
		for index, key := range keys {
			if index > 0 {
				buffer.WriteByte(',')
			}
			if err := writeManifestJSON(buffer, key); err != nil {
				return err
			}
			buffer.WriteByte(':')
			if err := writeManifestJSON(buffer, typed[key]); err != nil {
				return err
			}
		}
		buffer.WriteByte('}')
	case []interface{}:
		buffer.WriteByte('[')
		for index, entry := range typed {
			if index > 0 {
				buffer.WriteByte(',')
			}
			if err := writeManifestJSON(buffer, entry); err != nil {
				return err
			}
		}
		buffer.WriteByte(']')
	case json.Number:
		text := string(typed)
		if strings.ContainsAny(text, ".eE") {
			number, err := typed.Float64()
			if err != nil || math.IsInf(number, 0) || math.IsNaN(number) {
				return fmt.Errorf("invalid manifest number %s", text)
			}
			format := byte('f')
			if number != 0 && (math.Abs(number) < 1e-4 || math.Abs(number) >= 1e16) {
				format = 'e'
			}
			text = strconv.FormatFloat(number, format, -1, 64)
			if !strings.ContainsAny(text, ".e") {
				text += ".0"
			}
		} else if text == "-0" {
			text = "0"
		}
		buffer.WriteString(text)
	default:
		var encoded bytes.Buffer
		encoder := json.NewEncoder(&encoded)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(value); err != nil {
			return err
		}
		// Only replace actual escapes, not a literal backslash-u in a string.
		text := strings.TrimSuffix(encoded.String(), "\n")
		for i := 0; i < len(text); i++ {
			if text[i] == '\\' && i+1 < len(text) {
				if strings.HasPrefix(text[i:], `\u2028`) || strings.HasPrefix(text[i:], `\u2029`) {
					if text[i+5] == '8' {
						buffer.WriteRune('\u2028')
					} else {
						buffer.WriteRune('\u2029')
					}
					i += 5
					continue
				}
				buffer.WriteByte(text[i])
				i++
			}
			buffer.WriteByte(text[i])
		}
	}
	return nil
}
