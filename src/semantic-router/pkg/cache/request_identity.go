package cache

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const semanticQueryPlaceholder = "[response-cache-query]"

// RequestIdentity contains cache keys derived from one normalized request.
type RequestIdentity struct {
	Model                    string
	Query                    string
	ExactFingerprint         string
	CompatibilityFingerprint string
	SemanticSafe             bool
}

// BuildRequestIdentity derives exact and semantic-compatibility fingerprints
// from an OpenAI Chat Completions request.
func BuildRequestIdentity(requestBody []byte) (RequestIdentity, error) {
	request, err := decodeJSONMap(requestBody)
	if err != nil {
		return RequestIdentity{}, fmt.Errorf("invalid request body: %w", err)
	}
	model, _ := request["model"].(string)
	query := requestSemanticQuery(request)

	exactRequest := cloneJSONMap(request)
	exactFingerprint, err := fingerprintJSON(exactRequest)
	if err != nil {
		return RequestIdentity{}, err
	}

	compatibilityRequest := cloneJSONMap(exactRequest)
	delete(compatibilityRequest, "model")
	semanticSafe := replaceSemanticQuery(compatibilityRequest)
	compatibilityFingerprint, err := fingerprintJSON(compatibilityRequest)
	if err != nil {
		return RequestIdentity{}, err
	}

	return RequestIdentity{
		Model:                    model,
		Query:                    query,
		ExactFingerprint:         exactFingerprint,
		CompatibilityFingerprint: compatibilityFingerprint,
		SemanticSafe:             semanticSafe,
	}, nil
}

func requestSemanticQuery(request map[string]interface{}) string {
	if messages, ok := request["messages"].([]interface{}); ok {
		return lastRoleText(messages, "user")
	}
	switch input := request["input"].(type) {
	case string:
		return input
	case []interface{}:
		return lastRoleText(input, "user")
	default:
		return ""
	}
}

func lastRoleText(items []interface{}, role string) string {
	for index := len(items) - 1; index >= 0; index-- {
		item, ok := items[index].(map[string]interface{})
		if !ok || item["role"] != role {
			continue
		}
		return textContent(item["content"])
	}
	return ""
}

func textContent(content interface{}) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []interface{}:
		var combined string
		for _, rawPart := range typed {
			part, ok := rawPart.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := part["type"].(string)
			if partType != "text" && partType != "input_text" {
				continue
			}
			text, _ := part["text"].(string)
			if combined != "" && text != "" {
				combined += "\n"
			}
			combined += text
		}
		return combined
	default:
		return ""
	}
}

func replaceSemanticQuery(request map[string]interface{}) bool {
	if rawMessages, ok := request["messages"].([]interface{}); ok {
		return replaceLastRoleText(rawMessages, "user")
	}
	switch input := request["input"].(type) {
	case string:
		request["input"] = semanticQueryPlaceholder
		return true
	case []interface{}:
		return replaceLastRoleText(input, "user")
	default:
		return false
	}
}

func cloneJSONMap(value map[string]interface{}) map[string]interface{} {
	encoded, err := json.Marshal(value)
	if err != nil {
		return map[string]interface{}{}
	}
	cloned, err := decodeJSONMap(encoded)
	if err != nil {
		return map[string]interface{}{}
	}
	return cloned
}

func decodeJSONMap(encoded []byte) (map[string]interface{}, error) {
	decoder := json.NewDecoder(bytes.NewReader(encoded))
	decoder.UseNumber()
	var decoded map[string]interface{}
	if err := decoder.Decode(&decoded); err != nil {
		return nil, err
	}
	return decoded, nil
}

func replaceLastRoleText(rawMessages []interface{}, role string) bool {
	for index := len(rawMessages) - 1; index >= 0; index-- {
		message, ok := rawMessages[index].(map[string]interface{})
		if !ok || message["role"] != role {
			continue
		}
		switch content := message["content"].(type) {
		case string:
			message["content"] = semanticQueryPlaceholder
			return true
		case []interface{}:
			replaced := false
			for _, rawPart := range content {
				part, ok := rawPart.(map[string]interface{})
				if !ok {
					continue
				}
				partType, _ := part["type"].(string)
				if partType != "text" && partType != "input_text" {
					continue
				}
				part["text"] = semanticQueryPlaceholder
				replaced = true
			}
			return replaced
		default:
			return false
		}
	}
	return false
}

func fingerprintJSON(value interface{}) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("failed to normalize cache request: %w", err)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:]), nil
}

// FingerprintValue returns a stable SHA-256 fingerprint for JSON-compatible policy data.
func FingerprintValue(value interface{}) (string, error) {
	return fingerprintJSON(value)
}

// CombineFingerprints derives one stable key from ordered component fingerprints.
func CombineFingerprints(values ...string) string {
	sum := sha256.New()
	for _, value := range values {
		_, _ = sum.Write([]byte(value))
		_, _ = sum.Write([]byte{0})
	}
	return hex.EncodeToString(sum.Sum(nil))
}
