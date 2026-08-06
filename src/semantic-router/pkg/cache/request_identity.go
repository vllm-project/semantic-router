package cache

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
)

const semanticQueryPlaceholder = "[semantic-cache-query]"

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
	model, query, err := ExtractQueryFromOpenAIRequest(requestBody)
	if err != nil {
		return RequestIdentity{}, err
	}

	request, err := decodeJSONMap(requestBody)
	if err != nil {
		return RequestIdentity{}, fmt.Errorf("invalid request body: %w", err)
	}

	exactRequest := cloneJSONMap(request)
	exactFingerprint, err := fingerprintJSON(exactRequest)
	if err != nil {
		return RequestIdentity{}, err
	}

	compatibilityRequest := cloneJSONMap(exactRequest)
	delete(compatibilityRequest, "model")
	semanticSafe := replaceLastUserText(compatibilityRequest)
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

func replaceLastUserText(request map[string]interface{}) bool {
	rawMessages, ok := request["messages"].([]interface{})
	if !ok {
		return false
	}
	for index := len(rawMessages) - 1; index >= 0; index-- {
		message, ok := rawMessages[index].(map[string]interface{})
		if !ok || message["role"] != "user" {
			continue
		}
		if _, ok := message["content"].(string); !ok {
			return false
		}
		message["content"] = semanticQueryPlaceholder
		return true
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
