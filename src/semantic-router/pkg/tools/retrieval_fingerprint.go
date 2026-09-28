package tools

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"sort"
	"strings"

	"github.com/openai/openai-go"
)

// retrievalFingerprintInput is the bounded, content-minimized description of
// the inputs that can change database-backed tool retrieval. It intentionally
// stores digests rather than descriptions, tags, provider configuration, or
// embedding vectors themselves.
type retrievalFingerprintInput struct {
	Backend                string                      `json:"backend"`
	ModelType              string                      `json:"model_type"`
	TargetDimension        int                         `json:"target_dimension"`
	ProviderDimension      int                         `json:"provider_dimension"`
	ProviderIdentityDigest string                      `json:"provider_identity_digest,omitempty"`
	Entries                []retrievalFingerprintEntry `json:"entries"`
}

type retrievalFingerprintEntry struct {
	Name                 string `json:"name"`
	ParameterFingerprint string `json:"parameter_fingerprint"`
	MetadataFingerprint  string `json:"metadata_fingerprint"`
	EmbeddingLength      int    `json:"embedding_length"`
	EmbeddingFingerprint string `json:"embedding_fingerprint"`
}

type retrievalMetadataFingerprintInput struct {
	Description string   `json:"description"`
	Category    string   `json:"category"`
	Tags        []string `json:"tags,omitempty"`
}

// Snapshot returns the provider-facing tool values and their retrieval
// fingerprint under one read lock. Keeping both values from the same snapshot
// prevents a catalog mutation from pairing with stale retrieval metadata.
func (db *ToolsDatabase) Snapshot() ([]openai.ChatCompletionToolParam, string) {
	if db == nil || !db.enabled {
		return []openai.ChatCompletionToolParam{}, ""
	}

	db.mu.RLock()
	defer db.mu.RUnlock()

	tools := make([]openai.ChatCompletionToolParam, len(db.entries))
	for index, entry := range db.entries {
		tools[index] = cloneChatCompletionToolParam(entry.Tool)
	}
	return tools, db.retrievalFingerprint
}

// cloneChatCompletionToolParam keeps callers from mutating the database through
// the nested JSON-schema map carried by Function.Parameters. The provider
// parameter type is a value, but its map and slice members remain reference
// types and therefore need an explicit copy at the snapshot boundary.
func cloneChatCompletionToolParam(tool openai.ChatCompletionToolParam) openai.ChatCompletionToolParam {
	tool.Function.Parameters = cloneFunctionParameters(tool.Function.Parameters)
	return tool
}

func cloneFunctionParameters(parameters openai.FunctionParameters) openai.FunctionParameters {
	if parameters == nil {
		return nil
	}

	cloned := make(openai.FunctionParameters, len(parameters))
	for key, value := range parameters {
		cloned[key] = cloneFunctionParameterValue(value)
	}
	return cloned
}

func cloneFunctionParameterValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		cloned := make(map[string]any, len(typed))
		for key, nested := range typed {
			cloned[key] = cloneFunctionParameterValue(nested)
		}
		return cloned
	case openai.FunctionParameters:
		return cloneFunctionParameters(typed)
	case []any:
		cloned := make([]any, len(typed))
		for index, nested := range typed {
			cloned[index] = cloneFunctionParameterValue(nested)
		}
		return cloned
	case []map[string]any:
		cloned := make([]map[string]any, len(typed))
		for index, nested := range typed {
			cloned[index] = cloneFunctionParameterValue(nested).(map[string]any)
		}
		return cloned
	case []string:
		return append([]string(nil), typed...)
	case json.RawMessage:
		return append(json.RawMessage(nil), typed...)
	default:
		return value
	}
}

// RetrievalFingerprint returns the current bounded fingerprint of the
// retrieval database. Disabled databases intentionally return an empty value so
// callers preserve the historical stateless selection behavior.
func (db *ToolsDatabase) RetrievalFingerprint() string {
	if db == nil || !db.enabled {
		return ""
	}

	db.mu.RLock()
	defer db.mu.RUnlock()
	return db.retrievalFingerprint
}

func (db *ToolsDatabase) computeRetrievalFingerprint(entries []ToolEntry) string {
	if db == nil || !db.enabled {
		return ""
	}

	backend := strings.ToLower(strings.TrimSpace(db.backend))
	if backend == "" && db.provider != nil {
		backend = strings.ToLower(strings.TrimSpace(db.provider.Backend()))
	}
	if backend == "" {
		backend = "candle"
	}

	providerDimension := 0
	if db.provider != nil {
		providerDimension = db.provider.Dimension()
	}

	input := retrievalFingerprintInput{
		Backend:                backend,
		ModelType:              strings.ToLower(strings.TrimSpace(db.modelType)),
		TargetDimension:        db.targetDim,
		ProviderDimension:      providerDimension,
		ProviderIdentityDigest: providerIdentityDigest(db.providerIdentity),
		Entries:                make([]retrievalFingerprintEntry, 0, len(entries)),
	}
	for _, entry := range entries {
		input.Entries = append(input.Entries, retrievalFingerprintEntry{
			Name:                 entry.Tool.Function.Name,
			ParameterFingerprint: toolParameterFingerprint(entry.Tool),
			MetadataFingerprint:  toolMetadataFingerprint(entry),
			EmbeddingLength:      len(entry.Embedding),
			EmbeddingFingerprint: embeddingFingerprint(entry.Embedding),
		})
	}

	// Embeddings are generated concurrently while loading a file. Sort all
	// entry fields before hashing so worker completion order cannot alter the
	// fingerprint.
	sort.Slice(input.Entries, func(left, right int) bool {
		return retrievalFingerprintEntryLess(input.Entries[left], input.Entries[right])
	})
	return marshalRetrievalFingerprint(input)
}

func retrievalFingerprintEntryLess(left, right retrievalFingerprintEntry) bool {
	if left.Name != right.Name {
		return left.Name < right.Name
	}
	if left.ParameterFingerprint != right.ParameterFingerprint {
		return left.ParameterFingerprint < right.ParameterFingerprint
	}
	if left.MetadataFingerprint != right.MetadataFingerprint {
		return left.MetadataFingerprint < right.MetadataFingerprint
	}
	if left.EmbeddingLength != right.EmbeddingLength {
		return left.EmbeddingLength < right.EmbeddingLength
	}
	return left.EmbeddingFingerprint < right.EmbeddingFingerprint
}

func providerIdentityDigest(identity string) string {
	identity = strings.TrimSpace(identity)
	if identity == "" {
		return ""
	}
	return sha256Hex([]byte(identity))
}

func toolMetadataFingerprint(entry ToolEntry) string {
	tags := append([]string(nil), entry.Tags...)
	// Tags are a metadata set for retrieval purposes. Sort a copy, preserving
	// duplicate values so a cardinality change still invalidates state.
	sort.Strings(tags)
	return marshalRetrievalFingerprint(retrievalMetadataFingerprintInput{
		Description: entry.Description,
		Category:    entry.Category,
		Tags:        tags,
	})
}

func toolParameterFingerprint(tool openai.ChatCompletionToolParam) string {
	parameters := tool.Function.Parameters
	if len(parameters) == 0 {
		// SemanticTool projects an omitted parameter map to an object schema;
		// treating nil and an empty map alike avoids invalidation for equivalent
		// provider values.
		parameters = openai.FunctionParameters{}
	}
	encoded, err := json.Marshal(parameters)
	if err != nil {
		return sha256Hex([]byte("retrieval:invalid_parameters:" + err.Error()))
	}
	canonical, err := canonicalizeJSON(encoded)
	if err != nil {
		return sha256Hex([]byte("retrieval:invalid_parameters:" + sha256Hex(encoded)))
	}
	return sha256Hex(canonical)
}

func embeddingFingerprint(values []float32) string {
	encoded := make([]byte, len(values)*4)
	for index, value := range values {
		binary.BigEndian.PutUint32(encoded[index*4:], math.Float32bits(value))
	}
	return sha256Hex(encoded)
}

func marshalRetrievalFingerprint(value interface{}) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		return sha256Hex([]byte("retrieval:fingerprint_marshal_error"))
	}
	return sha256Hex(encoded)
}
