package llmprotocol

import (
	"encoding/json"
	"testing"
)

func TestValidateDynamoRequestNVExtAcceptsDocumentedFields(t *testing.T) {
	extension := &DynamoRequestNVExt{
		GreedSampling: Bool(true), UseRawPrompt: Bool(false),
		Annotations: []string{"worker_id", "timing"}, TokenData: []uint32{1, 2, 3},
		CacheSalt: "tenant-a", ExtraFields: []string{"worker_id", "timing", "engine_data"},
		MetadataUpload:     &DynamoMetadataUpload{URL: "https://metadata.example/upload"},
		AgentHints:         &DynamoAgentHints{Priority: int32Pointer(5), StrictPriority: uint32Pointer(1), OSL: uint32Pointer(1024), SpeculativePrefill: Bool(true), LatencySensitivity: float64Pointer(0.5)},
		RequestTimestampMS: float64Pointer(100),
		RoutingConstraints: &DynamoRoutingConstraints{RequiredTaints: []string{"gpu"}, PreferredTaints: map[string]float32{"zone-a": 0.75}},
		Router:             &DynamoRouterParams{TTFTTarget: float64Pointer(100), ITLTarget: float64Pointer(20)},
	}
	if err := ValidateDynamoRequestNVExt(extension, DefaultPolicy().Limits); err != nil {
		t.Fatalf("ValidateDynamoRequestNVExt() error = %v", err)
	}
}

func TestValidateDynamoRequestNVExtRejectsUnsupportedAndDuplicateExtraFields(t *testing.T) {
	for _, test := range []struct {
		name   string
		fields []string
		code   string
	}{
		{"unsupported", []string{"future_field"}, "unsupported_dynamo_nvext_extra_field"},
		{"duplicate", []string{"timing", "timing"}, "duplicate_dynamo_nvext_extra_field"},
	} {
		t.Run(test.name, func(t *testing.T) {
			extension := &DynamoRequestNVExt{ExtraFields: test.fields}
			requireLLMProtocolErrorCode(t, ValidateDynamoRequestNVExt(extension, DefaultPolicy().Limits), test.code)
		})
	}
}

func TestValidateDynamoRequestNVExtEnforcesStringItemAndTokenLimits(t *testing.T) {
	limits := DefaultPolicy().Limits
	limits.DynamoNVExtStringBytes = 4
	limits.DynamoNVExtItems = 1
	limits.DynamoNVExtTokenIDs = 1

	tests := []struct {
		name      string
		extension *DynamoRequestNVExt
		code      string
	}{
		{"string", &DynamoRequestNVExt{CacheSalt: "12345"}, "dynamo_nvext_string_limit"},
		{"items", &DynamoRequestNVExt{Annotations: []string{"a", "b"}}, "dynamo_nvext_items_limit"},
		{"tokens", &DynamoRequestNVExt{TokenData: []uint32{1, 2}}, "dynamo_nvext_token_limit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requireLLMProtocolErrorCode(t, ValidateDynamoRequestNVExt(test.extension, limits), test.code)
		})
	}
}

func TestValidateDynamoResponseNVExtAcceptsBoundedMetadata(t *testing.T) {
	extension := &DynamoResponseNVExt{
		WorkerID:   &DynamoWorkerInfo{PrefillWorkerID: uint64Pointer(1), DecodeDPRank: uint32Pointer(2)},
		Timing:     &DynamoTimingInfo{RequestReceivedMS: 100, TTFTMS: float64Pointer(45.2)},
		EngineData: json.RawMessage(`{"backend":"vllm"}`),
		StopReason: json.RawMessage(`"length"`), CompletionTokenIDs: []uint32{10, 11},
		PromptLogprobs: []map[uint32]DynamoPromptLogprobEntry{nil, {42: {Logprob: -0.25, Rank: uint32Pointer(1)}}},
	}
	if err := ValidateDynamoResponseNVExt(extension, DefaultPolicy().Limits); err != nil {
		t.Fatalf("ValidateDynamoResponseNVExt() error = %v", err)
	}
}

func TestValidateDynamoResponseNVExtRejectsMalformedDeepAndOversizedMetadata(t *testing.T) {
	tests := []struct {
		name   string
		limits Limits
		raw    json.RawMessage
		code   string
	}{
		{"malformed", DefaultPolicy().Limits, json.RawMessage(`{"bad":`), "invalid_dynamo_nvext"},
		{"duplicate", DefaultPolicy().Limits, json.RawMessage(`{"key":1,"key":2}`), "invalid_dynamo_nvext"},
		{"deep", func() Limits { limits := DefaultPolicy().Limits; limits.JSONDepth = 1; return limits }(), json.RawMessage(`{"a":{"b":1}}`), "invalid_dynamo_nvext"},
		{"oversized", func() Limits { limits := DefaultPolicy().Limits; limits.DynamoNVExtBytes = 8; return limits }(), json.RawMessage(`{"value":"too large"}`), "dynamo_nvext_size_limit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			extension := &DynamoResponseNVExt{EngineData: test.raw}
			requireLLMProtocolErrorCode(t, ValidateDynamoResponseNVExt(extension, test.limits), test.code)
		})
	}
}

func TestValidateDynamoResponseNVExtRejectsTokenLimit(t *testing.T) {
	extension := &DynamoResponseNVExt{TokenIDs: []uint32{1, 2}}
	limits := DefaultPolicy().Limits
	limits.DynamoNVExtTokenIDs = 1
	requireLLMProtocolErrorCode(t, ValidateDynamoResponseNVExt(extension, limits), "dynamo_nvext_token_limit")
}

func int32Pointer(value int32) *int32       { return &value }
func uint32Pointer(value uint32) *uint32    { return &value }
func uint64Pointer(value uint64) *uint64    { return &value }
func float64Pointer(value float64) *float64 { return &value }
