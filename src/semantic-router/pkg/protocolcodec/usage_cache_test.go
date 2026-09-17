package protocolcodec

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOpenAIResponseCacheUsagePresence(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, test := range []struct {
			name, details         string
			read, write, uncached *int64
		}{
			{name: "omitted"},
			{name: "null details", details: `null`},
			{name: "empty details", details: `{}`},
			{name: "null buckets", details: `{"cached_tokens":null,"cache_write_tokens":null,"created_cache_tokens":null}`},
			{name: "read only", details: `{"cached_tokens":3}`, read: llmprotocol.Int64(3)},
			{name: "explicit zero read", details: `{"cached_tokens":0}`, read: llmprotocol.Int64(0)},
			{name: "write only", details: `{"created_cache_tokens":2}`, write: llmprotocol.Int64(2)},
			{name: "vllm", details: `{"cached_tokens":3,"created_cache_tokens":2}`, read: llmprotocol.Int64(3), write: llmprotocol.Int64(2), uncached: llmprotocol.Int64(5)},
			{name: "both aliases agree", details: `{"cached_tokens":3,"cache_write_tokens":2,"created_cache_tokens":2}`, read: llmprotocol.Int64(3), write: llmprotocol.Int64(2), uncached: llmprotocol.Int64(5)},
			{name: "canonical null", details: `{"cached_tokens":3,"cache_write_tokens":null,"created_cache_tokens":0}`, read: llmprotocol.Int64(3), write: llmprotocol.Int64(0), uncached: llmprotocol.Int64(7)},
			{name: "alias null", details: `{"cached_tokens":3,"cache_write_tokens":0,"created_cache_tokens":null}`, read: llmprotocol.Int64(3), write: llmprotocol.Int64(0), uncached: llmprotocol.Int64(7)},
		} {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", format, test.name, streaming), func(t *testing.T) {
					engine := NewBuiltinEngine()
					body := cacheUsageFixture(t, format, streaming, test.details)
					response := decodeCacheUsageResponse(t, engine, format, body, streaming)
					assertCacheUsage(t, response.Usage, test.read, test.write, test.uncached)
					for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
						var encoded []byte
						var err error
						if streaming {
							encoded, _, err = engine.EncodeResponseStream(target, response, llmprotocol.StreamContext{PublicModel: response.Model, Options: llmprotocol.StreamOptions{IncludeUsage: boolPointer(true)}})
						} else {
							result, encodeErr := engine.EncodeResponse(target, response, llmprotocol.Envelope{})
							encoded, err = result.Body, encodeErr
						}
						if err != nil {
							t.Fatal(err)
						}
						roundtrip := decodeCacheUsageResponse(t, engine, target, encoded, streaming)
						assertCacheUsage(t, roundtrip.Usage, test.read, test.write, test.uncached)
					}
				})
			}
		}
	}
}

func TestOpenAIResponseCacheUsageRejectsInvalidEvidence(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for name, details := range map[string]string{
			"conflicting aliases": `{"cached_tokens":3,"cache_write_tokens":2,"created_cache_tokens":4}`,
			"negative":            `{"created_cache_tokens":-1}`,
			"fraction":            `{"created_cache_tokens":1.5}`,
			"string":              `{"created_cache_tokens":"2"}`,
			"unknown field":       `{"created_cache_tokens":2,"unknown_tokens":1}`,
			"excessive read":      `{"cached_tokens":11}`,
			"excessive subtotal":  `{"cached_tokens":9,"created_cache_tokens":2}`,
		} {
			for _, streaming := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", format, name, streaming), func(t *testing.T) {
					engine := NewBuiltinEngine()
					body := cacheUsageFixture(t, format, streaming, details)
					var err error
					if streaming {
						_, _, err = engine.DecodeResponseStream(format, body, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
					} else {
						_, _, _, err = engine.DecodeResponse(format, body)
					}
					if err == nil {
						t.Fatal("invalid cache accounting was accepted")
					}
					if name == "conflicting aliases" {
						assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "conflicting_cache_usage")
					}
				})
			}
		}
	}
}

func TestVLLMPromptUsageAcceptsMultimodalDetails(t *testing.T) {
	for _, streaming := range []bool{false, true} {
		for _, detail := range []string{`null`, `{"image":2}`} {
			body := cacheUsageFixture(t, llmprotocol.OpenAIChatV1, streaming, `{"cached_tokens":0,"created_cache_tokens":0,"multimodal_tokens":`+detail+`}`)
			response := decodeCacheUsageResponse(t, NewBuiltinEngine(), llmprotocol.OpenAIChatV1, body, streaming)
			assertCacheUsage(t, response.Usage, llmprotocol.Int64(0), llmprotocol.Int64(0), llmprotocol.Int64(10))
		}
	}
}

func decodeCacheUsageResponse(t *testing.T, engine *Engine, format llmprotocol.WireFormat, body []byte, streaming bool) llmprotocol.Response {
	t.Helper()
	var response llmprotocol.Response
	var err error
	if streaming {
		response, _, err = engine.DecodeResponseStream(format, body, llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model"})
	} else {
		response, _, _, err = engine.DecodeResponse(format, body)
	}
	if err != nil {
		t.Fatalf("cache usage decode failed: %v\n%s", err, body)
	}
	return response
}

func assertCacheUsage(t *testing.T, usage llmprotocol.Usage, read, write, uncached *int64) {
	t.Helper()
	for name, pair := range map[string][2]*int64{
		"read":     {usage.InputCacheRead.Value, read},
		"write":    {usage.InputCacheWrite.Value, write},
		"uncached": {usage.InputUncached.Value, uncached},
	} {
		if !reflect.DeepEqual(pair[0], pair[1]) {
			t.Fatalf("%s bucket differs: usage=%+v want=%v", name, usage, pair[1])
		}
	}
	if tokenValue(usage.InputTotal) != 10 || tokenValue(usage.OutputTotal) != 2 || tokenValue(usage.Total) != 12 {
		t.Fatalf("usage totals changed: %+v", usage)
	}
	for _, count := range []llmprotocol.TokenCount{usage.InputCacheRead, usage.InputCacheWrite} {
		if count.Value != nil && count.Provenance != llmprotocol.UsageAuthoritative {
			t.Fatalf("reported cache evidence lost authority: %+v", count)
		}
		if count.Value == nil && count.Provenance != llmprotocol.UsageUnknown && count.Provenance != "" {
			t.Fatalf("unknown cache usage has known provenance: %+v", count)
		}
	}
}

func cacheUsageFixture(t *testing.T, format llmprotocol.WireFormat, streaming bool, details string) []byte {
	t.Helper()
	input, output, detailKey := "prompt_tokens", "completion_tokens", "prompt_tokens_details"
	if format == llmprotocol.OpenAIResponsesV1 {
		input, output, detailKey = "input_tokens", "output_tokens", "input_tokens_details"
	}
	usage := map[string]any{input: 10, output: 2, "total_tokens": 12}
	if details != "" {
		usage[detailKey] = json.RawMessage(details)
	}
	replace := func(raw []byte) []byte {
		var value map[string]any
		if err := json.Unmarshal(raw, &value); err != nil {
			t.Fatal(err)
		}
		resource := value
		if nested, ok := value["response"].(map[string]any); ok {
			resource = nested
		}
		if _, exists := resource["usage"]; exists {
			resource["usage"] = usage
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		return encoded
	}
	if !streaming {
		return replace(responseFixture(format))
	}
	lines := strings.Split(string(streamFixture(format)), "\n")
	for i, line := range lines {
		if strings.HasPrefix(line, "data: {") {
			lines[i] = "data: " + string(replace([]byte(strings.TrimPrefix(line, "data: "))))
		}
	}
	return []byte(strings.Join(lines, "\n"))
}
