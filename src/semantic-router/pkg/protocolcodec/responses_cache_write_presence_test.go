package protocolcodec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesCacheWriteEncodingPreservesPresence(t *testing.T) {
	for _, source := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, test := range []struct {
			name, details string
			write         *int64
		}{
			{name: "omitted", details: `{"cached_tokens":0}`},
			{name: "null", details: `{"cached_tokens":0,"cache_write_tokens":null}`},
			{name: "zero", details: `{"cached_tokens":0,"cache_write_tokens":0}`, write: llmprotocol.Int64(0)},
			{name: "positive", details: `{"cached_tokens":0,"cache_write_tokens":7}`, write: llmprotocol.Int64(7)},
		} {
			for _, streaming := range []bool{false, true} {
				for _, failed := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/stream=%t/failed=%t", source, test.name, streaming, failed), func(t *testing.T) {
						engine := NewBuiltinEngine()
						response := decodeCacheUsageResponse(t, engine, source, cacheUsageFixture(t, source, streaming, test.details), streaming)
						if failed {
							response.Output = nil
							response.Alternatives = nil
							response.StopReason = llmprotocol.StopError
							response.Error = &llmprotocol.ProtocolError{Category: llmprotocol.ErrorUpstreamUnavailable, Code: "provider_error", Message: "fixture failure"}
						}
						var body []byte
						var err error
						if streaming {
							body, _, err = engine.EncodeResponseStream(llmprotocol.OpenAIResponsesV1, response, llmprotocol.StreamContext{PublicModel: response.Model})
						} else {
							result, encodeErr := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
							body, err = result.Body, encodeErr
						}
						if err != nil {
							t.Fatal(err)
						}
						resource := body
						if streaming {
							resource = terminalResponsesResource(t, body, failed)
						}
						var wire struct {
							Usage struct {
								InputDetails map[string]json.RawMessage `json:"input_tokens_details"`
							} `json:"usage"`
						}
						if err = json.Unmarshal(resource, &wire); err != nil {
							t.Fatal(err)
						}
						value, present := wire.Usage.InputDetails["cache_write_tokens"]
						if test.write == nil {
							if present {
								t.Fatalf("unknown cache-write count must be omitted, got %s", value)
							}
						} else if !present || string(value) != fmt.Sprint(*test.write) {
							t.Fatalf("reported cache-write count changed: got %s, want %d", value, *test.write)
						}
						if string(wire.Usage.InputDetails["cached_tokens"]) != "0" {
							t.Fatal("reported cache-read zero changed")
						}
						uncached := (*int64)(nil)
						if test.write != nil {
							uncached = llmprotocol.Int64(10 - *test.write)
						}
						assertCacheUsage(t, response.Usage, llmprotocol.Int64(0), test.write, uncached)
						roundtrip := decodeCacheUsageResponse(t, engine, llmprotocol.OpenAIResponsesV1, body, streaming)
						assertCacheUsage(t, roundtrip.Usage, llmprotocol.Int64(0), test.write, uncached)
					})
				}
			}
		}
	}
}

func terminalResponsesResource(t *testing.T, body []byte, failed bool) []byte {
	t.Helper()
	want := "response.completed"
	if failed {
		want = "response.failed"
	}
	for _, line := range strings.Split(string(body), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event struct {
			Type     string          `json:"type"`
			Response json.RawMessage `json:"response"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
			t.Fatal(err)
		}
		if event.Type == want {
			return event.Response
		}
	}
	t.Fatalf("stream has no %s event", want)
	return nil
}
