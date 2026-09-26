package protocolcodec

import (
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDynamoRequestRejectsClientMetadataUpload(t *testing.T) {
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		for _, value := range []string{
			`{"url":"file:///tmp/client-metadata"}`,
			`{"url":"s3://client-bucket/metadata"}`,
			`{"url":"gs://client-bucket/metadata"}`,
			`{"url":"az://client-container/metadata"}`,
			`{"url":"https://client.example/metadata"}`,
			`{"url":"/tmp/client-metadata"}`,
			`{"url":""}`, `{}`, `null`,
		} {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", format, value, stream), func(t *testing.T) {
					input := `"messages":[{"role":"user","content":"Tell me about the weather."}]`
					if format == llmprotocol.OpenAIResponsesV1 {
						input = `"input":"Tell me about the weather."`
					}
					// Client-chosen destinations must never reach a backend with worker credentials.
					// Empty and null values must not become absence.
					body := []byte(fmt.Sprintf(`{"model":"model-a",%s,"stream":%t,"nvext":{"metadata_upload":%s}}`, input, stream, value))
					engine := NewBuiltinEngine()
					_, _, _, err := engine.DecodeRequestForMutation(format, body)
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_metadata_upload")
					replayed, err := engine.TranslateRequest(format, format, body, nil)
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_metadata_upload")
					if len(replayed.Body) != 0 {
						t.Fatal("rejected metadata_upload reached source replay")
					}
					mutated := false
					result, err := engine.TranslateRequest(format, format, body, func(_ *llmprotocol.Request) error {
						mutated = true
						return nil
					})
					assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_dynamo_metadata_upload")
					if mutated || len(result.Body) != 0 {
						t.Fatal("rejected metadata_upload reached mutation or outbound encoding")
					}
				})
			}
		}
	}
}
