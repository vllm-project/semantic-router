package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesInlineFileDataURL(t *testing.T) {
	const data = "JVBERi0xLjQK"
	fixtures := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `{"model":"alias","messages":[{"role":"user","content":[{"type":"file","file":{"filename":"probe.pdf","file_data":"data:%s;base64,` + data + `"}}]}]}`,
		llmprotocol.OpenAIResponsesV1:   `{"model":"alias","input":[{"role":"user","content":[{"type":"input_file","filename":"probe.pdf","file_data":"data:%s;base64,` + data + `"}]}]}`,
		llmprotocol.AnthropicMessagesV1: `{"model":"alias","max_tokens":32,"messages":[{"role":"user","content":[{"type":"document","source":{"type":"base64","media_type":"%s","data":"` + data + `"}}]}]}`,
	}
	for source, fixture := range fixtures {
		for _, mediaType := range []string{"application/pdf", "application/octet-stream"} {
			for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1} {
				t.Run(string(source)+"/"+mediaType+"/"+string(target), func(t *testing.T) {
					body := []byte(fmt.Sprintf(fixture, mediaType))
					original := bytes.Clone(body)
					result, err := NewBuiltinEngine().TranslateRequest(source, target, body, func(request *llmprotocol.Request) error {
						request.Model = "provider-model"
						return nil
					})
					if err != nil {
						t.Fatal(err)
					}
					if len(result.Diagnostics) != 0 {
						t.Fatalf("diagnostics: %+v", result.Diagnostics)
					}
					if !bytes.Equal(body, original) {
						t.Fatal("source mutated")
					}
					var wire map[string]json.RawMessage
					if err := json.Unmarshal(result.Body, &wire); err != nil {
						t.Fatal(err)
					}
					if string(wire["model"]) != `"provider-model"` {
						t.Fatalf("model not rewritten: %s", result.Body)
					}
					want := `"file_data":"data:` + mediaType + `;base64,` + data + `"`
					if !bytes.Contains(result.Body, []byte(want)) || (source != llmprotocol.AnthropicMessagesV1 && !bytes.Contains(result.Body, []byte(`"filename":"probe.pdf"`))) {
						t.Fatalf("inline file changed: %s", result.Body)
					}
				})
			}
		}
	}
}

func TestResponsesFileReferencesUnchanged(t *testing.T) {
	for _, content := range []llmprotocol.Content{
		{Kind: llmprotocol.ContentFile, FileID: "file-123", Filename: "probe.pdf"},
		{Kind: llmprotocol.ContentFile, URL: "https://example.com/probe.pdf", Filename: "probe.pdf"},
	} {
		body, err := encodeResponsesContent([]llmprotocol.Content{content}, "input")
		if err != nil {
			t.Fatal(err)
		}
		var parts []responsesContentWire
		if err := json.Unmarshal(body, &parts); err != nil {
			t.Fatal(err)
		}
		if len(parts) != 1 || parts[0].FileID != content.FileID || parts[0].FileURL != content.URL || parts[0].Filename != content.Filename || parts[0].FileData != "" {
			t.Fatalf("file reference changed: %s", body)
		}
		if _, err := encodeResponsesContent([]llmprotocol.Content{content}, "output"); err == nil {
			t.Fatal("output file unexpectedly accepted")
		}
	}
}
