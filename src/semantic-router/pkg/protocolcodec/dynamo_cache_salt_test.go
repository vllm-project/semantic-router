package protocolcodec

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestChatCacheSaltIsIndependentOfDynamoNVExt(t *testing.T) {
	for _, tc := range []struct {
		name, fields string
		dynamo       bool
	}{
		{"top-level", `"cache_salt":"original"`, false},
		{"nvext", `"nvext":{"cache_salt":"dynamo-salt"}`, true},
		{"both", `"cache_salt":"original","nvext":{"cache_salt":"dynamo-salt"}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := NewBuiltinEngine()
			body := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],` + tc.fields + `}`)
			request, envelope, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body)
			if err != nil {
				t.Fatal(err)
			}
			if (envelope.Dynamo != nil) != tc.dynamo {
				t.Fatalf("unexpected Dynamo envelope: %+v", envelope.Dynamo)
			}
			if tc.name == "nvext" && request.CacheSalt != nil {
				t.Fatal("nvext cache salt leaked into neutral request")
			}
			if tc.name != "nvext" && (request.CacheSalt == nil || *request.CacheSalt != "original") {
				t.Fatal("missing neutral cache salt")
			}
			salt := "changed"
			request.CacheSalt = &salt
			request.Generation++
			encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			assertJSONField(t, encoded.Body, "cache_salt", "changed")
			if tc.dynamo {
				assertNestedJSONField(t, encoded.Body, "nvext", "cache_salt", "dynamo-salt")
			}
		})
	}
}
