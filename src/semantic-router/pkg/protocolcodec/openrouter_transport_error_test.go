package protocolcodec

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const openRouterRateLimitBody = `{"error":{"message":"Provider returned error","code":429,` +
	`"metadata":{"raw":"{\"error\":{\"code\":429,\"status\":\"RESOURCE_EXHAUSTED\"}}","provider_name":"Provider","is_byok":false}},` +
	`"user_id":"user_1"}`

func openRouterPolicy(vendor llmprotocol.ResponseVendor) llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.UnknownFields = llmprotocol.UnknownReject
	policy.ResponseVendor = vendor
	return policy
}

func TestOpenRouterErrorEnvelopeIsDecodedOnlyForTheOpenRouterVendor(t *testing.T) {
	body := []byte(openRouterRateLimitBody)
	codecs := map[string]interface {
		DecodeTransportError([]byte, llmprotocol.Policy) (llmprotocol.TransportError, llmprotocol.Diagnostics, error)
	}{"chat": OpenAIChatCodec{}, "responses": OpenAIResponsesCodec{}}
	for name, codec := range codecs {
		t.Run(name, func(t *testing.T) {
			if _, _, err := codec.DecodeTransportError(body, openRouterPolicy("")); err == nil {
				t.Fatal("strict canonical decode accepted the OpenRouter envelope")
			}
			decoded, _, err := codec.DecodeTransportError(body, openRouterPolicy(llmprotocol.ResponseVendorOpenRouter))
			if err != nil {
				t.Fatalf("DecodeTransportError() error = %v", err)
			}
			got := decoded.Error
			if got == nil || got.Category != llmprotocol.ErrorRateLimited || got.Code != "429" ||
				got.Message != "Provider returned error" {
				t.Fatalf("error = %+v", got)
			}
			if strings.Contains(got.Message, "RESOURCE_EXHAUSTED") {
				t.Fatal("provider raw metadata must not be forwarded")
			}
		})
	}
}

func TestOpenRouterErrorCategoryFollowsStatusCode(t *testing.T) {
	for code, want := range map[string]llmprotocol.ErrorCategory{
		"400": llmprotocol.ErrorInvalidRequest,
		"401": llmprotocol.ErrorAuthentication,
		"403": llmprotocol.ErrorPermission,
		"404": llmprotocol.ErrorNotFound,
		"408": llmprotocol.ErrorUpstreamTimeout,
		"429": llmprotocol.ErrorRateLimited,
		"502": llmprotocol.ErrorUpstreamUnavailable,
	} {
		body := []byte(`{"error":{"message":"m","code":` + code + `}}`)
		decoded, _, err := (OpenAIChatCodec{}).DecodeTransportError(body, openRouterPolicy(llmprotocol.ResponseVendorOpenRouter))
		if err != nil || decoded.Error.Category != want {
			t.Errorf("code %s: category = %v err = %v, want %v", code, decoded.Error, err, want)
		}
	}
}

func TestOpenRouterErrorEnvelopeKeepsStrictValidation(t *testing.T) {
	policy := openRouterPolicy(llmprotocol.ResponseVendorOpenRouter)
	for name, body := range map[string]string{
		"duplicate field":   `{"error":{"message":"a","message":"b","code":429}}`,
		"non-object meta":   `{"error":{"message":"m","code":429,"metadata":"raw"}}`,
		"unknown error key": `{"error":{"message":"m","code":429,"extra":1}}`,
		"missing message":   `{"error":{"code":429}}`,
		"unknown top-level": `{"error":{"message":"m","code":429},"extra":1}`,
	} {
		if _, _, err := (OpenAIChatCodec{}).DecodeTransportError([]byte(body), policy); err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestEngineDecodesOpenRouterErrorWithVendorPolicy(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	policy.ResponseVendor = llmprotocol.ResponseVendorOpenRouter
	engine, err := NewEngine(NewBuiltinRegistry(), policy)
	if err != nil {
		t.Fatalf("NewEngine() error = %v", err)
	}
	decoded, _, err := engine.DecodeTransportError(llmprotocol.OpenAIChatV1, []byte(openRouterRateLimitBody))
	if err != nil || decoded.Error.Category != llmprotocol.ErrorRateLimited {
		t.Fatalf("decoded = %+v err = %v", decoded, err)
	}
}
