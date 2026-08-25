package modelprobe

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type captureTransport struct {
	request  *http.Request
	body     []byte
	status   int
	response string
}

func (transport *captureTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	transport.request = request.Clone(request.Context())
	transport.request.Header = request.Header.Clone()
	transport.body, _ = io.ReadAll(request.Body)
	status := transport.status
	if status == 0 {
		status = http.StatusOK
	}
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(strings.NewReader(transport.response)),
		Request:    request,
	}, nil
}

type credentialResolver struct {
	header string
	prefix string
}

func (credentialResolver) Pin(context.Context, string, string, string) (string, error) {
	return "version-one", nil
}

func (resolver credentialResolver) ResolvePinned(
	context.Context,
	string,
	string,
	string,
	string,
) (backendinvoker.Credential, error) {
	return backendinvoker.Credential{
		Header: resolver.header, Prefix: resolver.prefix, Secret: "provider-secret", Version: "version-one",
	}, nil
}

func TestProberUsesInstalledOpenAIAndAnthropicWireAdapters(t *testing.T) {
	checkedAt := time.Date(2026, 8, 23, 4, 0, 0, 0, time.UTC)
	for _, test := range []struct {
		name       string
		protocol   llmprotocol.WireFormat
		path       string
		header     string
		prefix     string
		response   string
		assertWire func(*testing.T, *captureTransport)
	}{
		{
			name: "OpenAI", protocol: llmprotocol.OpenAIChatV1,
			path: "/chat/completions", header: "Authorization", prefix: "Bearer ",
			response: `{
				"id":"chatcmpl-probe","object":"chat.completion","created":1787457600,
				"model":"upstream-model","choices":[{"index":0,"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],
				"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}
			}`,
			assertWire: func(t *testing.T, transport *captureTransport) {
				if transport.request.URL.Path != "/v1/chat/completions" ||
					transport.request.Header.Get("Authorization") != "Bearer provider-secret" {
					t.Fatalf("OpenAI request = %s, %v", transport.request.URL.Path, transport.request.Header)
				}
				var body map[string]any
				if json.Unmarshal(transport.body, &body) != nil || body["model"] != "upstream-model" ||
					body["max_completion_tokens"] != float64(1) {
					t.Fatalf("OpenAI body = %s", transport.body)
				}
			},
		},
		{
			name: "Anthropic", protocol: llmprotocol.AnthropicMessagesV1,
			path: "/v1/messages", header: "X-API-Key",
			response: `{
				"id":"msg_probe","type":"message","role":"assistant","model":"upstream-model",
				"content":[{"type":"text","text":"OK"}],"stop_reason":"end_turn",
				"usage":{"input_tokens":4,"output_tokens":1}
			}`,
			assertWire: func(t *testing.T, transport *captureTransport) {
				if transport.request.URL.Path != "/v1/messages" ||
					transport.request.Header.Get("X-API-Key") != "provider-secret" ||
					transport.request.Header.Get("Anthropic-Version") == "" {
					t.Fatalf("Anthropic request = %s, %v", transport.request.URL.Path, transport.request.Header)
				}
				var body map[string]any
				if json.Unmarshal(transport.body, &body) != nil || body["model"] != "upstream-model" ||
					body["max_tokens"] != float64(1) {
					t.Fatalf("Anthropic body = %s", transport.body)
				}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			transport := &captureTransport{response: test.response}
			prober, err := New(Options{
				Credentials: credentialResolver{header: test.header, prefix: test.prefix},
				Codecs:      protocolcodec.NewBuiltinRegistry(), Transport: transport,
				Now: func() time.Time { return checkedAt },
			})
			if err != nil {
				t.Fatal(err)
			}
			result, err := prober.Probe(context.Background(), testProbeRequest(test.protocol, test.path))
			if err != nil || !result.Available || !result.CheckedAt.Equal(checkedAt) || result.Latency < 0 {
				t.Fatalf("Probe() = %+v, %v", result, err)
			}
			test.assertWire(t, transport)
		})
	}
}

func TestProberReportsRejectedProviderResponseAsUnavailable(t *testing.T) {
	transport := &captureTransport{status: http.StatusUnauthorized, response: `{"error":{"message":"denied"}}`}
	prober, err := New(Options{
		Credentials: credentialResolver{header: "Authorization", prefix: "Bearer "},
		Codecs:      protocolcodec.NewBuiltinRegistry(), Transport: transport,
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := prober.Probe(context.Background(), testProbeRequest(
		llmprotocol.OpenAIChatV1, "/chat/completions",
	))
	if err != nil || result.Available {
		t.Fatalf("Probe() = %+v, %v", result, err)
	}
}

func testProbeRequest(protocol llmprotocol.WireFormat, path string) routingmanagement.ProbeRequest {
	origin := "https://api.example.com/v1"
	if protocol == llmprotocol.AnthropicMessagesV1 {
		origin = "https://api.example.com"
	}
	return routingmanagement.ProbeRequest{
		NamespaceID: "11111111-1111-4111-8111-111111111111", Timeout: 5 * time.Second,
		Model: routingsnapshot.Model{
			ID: "model-one", Revision: 2,
			Execution: routingsnapshot.ModelExecution{
				MaxRetries:     2,
				RetryOn:        []string{"unavailable", "timeout"},
				RequestTimeout: "5s",
				StreamTimeout:  "30s",
			},
			Backends: []routingsnapshot.Backend{{
				ID: "22222222-2222-4222-8222-222222222222", ProviderID: "provider-one",
				WireFormat: protocol, Origin: origin,
				ProviderModelID: "upstream-model", ProviderCredentialID: "33333333-3333-4333-8333-333333333333",
				Connection: routingsnapshot.BackendConnection{Path: path, Headers: probeConnectionHeaders(protocol)}, Weight: "1",
			}},
		},
	}
}

func probeConnectionHeaders(protocol llmprotocol.WireFormat) map[string]string {
	if protocol == llmprotocol.AnthropicMessagesV1 {
		return map[string]string{"Anthropic-Version": "2023-06-01"}
	}
	return nil
}
