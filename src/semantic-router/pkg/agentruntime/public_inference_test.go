package agentruntime

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestHTTPPublicInferenceCapturesClosedRouterObservationAndSemanticUsage(t *testing.T) {
	requestIDs := make(chan string, 1)
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		requestIDs <- request.Header.Get(headers.RequestID)
		if request.Header.Get("Authorization") != "Bearer delegated-secret" {
			t.Errorf("Authorization = %q", request.Header.Get("Authorization"))
		}
		_, _ = io.Copy(io.Discard, request.Body)
		response.Header().Set("Content-Type", "text/event-stream")
		response.Header().Set(headers.VSRSelectedRecipe, "balance")
		response.Header().Set(headers.VSRSelectedDecision, "Complex")
		response.Header().Set(headers.VSRSelectedModel, "remote/frontier")
		response.Header().Set(headers.VSRSelectedAlgorithm, "static")
		response.Header().Set(headers.VSRResponsePath, headers.ResponsePathUpstream)
		_, _ = response.Write([]byte(
			"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"id\":\"response_1\",\"model\":\"source-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":3,\"total_tokens\":11,\"prompt_tokens_details\":{\"cached_tokens\":2},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n" +
				"data: [DONE]\n\n",
		))
	}))
	t.Cleanup(server.Close)

	client, err := NewHTTPPublicInferenceClient(HTTPPublicInferenceOptions{
		Endpoint: server.URL, Client: server.Client(), Codecs: protocolcodec.NewBuiltinRegistry(),
	})
	if err != nil {
		t.Fatal(err)
	}
	base := time.Unix(1_700_000_000, 0)
	times := []time.Time{base, base.Add(75 * time.Millisecond), base.Add(300 * time.Millisecond)}
	client.now = func() time.Time {
		value := times[0]
		times = times[1:]
		return value
	}
	var usage *llmprotocol.Usage
	observation, err := client.Generate(
		context.Background(), []byte("delegated-secret"), llmprotocol.Request{
			Model: "vllm-sr/blend", ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceAuto},
			Messages: []llmprotocol.Message{{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
			}},
		},
		func(event llmprotocol.Event) error {
			if event.Usage != nil && event.Usage.State == llmprotocol.UsageAvailable {
				usage = event.Usage
			}
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	requestID := <-requestIDs
	if uuid.Validate(requestID) != nil || observation.RequestID != requestID {
		t.Fatalf("request ID = request %q / observation %q", requestID, observation.RequestID)
	}
	if observation.SelectedRecipe != "balance" || observation.SelectedDecision != "Complex" ||
		observation.SelectedModel != "remote/frontier" || observation.SelectedAlgorithm != "static" ||
		observation.ResponsePath != headers.ResponsePathUpstream {
		t.Fatalf("Router observation = %#v", observation)
	}
	if observation.LatencyMilliseconds != 300 || observation.TTFTMilliseconds == nil ||
		*observation.TTFTMilliseconds != 75 {
		t.Fatalf("Router timings = %#v", observation)
	}
	if usage == nil || usage.Total.Value == nil || *usage.Total.Value != 11 ||
		usage.Total.Provenance != llmprotocol.UsageAuthoritative {
		t.Fatalf("semantic usage = %#v", usage)
	}
}

func TestPublicInferenceObservationDropsUntrustedHeaderExtensions(t *testing.T) {
	values := http.Header{}
	values.Set(headers.VSRSelectedModel, "safe\nleak")
	values.Set(headers.VSRResponsePath, headers.ResponsePathError)
	observation := publicInferenceObservationFromHeaders(values, "request-1")
	if observation.SelectedModel != "" || observation.ResponsePath != "" ||
		observation.RequestID != "request-1" {
		t.Fatalf("sanitized Router observation = %#v", observation)
	}
}
