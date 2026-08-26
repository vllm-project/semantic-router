package agentruntime

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

func TestHTTPPublicInferenceReturnsTypedSanitizedHTTPFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusUnauthorized)
		_, _ = response.Write([]byte(`{"error":{"message":"must not enter Router logs"}}`))
	}))
	t.Cleanup(server.Close)

	client, err := NewHTTPPublicInferenceClient(HTTPPublicInferenceOptions{
		Endpoint: server.URL, Client: server.Client(), Codecs: protocolcodec.NewBuiltinRegistry(),
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Generate(
		context.Background(), []byte("delegated-secret"), llmprotocol.Request{Model: "vllm-sr/blend"},
		func(llmprotocol.Event) error { return nil },
	)
	var failure *publicInferenceHTTPError
	if !errors.As(err, &failure) || failure.statusCode != http.StatusUnauthorized {
		t.Fatalf("Generate() error = %v, want typed HTTP 401", err)
	}
	if err.Error() != "agent public inference returned HTTP 401" {
		t.Fatalf("Generate() error leaked the response body: %q", err)
	}
	diagnostic := safeWorkerFailureDiagnostic(err)
	if diagnostic.class != "public_inference_http" || diagnostic.upstreamStatus != http.StatusUnauthorized {
		t.Fatalf("safeWorkerFailureDiagnostic() = %#v", diagnostic)
	}
}

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

func TestHTTPPublicInferenceCompletesRealVLLMStreamShape(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		_, _ = io.Copy(io.Discard, request.Body)
		response.Header().Set("Content-Type", "text/event-stream")
		response.Header().Set(headers.VSRSelectedRecipe, "balance")
		response.Header().Set(headers.VSRSelectedDecision, "Complex")
		response.Header().Set(headers.VSRSelectedModel, "local/coder")
		response.Header().Set(headers.VSRSelectedAlgorithm, "static")
		response.Header().Set(headers.VSRResponsePath, headers.ResponsePathUpstream)
		_, _ = response.Write([]byte(
			"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"service_tier\":\"default\",\"system_fingerprint\":\"fp_1\",\"prompt_logprobs\":null,\"prompt_token_ids\":[],\"prompt_text\":null,\"kv_transfer_params\":{},\"ec_transfer_params\":null,\"metrics\":null,\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null},\"finish_reason\":null,\"logprobs\":null,\"stop_reason\":null,\"token_ids\":[],\"routed_experts\":null}]}\n\n" +
				"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}\n\n" +
				"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":526,\"completion_tokens\":3,\"total_tokens\":529,\"prompt_tokens_details\":{\"cached_tokens\":17},\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n" +
				"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
				"data: {\"id\":\"chatcmpl-real\",\"object\":\"chat.completion.chunk\",\"created\":7,\"model\":\"provider-model\",\"choices\":[],\"usage\":{\"prompt_tokens\":526,\"completion_tokens\":3,\"total_tokens\":529}}\n\n" +
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
	capture := &liveEventCapture{events: make(chan agentmanagement.LiveModelStepEvent, 4)}
	worker := &Worker{liveEvents: capture, now: time.Now}
	lease := agentmanagement.TurnLease{
		NamespaceID: uuid.NewString(), SessionID: uuid.NewString(), TurnID: uuid.NewString(), Fence: 1,
	}
	collector := newModelStepCollector(
		context.Background(), worker, lease, nil, agentmanagement.ToolPolicy{}, uuid.NewString(), 0,
	)
	observation, err := client.Generate(
		context.Background(), []byte("delegated-secret"), llmprotocol.Request{Model: "vllm-sr/blend"},
		collector.consume,
	)
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if err := collector.observe(observation); err != nil {
		t.Fatal(err)
	}
	output, err := collector.finish()
	if err != nil {
		t.Fatalf("finish() error = %v", err)
	}
	if output.StopReason != llmprotocol.StopEndTurn || len(output.Events) != 2 {
		t.Fatalf("model step output = %#v", output)
	}
	var assistant agentmanagement.AssistantDeltaEvent
	if err := json.Unmarshal(output.Events[0].Payload, &assistant); err != nil {
		t.Fatal(err)
	}
	if assistant.Delta.Text != "hello" {
		t.Fatalf("assistant text = %q", assistant.Delta.Text)
	}
	var summary agentmanagement.ModelStepSummaryEvent
	if err := json.Unmarshal(output.Events[1].Payload, &summary); err != nil {
		t.Fatal(err)
	}
	if summary.SelectedModel != "local/coder" || summary.Usage == nil ||
		summary.Usage.InputTokens != 526 || summary.Usage.OutputTokens != 3 ||
		summary.Usage.TotalTokens != 529 || summary.Usage.InputCacheReadTokens == nil ||
		*summary.Usage.InputCacheReadTokens != 17 || summary.Usage.OutputReasoningTokens == nil ||
		*summary.Usage.OutputReasoningTokens != 1 {
		t.Fatalf("model step summary = %#v", summary)
	}
	select {
	case live := <-capture.events:
		if live.Delta == nil || live.Delta.Text != "hello" {
			t.Fatalf("live delta = %#v", live)
		}
	default:
		t.Fatal("real stream did not publish its text delta live")
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
