package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

type reentryObservation struct {
	accepted                bool
	decision                string
	iteration               int
	internalHeaderInContext bool
	internalHeadersRemoved  bool
}

func TestAuthenticatedLooperReentryAcrossSharedSecretReplicas(t *testing.T) {
	const decisionName = "authenticated-reentry"
	sharedSecret := strings.Repeat("ab", 32)

	outboundAuthenticator, err := looper.NewRequestAuthenticatorFromSharedSecret(sharedSecret)
	if err != nil {
		t.Fatalf("NewRequestAuthenticatorFromSharedSecret() outbound error = %v", err)
	}
	inboundAuthenticator, err := looper.NewRequestAuthenticatorFromSharedSecret(sharedSecret)
	if err != nil {
		t.Fatalf("NewRequestAuthenticatorFromSharedSecret() inbound error = %v", err)
	}

	observed := make(chan reentryObservation, 1)

	inboundRouter := &OpenAIRouter{
		Config:              &config.RouterConfig{},
		looperAuthenticator: inboundAuthenticator,
	}
	reentryServer := newLooperReentryTestServer(inboundRouter, observed)
	defer reentryServer.Close()

	runtimeLooper := looper.FactoryWithSelectionRegistryAndAuthenticator(
		&config.LooperConfig{Endpoint: reentryServer.URL},
		"confidence",
		nil,
		outboundAuthenticator,
	)
	response, err := runtimeLooper.Execute(context.Background(), &looper.Request{
		OriginalRequest: &openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage("verify authenticated reentry"),
			},
		},
		ModelRefs: []config.ModelRef{
			{Model: "model-a"},
			{Model: "model-b"},
		},
		Algorithm: &config.AlgorithmConfig{
			Type: "confidence",
			Confidence: &config.ConfidenceAlgorithmConfig{
				ConfidenceMethod: "avg_logprob",
				Threshold:        0.5,
			},
		},
		DecisionName: decisionName,
	})
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if got := string(response.Body); !strings.Contains(got, "authenticated reentry accepted") {
		t.Fatalf("Execute() body = %s, want authenticated fixture response", got)
	}
	if response.Model != "model-a" || response.Iterations != 1 {
		t.Fatalf(
			"Execute() selected model %q after %d iterations, want model-a after one authenticated call",
			response.Model,
			response.Iterations,
		)
	}

	observation := <-observed
	if !observation.accepted {
		t.Fatal("shared-secret reentry was not accepted as an internal Looper request")
	}
	if observation.decision != decisionName {
		t.Fatalf("authenticated decision = %q, want %q", observation.decision, decisionName)
	}
	if observation.iteration != 1 {
		t.Fatalf("authenticated iteration = %d, want 1", observation.iteration)
	}
	if observation.internalHeaderInContext {
		t.Fatal("Looper internal metadata was retained in the generic request context")
	}
	if !observation.internalHeadersRemoved {
		t.Fatal("Looper credential was not removed before upstream forwarding")
	}
}

func newLooperReentryTestServer(
	router *OpenAIRouter,
	observed chan<- reentryObservation,
) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestContext := &RequestContext{Headers: make(map[string]string)}
		processingResponse, err := router.handleRequestHeaders(
			looperReentryProcessingRequest(r),
			requestContext,
		)
		if err != nil {
			observed <- reentryObservation{}
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if writeLooperReentryImmediateResponse(w, processingResponse, observed) {
			return
		}

		observed <- observeLooperReentry(requestContext, processingResponse)
		writeLooperReentryFixture(w)
	}))
}

func looperReentryProcessingRequest(
	r *http.Request,
) *ext_proc.ProcessingRequest_RequestHeaders {
	requestHeaders := []*core.HeaderValue{
		{Key: ":method", Value: r.Method},
		{Key: ":path", Value: r.URL.RequestURI()},
	}
	for name, values := range r.Header {
		for _, value := range values {
			requestHeaders = append(requestHeaders, &core.HeaderValue{Key: name, Value: value})
		}
	}
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: requestHeaders},
		},
	}
}

func writeLooperReentryImmediateResponse(
	w http.ResponseWriter,
	response *ext_proc.ProcessingResponse,
	observed chan<- reentryObservation,
) bool {
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		return false
	}
	observed <- reentryObservation{}
	w.WriteHeader(int(immediate.GetStatus().GetCode()))
	_, _ = w.Write(immediate.GetBody())
	return true
}

func observeLooperReentry(
	requestContext *RequestContext,
	response *ext_proc.ProcessingResponse,
) reentryObservation {
	internalHeaderInContext := false
	for name := range requestContext.Headers {
		if isLooperInternalRequestHeader(name) {
			internalHeaderInContext = true
			break
		}
	}
	mutation := response.GetRequestHeaders().GetResponse().GetHeaderMutation()
	return reentryObservation{
		accepted:                requestContext.LooperRequest,
		decision:                requestContext.LooperDecision,
		iteration:               requestContext.LooperIteration,
		internalHeaderInContext: internalHeaderInContext,
		internalHeadersRemoved: looperReentryMutationRemoves(
			mutation.GetRemoveHeaders(),
			headers.VSRLooperSecret,
		),
	}
}

func writeLooperReentryFixture(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-authenticated-reentry",
		"object":  "chat.completion",
		"created": 1,
		"model":   "model-a",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": "authenticated reentry accepted",
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     1,
			"completion_tokens": 2,
			"total_tokens":      3,
		},
	})
}

func looperReentryMutationRemoves(names []string, want string) bool {
	for _, name := range names {
		if name == want {
			return true
		}
	}
	return false
}
