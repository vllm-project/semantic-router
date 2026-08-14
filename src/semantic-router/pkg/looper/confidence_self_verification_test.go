package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestConfidenceLooperAccountsForSelfVerificationCall(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var requestBody struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		content := "candidate secret answer"
		usage := map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
		if calls == 2 {
			content = `{"confidence":0.95,"reason":"verifier private rationale"}`
			usage = map[string]int{"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":      "chatcmpl-self-verify",
			"object":  "chat.completion",
			"created": 0,
			"model":   requestBody.Model,
			"choices": []map[string]interface{}{{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": content},
				"finish_reason": "stop",
			}},
			"usage": usage,
		})
	}))
	defer server.Close()

	request := confidenceRequest("skip", "self_verify")
	request.ModelRefs = request.ModelRefs[:1]
	response, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), request)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if calls != 2 || response.Iterations != 2 {
		t.Fatalf("calls/iterations = %d/%d, want 2/2", calls, response.Iterations)
	}
	if !slices.Equal(response.ModelsUsed, []string{"small", "small"}) {
		t.Fatalf("models used = %v, want generation and verifier", response.ModelsUsed)
	}
	if response.Usage != (TokenUsage{PromptTokens: 6, CompletionTokens: 4, TotalTokens: 10}) {
		t.Fatalf("usage = %+v, want generation plus verifier", response.Usage)
	}
	if !strings.Contains(string(response.Body), "candidate secret answer") || strings.Contains(string(response.Body), "verifier private rationale") {
		t.Fatalf("response did not isolate candidate content from verifier evidence: %s", response.Body)
	}
	var completion struct {
		Usage TokenUsage `json:"usage"`
	}
	if err := json.Unmarshal(response.Body, &completion); err != nil {
		t.Fatalf("decode response body: %v", err)
	}
	if completion.Usage != response.Usage {
		t.Fatalf("body usage = %+v, want %+v", completion.Usage, response.Usage)
	}
}

type selfVerificationFailureCase struct {
	name              string
	nonOK             bool
	wantErrorFragment string
	wantFailUsage     TokenUsage
	wantSkipUsage     TokenUsage
}

func TestConfidenceLooperSelfVerificationErrorsHonorOnError(t *testing.T) {
	failures := []selfVerificationFailureCase{
		{
			name:              "parse error after paid response",
			wantErrorFragment: "could not parse verifier result",
			wantFailUsage:     TokenUsage{PromptTokens: 6, CompletionTokens: 4, TotalTokens: 10},
			wantSkipUsage:     TokenUsage{PromptTokens: 12, CompletionTokens: 8, TotalTokens: 20},
		},
		{
			name:              "model call error",
			nonOK:             true,
			wantErrorFragment: "verifier model call failed",
			wantFailUsage:     TokenUsage{PromptTokens: 2, CompletionTokens: 1, TotalTokens: 3},
			wantSkipUsage:     TokenUsage{PromptTokens: 8, CompletionTokens: 5, TotalTokens: 13},
		},
	}

	for _, failure := range failures {
		for _, onError := range []string{"fail", "skip"} {
			t.Run(failure.name+"/"+onError, func(t *testing.T) {
				runSelfVerificationFailureCase(t, failure, onError)
			})
		}
	}
}

func runSelfVerificationFailureCase(t *testing.T, failure selfVerificationFailureCase, onError string) {
	t.Helper()
	server, calls := newSelfVerificationFailureServer(t, failure)
	defer server.Close()
	response, err := NewConfidenceLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(
		context.Background(),
		confidenceRequest(onError, "self_verify"),
	)
	if onError == "fail" {
		assertSelfVerificationFailResult(t, failure, response, err, calls.Load())
		return
	}
	assertSelfVerificationSkipResult(t, failure, response, err, calls.Load())
}

func newSelfVerificationFailureServer(
	t *testing.T,
	failure selfVerificationFailureCase,
) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		serveSelfVerificationFailure(t, w, r, int(calls.Add(1)), failure)
	}))
	return server, &calls
}

func serveSelfVerificationFailure(
	t *testing.T,
	w http.ResponseWriter,
	r *http.Request,
	call int,
	failure selfVerificationFailureCase,
) {
	t.Helper()
	var requestBody struct {
		Model string `json:"model"`
	}
	if err := json.NewDecoder(r.Body).Decode(&requestBody); err != nil {
		t.Errorf("decode request: %v", err)
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	if call == 2 && failure.nonOK {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(strings.Repeat("private candidate echo must stay hidden", 400)))
		return
	}
	content := requestBody.Model + " private candidate answer"
	usage := map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
	if call%2 == 0 {
		content = `{"confidence":0.95,"reason":"private verifier rationale"}`
		usage = map[string]int{"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
	}
	if call == 2 {
		content = "malformed private verifier evidence"
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id": "chatcmpl-self-verify-error", "object": "chat.completion", "created": 0, "model": requestBody.Model,
		"choices": []map[string]interface{}{{
			"index": 0, "message": map[string]interface{}{"role": "assistant", "content": content}, "finish_reason": "stop",
		}},
		"usage": usage,
	})
}

func assertSelfVerificationFailResult(
	t *testing.T,
	failure selfVerificationFailureCase,
	response *Response,
	err error,
	calls int32,
) {
	t.Helper()
	if err == nil || !strings.Contains(err.Error(), failure.wantErrorFragment) || response != nil || calls != 2 {
		t.Fatalf("fail result = response %+v / error %v / calls %d", response, err, calls)
	}
	evidence, ok := ConfidenceEvidenceFromError(err)
	if !ok || evidence.Iterations != 2 || !slices.Equal(evidence.ModelsUsed, []string{"small", "small"}) ||
		evidence.Usage != failure.wantFailUsage {
		t.Fatalf("partial evidence = %+v (present=%v), want usage %+v", evidence, ok, failure.wantFailUsage)
	}
	if strings.Contains(err.Error(), "private") {
		t.Fatalf("partial error leaked verifier or candidate content: %v", err)
	}
}

func assertSelfVerificationSkipResult(
	t *testing.T,
	failure selfVerificationFailureCase,
	response *Response,
	err error,
	calls int32,
) {
	t.Helper()
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}
	if calls != 4 || response.Iterations != 4 || response.Usage != failure.wantSkipUsage ||
		!slices.Equal(response.ModelsUsed, []string{"small", "small", "large", "large"}) {
		t.Fatalf("skip result = calls %d response %+v, want usage %+v", calls, response, failure.wantSkipUsage)
	}
	body := string(response.Body)
	if !strings.Contains(body, "large private candidate answer") ||
		strings.Contains(body, "small private candidate answer") ||
		strings.Contains(body, "malformed private verifier evidence") ||
		strings.Contains(body, "private verifier rationale") {
		t.Fatalf("response did not isolate the successful candidate from skipped evidence: %s", body)
	}
}
