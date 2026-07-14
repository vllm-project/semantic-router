package handlers

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	sharednlgen "github.com/vllm-project/semantic-router/src/semantic-router/pkg/nlgen"
)

func TestBuilderNLHTTPSConnectionForwardsCredentialToExactOrigin(t *testing.T) {
	var authorization string
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		writeBuilderNLTestChatResponse(t, w, "Connection verified.")
	}))
	defer server.Close()

	_, err := callBuilderNLOpenAICompatibleMessagesWithClient(
		server.Client(),
		context.Background(),
		server.URL+"/v1/chat/completions",
		"gpt-4o-mini",
		"secret",
		sharednlgen.ChatCompletionRequest{Messages: []sharednlgen.ChatMessage{{Role: "user", Content: "ping"}}, MaxTokens: 8},
	)
	if err != nil {
		t.Fatalf("HTTPS model call failed: %v", err)
	}
	if authorization != "Bearer secret" {
		t.Fatalf("authorization = %q, want bearer credential", authorization)
	}
}

func TestBuilderNLSecureClientRejectsRedirectWithoutForwardingRequest(t *testing.T) {
	var redirectedHits atomic.Int32
	redirected := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		redirectedHits.Add(1)
		writeBuilderNLTestChatResponse(t, w, "unexpected")
	}))
	defer redirected.Close()

	origin := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, redirected.URL, http.StatusTemporaryRedirect)
	}))
	defer origin.Close()

	_, err := callBuilderNLOpenAICompatibleMessagesWithClient(
		newBuilderNLHTTPClient(),
		context.Background(),
		origin.URL+"/v1/chat/completions",
		"test-model",
		"",
		sharednlgen.ChatCompletionRequest{Messages: []sharednlgen.ChatMessage{{Role: "user", Content: "private prompt"}}, MaxTokens: 8},
	)
	if err == nil {
		t.Fatal("redirecting model endpoint unexpectedly succeeded")
	}
	if redirectedHits.Load() != 0 {
		t.Fatalf("redirect target received %d requests, want 0", redirectedHits.Load())
	}
}

func TestBuilderNLSecureClientIgnoresAmbientProxy(t *testing.T) {
	var proxyHits atomic.Int32
	proxy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		proxyHits.Add(1)
		http.Error(w, "proxy should not be used", http.StatusBadGateway)
	}))
	defer proxy.Close()
	t.Setenv("HTTP_PROXY", proxy.URL)
	t.Setenv("http_proxy", proxy.URL)

	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		writeBuilderNLTestChatResponse(t, w, "direct")
	}))
	defer target.Close()

	_, err := callBuilderNLOpenAICompatibleMessagesWithClient(
		newBuilderNLHTTPClient(),
		context.Background(),
		target.URL+"/v1/chat/completions",
		"test-model",
		"",
		sharednlgen.ChatCompletionRequest{Messages: []sharednlgen.ChatMessage{{Role: "user", Content: "ping"}}, MaxTokens: 8},
	)
	if err != nil {
		t.Fatalf("direct model call failed: %v", err)
	}
	if proxyHits.Load() != 0 {
		t.Fatalf("ambient proxy received %d requests, want 0", proxyHits.Load())
	}
}

func TestBuilderNLResponseBudgetAndErrorRedaction(t *testing.T) {
	tests := []struct {
		name     string
		handler  http.HandlerFunc
		wantText string
	}{
		{
			name: "oversized success",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = io.WriteString(w, strings.Repeat("x", int(builderNLMaxResponseBodyBytes)+1))
			},
			wantText: errBuilderNLResponseTooLarge.Error(),
		},
		{
			name: "provider error body",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				http.Error(w, "sentinel-upstream-secret", http.StatusUnauthorized)
			},
			wantText: "HTTP status 401",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(test.handler)
			defer server.Close()

			_, err := callBuilderNLOpenAICompatibleMessagesWithClient(
				newBuilderNLHTTPClient(),
				context.Background(),
				server.URL+"/v1/chat/completions",
				"test-model",
				"",
				sharednlgen.ChatCompletionRequest{Messages: []sharednlgen.ChatMessage{{Role: "user", Content: "ping"}}, MaxTokens: 8},
			)
			if err == nil || !strings.Contains(err.Error(), test.wantText) {
				t.Fatalf("error = %v, want %q", err, test.wantText)
			}
			if strings.Contains(err.Error(), "sentinel-upstream-secret") {
				t.Fatalf("error leaked upstream response body: %v", err)
			}
		})
	}
}

func TestBuilderNLConnectionValidationRejectsCredentialLeaks(t *testing.T) {
	tests := []builderNLConnection{
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "http://example.com", AccessKey: "secret"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://user:secret@example.com"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://example.com#fragment"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://example.com?token=secret"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://example.com", AccessKey: "secret\r\nX-Evil: yes"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model\nforged", BaseURL: "https://example.com"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", EndpointName: "endpoint\u202e", BaseURL: "https://example.com"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://example.com", AccessKey: "secret\n"},
		{ProviderKind: builderNLProviderOpenAICompatible, ModelName: "model", BaseURL: "https://example.com\n"},
	}
	for _, conn := range tests {
		if _, err := validateBuilderNLConnection(conn); err == nil {
			t.Fatalf("unsafe connection unexpectedly accepted: base=%q", conn.BaseURL)
		} else if strings.Contains(err.Error(), "secret") {
			t.Fatalf("validation error leaked credential: %v", err)
		}
	}
}

func TestBuilderNLTimeoutClampCannotOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	if got := clampBuilderNLTimeout(&maxInt); got != builderNLMaxTimeout {
		t.Fatalf("clampBuilderNLTimeout(max int) = %s, want %s", got, builderNLMaxTimeout)
	}
	minInt := -maxInt
	if got := clampBuilderNLTimeout(&minInt); got != builderNLMinTimeout {
		t.Fatalf("clampBuilderNLTimeout(min int) = %s, want %s", got, builderNLMinTimeout)
	}
}

func TestBuilderNLRequestErrorSentinelsDoNotWrapNetworkDetails(t *testing.T) {
	_, err := callBuilderNLOpenAICompatibleMessagesWithClient(
		errorDoer{},
		context.Background(),
		"https://example.com/v1/chat/completions",
		"model",
		"secret",
		sharednlgen.ChatCompletionRequest{Messages: []sharednlgen.ChatMessage{{Role: "user", Content: "ping"}}, MaxTokens: 8},
	)
	if !errors.Is(err, errBuilderNLRequestFailed) || strings.Contains(err.Error(), "network-secret") {
		t.Fatalf("network failure was not sanitized: %v", err)
	}
}

func TestBuilderNLRejectsCustomConnectionBeforeAnyProgressSink(t *testing.T) {
	var events []BuilderNLProgressEvent
	reporter := func(event BuilderNLProgressEvent) {
		events = append(events, event)
	}
	request := BuilderNLGenerateRequest{
		Prompt:         "create a route",
		ConnectionMode: builderNLConnectionModeCustom,
		CustomConnection: &builderNLConnection{
			ProviderKind: builderNLProviderKind("sentinel-provider\nforged-log"),
			ModelName:    "sentinel-model",
			BaseURL:      "https://example.com",
		},
	}

	_, err := generateBuilderNLDraftWithProgress(context.Background(), "", "", request, reporter)
	if err == nil {
		t.Fatal("invalid custom connection unexpectedly succeeded")
	}
	if len(events) != 0 {
		t.Fatalf("invalid metadata reached progress sink before validation: %#v", events)
	}
	if strings.Contains(err.Error(), "sentinel-provider") || strings.Contains(err.Error(), "sentinel-model") {
		t.Fatalf("validation error echoed untrusted connection metadata: %v", err)
	}
}

type errorDoer struct{}

func (errorDoer) Do(*http.Request) (*http.Response, error) {
	return nil, errors.New("network-secret")
}
