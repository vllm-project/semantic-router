package handlers

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestModelDiscoveryHandlerListsAndSortsProviderModels(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Fatalf("authorization = %q", got)
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"zeta"},{"id":"alpha"},{"id":"alpha"}]}`))
	}))
	defer provider.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+provider.URL+`/v1","apiKey":"secret","authMode":"bearer"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(provider.Client()).ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	if got := response.Body.String(); !strings.Contains(got, `"models":["alpha","zeta"]`) {
		t.Fatalf("body = %s", got)
	}
}

func TestModelDiscoveryHandlerUsesAnthropicHeaders(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("x-api-key"); got != "secret" {
			t.Fatalf("x-api-key = %q", got)
		}
		if got := r.Header.Get("anthropic-version"); got == "" {
			t.Fatal("anthropic-version is empty")
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"claude"}]}`))
	}))
	defer provider.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+provider.URL+`/v1","apiKey":"secret","authMode":"anthropic"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(provider.Client()).ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryHandlerRejectsInvalidBaseURL(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"file:///etc/passwd","authMode":"bearer"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(nil).ServeHTTP(response, request)
	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryHandlerDoesNotForwardCredentialsAcrossRedirects(t *testing.T) {
	targetCalled := false
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		targetCalled = true
	}))
	defer target.Close()
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", target.URL)
		w.WriteHeader(http.StatusTemporaryRedirect)
	}))
	defer provider.Close()

	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(
		`{"baseUrl":"`+provider.URL+`","apiKey":"secret","authMode":"bearer"}`,
	))
	response := httptest.NewRecorder()
	ModelDiscoveryHandler(provider.Client()).ServeHTTP(response, request)

	if response.Code != http.StatusBadGateway {
		t.Fatalf("status = %d body = %s", response.Code, response.Body.String())
	}
	if targetCalled {
		t.Fatal("redirect target received the credentialed discovery request")
	}
}
