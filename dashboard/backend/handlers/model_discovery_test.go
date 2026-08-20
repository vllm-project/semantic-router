package handlers

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type modelDiscoveryRoundTripper func(*http.Request) (*http.Response, error)

func (fn modelDiscoveryRoundTripper) Do(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func TestModelDiscoveryListsSortedUniqueModelsWithoutEchoingSecret(t *testing.T) {
	client := modelDiscoveryRoundTripper(func(request *http.Request) (*http.Response, error) {
		if request.URL.String() != "http://provider.test/v1/models" {
			t.Fatalf("discovery URL = %q", request.URL.String())
		}
		if got := request.Header.Get("Authorization"); got != "Bearer secret-value" {
			t.Fatalf("authorization = %q", got)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"data":[{"id":"z-model"},{"id":"a-model","owned_by":"team"},{"id":"a-model"}]}`)),
			Header:     make(http.Header),
		}, nil
	})
	handler := newModelDiscoveryHandler(client)
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(`{"baseUrl":"http://provider.test/v1","apiKey":"secret-value"}`))
	response := httptest.NewRecorder()

	handler(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if got := response.Body.String(); got != "{\"models\":[{\"id\":\"a-model\"},{\"id\":\"z-model\"}]}\n" || strings.Contains(got, "secret-value") {
		t.Fatalf("unexpected response: %s", got)
	}
}

func TestModelDiscoveryFallsBackToV1ForRootURL(t *testing.T) {
	paths := make([]string, 0, 2)
	client := modelDiscoveryRoundTripper(func(request *http.Request) (*http.Response, error) {
		paths = append(paths, request.URL.Path)
		status, body := http.StatusNotFound, `{}`
		if request.URL.Path == "/v1/models" {
			status, body = http.StatusOK, `{"data":[]}`
		}
		return &http.Response{StatusCode: status, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
	})
	handler := newModelDiscoveryHandler(client)
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(`{"baseUrl":"http://provider.test"}`))
	response := httptest.NewRecorder()

	handler(response, request)

	if response.Code != http.StatusOK || strings.Join(paths, ",") != "/models,/v1/models" {
		t.Fatalf("status=%d paths=%v body=%s", response.Code, paths, response.Body.String())
	}
}

func TestModelDiscoveryForwardsNativeProviderHeaders(t *testing.T) {
	client := modelDiscoveryRoundTripper(func(request *http.Request) (*http.Response, error) {
		if got := request.Header.Get("x-api-key"); got != "anthropic-secret" {
			t.Fatalf("x-api-key = %q", got)
		}
		if got := request.Header.Get("anthropic-version"); got != "2023-06-01" {
			t.Fatalf("anthropic-version = %q", got)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(`{"data":[{"id":"claude-sonnet"}]}`)),
			Header:     make(http.Header),
		}, nil
	})
	handler := newModelDiscoveryHandler(client)
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(`{"baseUrl":"https://api.anthropic.com","apiKey":"anthropic-secret","authHeader":"x-api-key","authPrefix":"","extraHeaders":{"anthropic-version":"2023-06-01"}}`))
	response := httptest.NewRecorder()

	handler(response, request)

	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "claude-sonnet") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryRejectsUnsafeExtraHeader(t *testing.T) {
	handler := newModelDiscoveryHandler(modelDiscoveryRoundTripper(func(*http.Request) (*http.Response, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}))
	request := httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(`{"baseUrl":"http://provider.test/v1","extraHeaders":{"Host":"evil.test"}}`))
	response := httptest.NewRecorder()

	handler(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}

func TestModelDiscoveryRejectsUnsafeURLAndMethod(t *testing.T) {
	handler := newModelDiscoveryHandler(modelDiscoveryRoundTripper(func(*http.Request) (*http.Response, error) {
		t.Fatal("client should not be called")
		return nil, nil
	}))

	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, modelDiscoveryPath, strings.NewReader(`{"baseUrl":"file:///etc/passwd"}`)))
	if response.Code != http.StatusBadRequest {
		t.Fatalf("unsafe URL status=%d body=%s", response.Code, response.Body.String())
	}
	response = httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodGet, modelDiscoveryPath, nil))
	if response.Code != http.StatusMethodNotAllowed || response.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("GET status=%d allow=%q", response.Code, response.Header().Get("Allow"))
	}
}
