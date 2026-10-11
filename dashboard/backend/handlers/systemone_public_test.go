package handlers

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

func TestPublicSystemOneForwardsClientCredentialsOnlyInsideNativeEnvelope(t *testing.T) {
	t.Setenv("VLLM_SR_SYSTEMONE_LISTENER", "native")
	var calls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Method != http.MethodPost || r.URL.Path != systemone.ForwardPath || r.URL.RawQuery != "" {
			t.Errorf("unexpected management target: %s %s", r.Method, r.URL)
		}
		if r.Header.Get("Authorization") != "Bearer trusted-management" || r.Header.Get("Api-Key") != "" || r.Header.Get("Cookie") != "" {
			t.Error("public credentials escaped into management headers")
		}
		var forwarded systemone.ForwardRequest
		if err := json.NewDecoder(r.Body).Decode(&forwarded); err != nil {
			t.Fatal(err)
		}
		if forwarded.Listener != "native" || forwarded.Authorization != "Bearer client-key" || forwarded.APIKey != "client-fallback" {
			t.Error("selected listener or original client credentials changed")
		}
		if !forwarded.BackendRequest {
			t.Error("forwarding lost the native backend recursion guard")
		}
		if forwarded.Method == http.MethodGet {
			if forwarded.Path != "/v1/systemone/models" || len(forwarded.Request) != 0 {
				t.Error("discovery was changed into inference")
			}
			w.Header().Set("WWW-Authenticate", `Bearer realm="vllm-semantic-router"`)
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"code":"invalid_api_key"}}`)
			return
		}
		if !strings.Contains(string(forwarded.Request), `"questions":{"z":{"type":"noul"},"a":{"type":"choice"}}`) {
			t.Error("native question order/types changed")
		}
		_, _ = io.WriteString(w, `{"model":"judge","answers":{}}`)
	}))
	defer upstream.Close()
	handler := PublicSystemOneHandler(upstream.URL, decisionModelCredentialProvider{token: "trusted-management"})
	for _, path := range []string{"/v1/systemone", "/v1/decisions", "/v1/systemone/models"} {
		method, want := http.MethodPost, http.StatusOK
		if path == "/v1/systemone/models" {
			method, want = http.MethodGet, http.StatusUnauthorized
		}
		request := httptest.NewRequest(method, path+"?url=http://untrusted&authToken=browser", strings.NewReader(`{"model":"judge","questions":{"z":{"type":"noul"},"a":{"type":"choice"}}}`))
		request.Header.Set("Authorization", "Bearer client-key")
		request.Header.Set("Api-Key", "client-fallback")
		request.Header.Set("Cookie", "session=browser")
		request.Header.Set(systemone.BackendRequestHeader, "1")
		response := httptest.NewRecorder()
		handler(response, request)
		if response.Code != want || response.Header().Get("Cache-Control") != "no-store" {
			t.Fatalf("%s => %d %s", path, response.Code, response.Body.String())
		}
		if want == http.StatusUnauthorized && response.Header().Get("WWW-Authenticate") == "" {
			t.Fatal("frontend auth challenge was lost")
		}
	}
	if calls.Load() != 3 {
		t.Fatalf("forwarded calls=%d", calls.Load())
	}
}

func TestPublicSystemOneDoesNotFollowManagementRedirectOrLeakFailures(t *testing.T) {
	var redirected atomic.Bool
	target := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { redirected.Store(true) }))
	defer target.Close()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer upstream.Close()
	response := httptest.NewRecorder()
	PublicSystemOneHandler(upstream.URL, decisionModelCredentialProvider{token: "trusted-management"})(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"judge"}`)))
	if response.Code != 503 || redirected.Load() || strings.Contains(response.Body.String(), target.URL) {
		t.Fatalf("redirect followed or target exposed: %d %s", response.Code, response.Body.String())
	}
}

func TestPublicSystemOnePreservesBoundedHTMLLikeInput(t *testing.T) {
	state := strings.Repeat("<>&", decisionModelRequestLimit/4)
	body := `{"model":"judge","state":"` + state + `","questions":{}}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		if len(data) > decisionModelRequestLimit+8192 {
			t.Error("native payload expanded beyond the frontend transport limit")
		}
		var forwarded systemone.ForwardRequest
		if err := json.Unmarshal(data, &forwarded); err != nil {
			t.Error(err)
		}
		if string(forwarded.Request) != body {
			t.Error("forwarding rewrote the native input")
		}
		_, _ = io.WriteString(w, `{"model":"judge","answers":{}}`)
	}))
	defer upstream.Close()
	response := httptest.NewRecorder()
	PublicSystemOneHandler(upstream.URL)(response, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("bounded native request failed: %d", response.Code)
	}
}
