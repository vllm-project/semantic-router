package handlers

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDecisionRoutesUsesManagementIdentityAndPreservesNativeBody(t *testing.T) {
	const input = `{"model":"vllm-sr/auto","request":{"state":"text","questions":{"task":{"type":"choice","criteria":{"z":"Z","a":"A"}}}}}`
	const routes = `{"available":true,"routes":[{"model":"vllm-sr/auto","recipe":"native","algorithms":["cascade"],"question_types":["choice","score","noul"],"timeout_ms":120000}]}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != decisionModelRoutesPath || r.URL.RawQuery != "" || r.Header.Get("Authorization") != "Bearer management" || r.Header.Get("Cookie") != "" || r.Header.Get("Api-Key") != "" {
			t.Errorf("unexpected management request: %s %v", r.URL, r.Header)
		}
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, routes)
			return
		}
		body, _ := io.ReadAll(r.Body)
		if string(body) != input {
			t.Errorf("native input changed: %s", body)
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = io.WriteString(w, `{"error":{"code":"systemone_unresolved"}}`)
	}))
	defer upstream.Close()
	handler := DecisionModelRoutesHandler(upstream.URL, decisionModelCredentialProvider{token: "management"})
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		r := httptest.NewRequest(method, "/api/decision-model/routes?url=http://evil", strings.NewReader(input))
		r.Header.Set("Authorization", "Bearer browser")
		r.Header.Set("Cookie", "session=browser")
		r.Header.Set("Api-Key", "client-secret")
		w := httptest.NewRecorder()
		handler(w, r)
		if method == http.MethodGet && (w.Code != 200 || w.Body.String() != routes) {
			t.Fatalf("discovery: %d %s", w.Code, w.Body.String())
		}
		if method == http.MethodPost && (w.Code != 503 || !strings.Contains(w.Body.String(), "systemone_unresolved")) {
			t.Fatalf("execution: %d %s", w.Code, w.Body.String())
		}
	}
}

func TestDecisionRoutesRejectsMalformedWrappers(t *testing.T) {
	for _, input := range []string{`{}`, `{"model":"auto","request":null}`, `{"model":"auto","request":[],"url":"http://evil"}`, `{"model":"auto","request":{}} {}`} {
		w := httptest.NewRecorder()
		DecisionModelRoutesHandler("http://127.0.0.1:1")(w, httptest.NewRequest(http.MethodPost, "/api/decision-model/routes", strings.NewReader(input)))
		if w.Code != 400 {
			t.Fatalf("accepted malformed request: %d %s", w.Code, input)
		}
	}
}

func TestDecisionRoutesCancelsExecutionAndRejectsRedirects(t *testing.T) {
	received := make(chan struct{})
	canceled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			http.Redirect(w, r, "http://other.invalid", http.StatusTemporaryRedirect)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		close(received)
		<-r.Context().Done()
		close(canceled)
	}))
	defer upstream.Close()
	defer upstream.CloseClientConnections()
	handler := DecisionModelRoutesHandler(upstream.URL)
	w := httptest.NewRecorder()
	handler(w, httptest.NewRequest(http.MethodGet, "/api/decision-model/routes", nil))
	if w.Code != 502 {
		t.Fatalf("redirect accepted: %d", w.Code)
	}
	ctx, cancel := context.WithCancel(t.Context())
	done := make(chan struct{})
	go func() {
		handler(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/api/decision-model/routes", strings.NewReader(`{"model":"auto","request":{}}`)).WithContext(ctx))
		close(done)
	}()
	select {
	case <-received:
	case <-time.After(time.Second):
		cancel()
		t.Fatal("upstream request did not arrive")
	}
	cancel()
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("upstream was not canceled")
	}
	<-done
}

func TestDecisionRoutesRejectsOversizedBody(t *testing.T) {
	w := httptest.NewRecorder()
	body := `{"model":"auto","request":{"state":"` + strings.Repeat("x", decisionModelRequestLimit) + `"}}`
	DecisionModelRoutesHandler("http://127.0.0.1:1")(w, httptest.NewRequest(http.MethodPost, "/api/decision-model/routes", strings.NewReader(body)))
	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversized body: %d", w.Code)
	}
}
