package handlers

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const (
	decisionTestCapabilities = `{"deployments":[{"id":"selected","model":"served","ready":true,"question_types":["choice","score","noul","span","set"],"surfaces":["decisions"]}]}`
	decisionTestBody         = `{"deployment":"selected","request":{"model":"unselected","state":"张三","questions":{"z":{"type":"noul","instructions":"z?"},"a":{"type":"noul","instructions":"a?"}},"options":{"return_meta":true}}}`
	decisionTestResponse     = `{"model":"served","answers":{"z":{"type":"noul","noul":0.8},"a":{"type":"noul","error":"invalid_question","message":"unsupported"}},"spans":{"names":[{"start":0,"end":2,"text":"张三","label":"person","probability":0.9}]},"usage":{"input_tokens":5,"output_tokens":0}}`
)

type decisionModelCredentialProvider struct {
	token string
}

func (provider decisionModelCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, nil
}

func TestDecisionModelRouterForwardingPreservesNativeContract(t *testing.T) {
	var posts atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer trusted-management" || r.Header.Get("Cookie") != "" || r.URL.RawQuery != "" {
			t.Errorf("browser credentials escaped: %v %s", r.Header, r.URL)
		}
		if r.URL.Path != decisionModelDiagnosticPath {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, decisionTestCapabilities)
			return
		}
		posts.Add(1)
		body, _ := io.ReadAll(r.Body)
		var payload map[string]json.RawMessage
		_ = json.Unmarshal(body, &payload)
		if !strings.Contains(string(payload["request"]), `"z":{"type":"noul","instructions":"z?"},"a"`) {
			t.Errorf("question order changed %s", body)
		}
		w.Header().Set("Server-Timing", "total;dur=2")
		_, _ = io.WriteString(w, decisionTestResponse)
	}))
	defer upstream.Close()
	handler := DecisionModelHandler(upstream.URL, decisionModelCredentialProvider{token: "trusted-management"})
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		path := "/api/decision-model/capabilities"
		if method == http.MethodPost {
			path = "/api/decision-model/test"
		}
		request := httptest.NewRequest(method, path+"?url=http://evil&authToken=browser", strings.NewReader(decisionTestBody))
		request.Header.Set("Authorization", "Bearer browser")
		request.Header.Set("Cookie", "session=browser")
		response := httptest.NewRecorder()
		handler(response, request)
		if response.Code != 200 {
			t.Fatalf("%s %d %s", method, response.Code, response.Body.String())
		}
		if method == http.MethodGet && !strings.Contains(response.Body.String(), `"serving_mode":"router"`) {
			t.Fatalf("mode absent: %s", response.Body.String())
		}
		if method == http.MethodPost && (response.Body.String() != decisionTestResponse || response.Header().Get("Server-Timing") == "") {
			t.Fatalf("native response changed: %s", response.Body.String())
		}
	}
	if posts.Load() != 1 {
		t.Fatalf("posts=%d", posts.Load())
	}
}

func TestDecisionModelEngineUsesOnlyKnownModel(t *testing.T) {
	var posted string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != decisionModelDiagnosticPath {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if r.Method == http.MethodGet {
			_, _ = io.WriteString(w, strings.Replace(decisionTestCapabilities, `{"deployments":`, `{"serving_mode":"engine","deployments":`, 1))
			return
		}
		body, _ := io.ReadAll(r.Body)
		posted = string(body)
		_, _ = io.WriteString(w, decisionTestResponse)
	}))
	defer upstream.Close()
	handler := DecisionModelHandler(upstream.URL)
	response := httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, "/api/decision-model/test", strings.NewReader(decisionTestBody)))
	if response.Code != 200 || !strings.Contains(posted, `"deployment":"selected"`) || !strings.Contains(posted, `"request"`) {
		t.Fatalf("status=%d posted=%s body=%s", response.Code, posted, response.Body.String())
	}
	response = httptest.NewRecorder()
	handler(response, httptest.NewRequest(http.MethodPost, "/api/decision-model/test", strings.NewReader(strings.Replace(decisionTestBody, `"deployment":"selected"`, `"deployment":"http://evil"`, 1))))
	if response.Code != 404 {
		t.Fatalf("unknown model accepted: %d", response.Code)
	}
}

func TestDecisionModelRejectsInvalidInputUnavailableModelsAndNativeErrors(t *testing.T) {
	for _, tc := range []struct {
		name, capabilities, body string
		status                   int
		want                     int
	}{
		{"invalid wrapper", decisionTestCapabilities, `{"deployment":"selected","request":{},"url":"http://evil"}`, 200, 400},
		{"too large", decisionTestCapabilities, `{"deployment":"selected","request":{"state":"` + strings.Repeat("x", decisionModelRequestLimit) + `"}}`, 200, 413},
		{"unknown model", decisionTestCapabilities, strings.Replace(decisionTestBody, `"deployment":"selected"`, `"deployment":"missing"`, 1), 200, 404},
		{"loading", strings.Replace(decisionTestCapabilities, `"ready":true`, `"ready":false`, 1), decisionTestBody, 200, 503},
		{"specialist", strings.Replace(decisionTestCapabilities, `["choice","score","noul","span","set"]`, `[]`, 1), decisionTestBody, 200, 422},
		{"native overload", decisionTestCapabilities, decisionTestBody, 429, 429},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					_, _ = io.WriteString(w, tc.capabilities)
					return
				}
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, `{"error":{"code":"overloaded","message":"busy"}}`)
			}))
			defer upstream.Close()
			response := httptest.NewRecorder()
			DecisionModelHandler(upstream.URL)(response, httptest.NewRequest(http.MethodPost, "/api/decision-model/test", strings.NewReader(tc.body)))
			if response.Code != tc.want {
				t.Fatalf("got %d want %d: %s", response.Code, tc.want, response.Body.String())
			}
		})
	}
}

func TestDecisionModelDoesNotFollowRedirectsAndCancelsUpstream(t *testing.T) {
	var leaked atomic.Int32
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked.Add(1) }))
	defer target.Close()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, target.URL, http.StatusTemporaryRedirect)
	}))
	defer upstream.Close()
	response := httptest.NewRecorder()
	DecisionModelHandler(upstream.URL, decisionModelCredentialProvider{token: "secret"})(response, httptest.NewRequest(http.MethodGet, "/api/decision-model/capabilities", nil))
	if response.Code != 502 || leaked.Load() != 0 || strings.Contains(response.Body.String(), upstream.URL) {
		t.Fatalf("redirect/transport leaked: %d %s", response.Code, response.Body.String())
	}
	canceled := make(chan struct{})
	blocked := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done(); close(canceled) }))
	defer blocked.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	response = httptest.NewRecorder()
	DecisionModelHandler(blocked.URL)(response, httptest.NewRequest(http.MethodGet, "/api/decision-model/capabilities", nil).WithContext(ctx))
	if response.Code != 504 {
		t.Fatalf("deadline=%d body=%s", response.Code, response.Body.String())
	}
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("upstream not canceled")
	}
}
