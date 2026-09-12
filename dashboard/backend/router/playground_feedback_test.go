package router

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

type fakePlaygroundFeedbackStore struct {
	validateErr    error
	bindSessionID  string
	bindReplayID   string
	bindTargetRef  string
	completeCalled bool
	claimCalled    bool
	claimKey       string
	claimErr       error
	finishCalled   bool
	finishSuccess  bool
	finishResults  []bool
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func (s *fakePlaygroundFeedbackStore) BindPlaygroundReplay(_ context.Context, sessionID, replayID, targetRef string) error {
	s.bindSessionID = sessionID
	s.bindReplayID = replayID
	s.bindTargetRef = targetRef
	return nil
}

func (s *fakePlaygroundFeedbackStore) CompletePlaygroundReplay(_ context.Context, _, _ string) error {
	s.completeCalled = true
	return nil
}

func (s *fakePlaygroundFeedbackStore) ValidatePlaygroundReplay(context.Context, string, string, string) error {
	return s.validateErr
}

func (s *fakePlaygroundFeedbackStore) ClaimPlaygroundReplay(
	context.Context,
	string,
	string,
	string,
	int,
	time.Duration,
) (string, error) {
	s.claimCalled = true
	if s.claimErr != nil {
		return "", s.claimErr
	}
	if s.claimKey == "" {
		s.claimKey = "server-owned-feedback-key"
	}
	return s.claimKey, nil
}

func (s *fakePlaygroundFeedbackStore) FinishPlaygroundReplay(_ context.Context, _, _ string, success bool) error {
	s.finishCalled = true
	s.finishSuccess = success
	s.finishResults = append(s.finishResults, success)
	return nil
}

func TestTrackPlaygroundReplayResponseCompletesAfterBodyEOF(t *testing.T) {
	store := &fakePlaygroundFeedbackStore{}
	request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{SessionID: "session-1"}))
	response := &http.Response{
		StatusCode: http.StatusOK,
		Request:    request,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader("streamed response")),
	}
	response.Header.Set(headers.RouterReplayID, "replay-1")
	response.Header.Set(headers.VSRSelectedModel, "model-a")

	trackPlaygroundReplayResponse(response, store)
	if store.bindSessionID != "session-1" || store.bindReplayID != "replay-1" || store.bindTargetRef != "model-a" {
		t.Fatalf("binding = session:%q replay:%q target:%q", store.bindSessionID, store.bindReplayID, store.bindTargetRef)
	}
	if store.completeCalled {
		t.Fatal("replay completed before response body was delivered")
	}
	if _, err := io.ReadAll(response.Body); err != nil {
		t.Fatal(err)
	}
	if !store.completeCalled {
		t.Fatal("replay was not completed after response body EOF")
	}
}

func TestPlaygroundOutcomeProxyForcesRecordOnlyForReadRole(t *testing.T) {
	var posted playgroundOutcomeRequest
	var upstreamAuthorizations []string
	var upstreamIdempotencyKeys []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamAuthorizations = append(upstreamAuthorizations, r.Header.Get("Authorization"))
		upstreamIdempotencyKeys = append(upstreamIdempotencyKeys, r.Header.Get("Idempotency-Key"))
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodGet:
			_, _ = w.Write([]byte(`{"id":"replay-1","selected_model":"model-a","lifecycle_state":"completed"}`))
		case http.MethodPost:
			if err := json.NewDecoder(r.Body).Decode(&posted); err != nil {
				t.Fatalf("decode upstream outcome: %v", err)
			}
			_, _ = w.Write([]byte(`{"recorded":true}`))
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{}
	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		store,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/router/api/v1/observability/outcomes",
		strings.NewReader(`{"replay_id":"replay-1","source":"user","target":"model","target_ref":"model-a","verdict":"good_fit"}`),
	)
	request.Header.Set("Authorization", "Bearer browser-token")
	request.Header.Set("Idempotency-Key", "browser-controlled-key")
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{
		SessionID: "session-1",
		Role:      auth.RoleRead,
	}))
	response := httptest.NewRecorder()

	mux.ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if !posted.RecordOnly {
		t.Fatal("read-role outcome was not forced into record-only mode")
	}
	if !store.claimCalled || !store.finishCalled || !store.finishSuccess {
		t.Fatalf("store lifecycle: claimed=%v finished=%v success=%v", store.claimCalled, store.finishCalled, store.finishSuccess)
	}
	for _, authorization := range upstreamAuthorizations {
		if authorization != "Bearer router-service-token" {
			t.Fatalf("upstream Authorization = %q", authorization)
		}
	}
	if got := upstreamIdempotencyKeys[len(upstreamIdempotencyKeys)-1]; got != "server-owned-feedback-key" {
		t.Fatalf("upstream Idempotency-Key = %q", got)
	}
}

func TestPlaygroundOutcomeProxyPreservesWriterLearningBehavior(t *testing.T) {
	var posted playgroundOutcomeRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`{"id":"replay-1","selected_model":"model-a","lifecycle_state":"completed"}`))
			return
		}
		_ = json.NewDecoder(r.Body).Decode(&posted)
		_, _ = w.Write([]byte(`{"recorded":true}`))
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{}
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: server.URL}, nil, store, routerProxyCredentialProvider{token: "router-token"})
	request := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes", strings.NewReader(
		`{"replay_id":"replay-1","target":"model","target_ref":"model-a","verdict":"good_fit"}`,
	))
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{SessionID: "session-1", Role: auth.RoleWrite}))
	response := httptest.NewRecorder()

	mux.ServeHTTP(response, request)

	if response.Code != http.StatusOK || posted.RecordOnly {
		t.Fatalf("status=%d record_only=%v body=%s", response.Code, posted.RecordOnly, response.Body.String())
	}
}

func TestPlaygroundOutcomeProxyRejectsForeignSessionBeforeRouterLookup(t *testing.T) {
	var routerCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		routerCalls++
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{validateErr: auth.ErrPlaygroundReplayNotOwned}
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: server.URL}, nil, store, routerProxyCredentialProvider{token: "router-token"})
	request := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes", strings.NewReader(
		`{"replay_id":"replay-1","target":"model","target_ref":"model-a","verdict":"good_fit"}`,
	))
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{SessionID: "foreign-session", Role: auth.RoleRead}))
	response := httptest.NewRecorder()

	mux.ServeHTTP(response, request)

	if response.Code != http.StatusForbidden || routerCalls != 0 || store.claimCalled {
		t.Fatalf("status=%d routerCalls=%d claimed=%v body=%s", response.Code, routerCalls, store.claimCalled, response.Body.String())
	}
}

func TestPlaygroundOutcomeProxySurfacesRouterFailureAndReleasesClaim(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			_, _ = w.Write([]byte(`{"id":"replay-1","selected_model":"model-a","lifecycle_state":"completed"}`))
			return
		}
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"error":{"message":"learning runtime unavailable"}}`))
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{}
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: server.URL}, nil, store, routerProxyCredentialProvider{token: "router-token"})
	request := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes", strings.NewReader(
		`{"replay_id":"replay-1","target":"model","target_ref":"model-a","verdict":"good_fit"}`,
	))
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{SessionID: "session-1", Role: auth.RoleRead}))
	response := httptest.NewRecorder()

	mux.ServeHTTP(response, request)

	if response.Code != http.StatusBadGateway || !strings.Contains(response.Body.String(), "learning runtime unavailable") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if !store.finishCalled || store.finishSuccess {
		t.Fatalf("finish called=%v success=%v", store.finishCalled, store.finishSuccess)
	}
}

func TestPlaygroundOutcomeProxyReusesIdempotencyKeyAfterLostResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"replay-1","selected_model":"model-a","lifecycle_state":"completed"}`))
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{claimKey: "persisted-feedback-key"}
	mux := http.NewServeMux()
	proxy := registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		store,
		routerProxyCredentialProvider{token: "router-token"},
	)
	var forwardedKeys []string
	appliedKeys := map[string]bool{}
	proxy.Transport = roundTripFunc(func(request *http.Request) (*http.Response, error) {
		key := request.Header.Get("Idempotency-Key")
		forwardedKeys = append(forwardedKeys, key)
		if !appliedKeys[key] {
			appliedKeys[key] = true
			return nil, io.ErrUnexpectedEOF
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body:       io.NopCloser(strings.NewReader(`{"success":true,"duplicate":true}`)),
			Request:    request,
		}, nil
	})

	submit := func(browserKey string) *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes", strings.NewReader(
			`{"replay_id":"replay-1","target":"model","target_ref":"model-a","verdict":"good_fit"}`,
		))
		request.Header.Set("Idempotency-Key", browserKey)
		request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{
			SessionID: "session-1",
			Role:      auth.RoleWrite,
		}))
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, request)
		return response
	}

	first := submit("browser-key-1")
	if first.Code != http.StatusBadGateway {
		t.Fatalf("first status = %d body=%s", first.Code, first.Body.String())
	}
	second := submit("browser-key-2")
	if second.Code != http.StatusOK {
		t.Fatalf("second status = %d body=%s", second.Code, second.Body.String())
	}
	if len(forwardedKeys) != 2 || forwardedKeys[0] != "persisted-feedback-key" || forwardedKeys[1] != forwardedKeys[0] {
		t.Fatalf("forwarded idempotency keys = %#v", forwardedKeys)
	}
	if len(appliedKeys) != 1 {
		t.Fatalf("applied idempotency keys = %#v, want one", appliedKeys)
	}
	if len(store.finishResults) != 2 || store.finishResults[0] || !store.finishResults[1] {
		t.Fatalf("finish results = %#v, want [false true]", store.finishResults)
	}
}

func TestPlaygroundOutcomeProxyRejectsInProgressRouterReplay(t *testing.T) {
	var outcomePosts int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			outcomePosts++
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"replay-1","selected_model":"model-a","lifecycle_state":"in_progress"}`))
	}))
	defer server.Close()

	store := &fakePlaygroundFeedbackStore{}
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: server.URL}, nil, store, routerProxyCredentialProvider{token: "router-token"})
	request := httptest.NewRequest(http.MethodPost, "/api/router/api/v1/observability/outcomes", strings.NewReader(
		`{"replay_id":"replay-1","target":"model","target_ref":"model-a","verdict":"good_fit"}`,
	))
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{SessionID: "session-1", Role: auth.RoleRead}))
	response := httptest.NewRecorder()

	mux.ServeHTTP(response, request)

	if response.Code != http.StatusConflict || outcomePosts != 0 || store.claimCalled {
		t.Fatalf("status=%d posts=%d claimed=%v body=%s", response.Code, outcomePosts, store.claimCalled, response.Body.String())
	}
}

func TestMapPlaygroundStoreError(t *testing.T) {
	if got := mapPlaygroundStoreError(errors.New("database unavailable")); got.status != http.StatusInternalServerError {
		t.Fatalf("status = %d", got.status)
	}
}
