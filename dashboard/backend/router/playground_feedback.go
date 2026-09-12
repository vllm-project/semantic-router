package router

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const (
	playgroundFeedbackBodyLimit  = 1 << 20
	playgroundFeedbackRateLimit  = 60
	playgroundFeedbackRateWindow = time.Minute
)

type playgroundFeedbackStore interface {
	BindPlaygroundReplay(context.Context, string, string, string) error
	CompletePlaygroundReplay(context.Context, string, string) error
	ValidatePlaygroundReplay(context.Context, string, string, string) error
	ClaimPlaygroundReplay(context.Context, string, string, string, int, time.Duration) (string, error)
	FinishPlaygroundReplay(context.Context, string, string, bool) error
}

type playgroundOutcomeRequest struct {
	ReplayID   string            `json:"replay_id"`
	Source     string            `json:"source,omitempty"`
	Target     string            `json:"target"`
	TargetRef  string            `json:"target_ref,omitempty"`
	Verdict    string            `json:"verdict"`
	Reason     string            `json:"reason,omitempty"`
	Score      *float64          `json:"score,omitempty"`
	Metadata   map[string]string `json:"metadata,omitempty"`
	RecordOnly bool              `json:"record_only,omitempty"`
}

type playgroundReplayRecord struct {
	ID             string `json:"id"`
	SelectedModel  string `json:"selected_model"`
	LifecycleState string `json:"lifecycle_state"`
}

type playgroundFeedbackError struct {
	status  int
	code    string
	message string
}

func attachPlaygroundReplayTracking(proxy *httputil.ReverseProxy, store playgroundFeedbackStore) {
	if proxy == nil || store == nil {
		return
	}
	originalModifyResponse := proxy.ModifyResponse
	proxy.ModifyResponse = func(response *http.Response) error {
		if originalModifyResponse != nil {
			if err := originalModifyResponse(response); err != nil {
				return err
			}
		}
		trackPlaygroundReplayResponse(response, store)
		return nil
	}
}

func trackPlaygroundReplayResponse(response *http.Response, store playgroundFeedbackStore) {
	if response == nil || response.Request == nil || response.StatusCode < 200 || response.StatusCode >= 300 {
		return
	}
	if response.Request.URL.Path != "/v1/chat/completions" {
		return
	}
	principal, ok := auth.AuthFromContext(response.Request)
	if !ok || strings.TrimSpace(principal.SessionID) == "" {
		return
	}
	replayID := strings.TrimSpace(response.Header.Get(headers.RouterReplayID))
	model := strings.TrimSpace(response.Header.Get(headers.VSRSelectedModel))
	if replayID == "" || model == "" {
		return
	}
	if err := store.BindPlaygroundReplay(response.Request.Context(), principal.SessionID, replayID, model); err != nil {
		log.Printf("playground replay binding failed: %v", err)
		return
	}
	if response.Body == nil {
		completePlaygroundReplay(store, principal.SessionID, replayID)
		return
	}
	response.Body = &playgroundCompletionBody{
		ReadCloser: response.Body,
		onComplete: func() { completePlaygroundReplay(store, principal.SessionID, replayID) },
	}
}

type playgroundCompletionBody struct {
	io.ReadCloser
	onComplete func()
	once       sync.Once
}

func (b *playgroundCompletionBody) Read(buffer []byte) (int, error) {
	n, err := b.ReadCloser.Read(buffer)
	if errors.Is(err, io.EOF) {
		b.once.Do(b.onComplete)
	}
	return n, err
}

func completePlaygroundReplay(store playgroundFeedbackStore, sessionID, replayID string) {
	if err := store.CompletePlaygroundReplay(context.Background(), sessionID, replayID); err != nil {
		log.Printf("playground replay completion failed: %v", err)
	}
}

func servePlaygroundOutcome(
	w http.ResponseWriter,
	r *http.Request,
	routerAPIURL string,
	proxy *httputil.ReverseProxy,
	store playgroundFeedbackStore,
	credentialProvider routerauth.CredentialProvider,
) {
	principal, ok := auth.AuthFromContext(r)
	if !ok || strings.TrimSpace(principal.SessionID) == "" {
		writePlaygroundFeedbackError(w, playgroundFeedbackError{
			status:  http.StatusUnauthorized,
			code:    "PLAYGROUND_SESSION_REQUIRED",
			message: "A current Dashboard login session is required to submit feedback.",
		})
		return
	}

	payload, rawBody, parseErr := readPlaygroundOutcomeRequest(r)
	if parseErr != nil {
		writePlaygroundFeedbackError(w, *parseErr)
		return
	}
	if err := store.ValidatePlaygroundReplay(r.Context(), principal.SessionID, payload.ReplayID, payload.TargetRef); err != nil {
		writePlaygroundFeedbackError(w, mapPlaygroundStoreError(err))
		return
	}
	if err := validateRouterReplayForFeedback(r.Context(), routerAPIURL, credentialProvider, payload); err != nil {
		writePlaygroundFeedbackError(w, *err)
		return
	}
	idempotencyKey, claimErr := store.ClaimPlaygroundReplay(
		r.Context(),
		principal.SessionID,
		payload.ReplayID,
		payload.TargetRef,
		playgroundFeedbackRateLimit,
		playgroundFeedbackRateWindow,
	)
	if claimErr != nil {
		writePlaygroundFeedbackError(w, mapPlaygroundStoreError(claimErr))
		return
	}

	if principal.Role == auth.RoleRead {
		payload.RecordOnly = true
		rawBody, _ = json.Marshal(payload)
	}
	r.Body = io.NopCloser(bytes.NewReader(rawBody))
	r.ContentLength = int64(len(rawBody))
	r.Header.Set("Idempotency-Key", idempotencyKey)
	if err := routerauth.RewriteAuthorization(r, credentialProvider); err != nil {
		_ = store.FinishPlaygroundReplay(context.Background(), principal.SessionID, payload.ReplayID, false)
		writePlaygroundFeedbackError(w, playgroundFeedbackError{
			status:  http.StatusServiceUnavailable,
			code:    "ROUTER_CREDENTIAL_UNAVAILABLE",
			message: "Router management credential is unavailable.",
		})
		return
	}

	statusWriter := &playgroundOutcomeResponseWriter{ResponseWriter: w, status: http.StatusOK}
	proxy.ServeHTTP(statusWriter, r)
	submitted := statusWriter.status >= http.StatusOK && statusWriter.status < http.StatusMultipleChoices
	if err := store.FinishPlaygroundReplay(context.Background(), principal.SessionID, payload.ReplayID, submitted); err != nil {
		log.Printf("playground replay submission finalization failed: %v", err)
	}
}

func readPlaygroundOutcomeRequest(r *http.Request) (playgroundOutcomeRequest, []byte, *playgroundFeedbackError) {
	var payload playgroundOutcomeRequest
	body, err := io.ReadAll(io.LimitReader(r.Body, playgroundFeedbackBodyLimit+1))
	if err != nil {
		return payload, nil, &playgroundFeedbackError{
			status: http.StatusBadRequest, code: "INVALID_FEEDBACK", message: "Failed to read feedback request.",
		}
	}
	if len(body) > playgroundFeedbackBodyLimit {
		return payload, nil, &playgroundFeedbackError{
			status: http.StatusRequestEntityTooLarge, code: "FEEDBACK_TOO_LARGE", message: "Feedback request is too large.",
		}
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return payload, nil, &playgroundFeedbackError{
			status: http.StatusBadRequest, code: "INVALID_FEEDBACK", message: "Feedback request must be valid JSON.",
		}
	}
	payload.ReplayID = strings.TrimSpace(payload.ReplayID)
	payload.Target = strings.TrimSpace(payload.Target)
	payload.TargetRef = strings.TrimSpace(payload.TargetRef)
	payload.Verdict = strings.TrimSpace(payload.Verdict)
	if payload.ReplayID == "" || payload.TargetRef == "" {
		return payload, nil, &playgroundFeedbackError{
			status: http.StatusBadRequest, code: "INVALID_FEEDBACK", message: "replay_id and target_ref are required.",
		}
	}
	if payload.Target != "model" || (payload.Verdict != "good_fit" && payload.Verdict != "underpowered") {
		return payload, nil, &playgroundFeedbackError{
			status:  http.StatusBadRequest,
			code:    "INVALID_FEEDBACK",
			message: "Playground feedback must rate a model as good_fit or underpowered.",
		}
	}
	return payload, body, nil
}

func validateRouterReplayForFeedback(
	ctx context.Context,
	routerAPIURL string,
	credentialProvider routerauth.CredentialProvider,
	payload playgroundOutcomeRequest,
) *playgroundFeedbackError {
	base, err := url.Parse(strings.TrimRight(routerAPIURL, "/"))
	if err != nil || base.Scheme == "" || base.Host == "" {
		return &playgroundFeedbackError{
			status: http.StatusServiceUnavailable, code: "ROUTER_UNAVAILABLE", message: "Router Replay is unavailable.",
		}
	}
	base.Path = strings.TrimRight(base.Path, "/") + "/api/v1/observability/replays/" + url.PathEscape(payload.ReplayID)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, base.String(), nil)
	if err != nil {
		return &playgroundFeedbackError{
			status: http.StatusInternalServerError, code: "REPLAY_VERIFICATION_FAILED", message: "Could not verify the replay record.",
		}
	}
	authorizationErr := routerauth.RewriteAuthorization(request, credentialProvider)
	if authorizationErr != nil {
		return &playgroundFeedbackError{
			status:  http.StatusServiceUnavailable,
			code:    "ROUTER_CREDENTIAL_UNAVAILABLE",
			message: "Router management credential is unavailable.",
		}
	}
	response, err := (&http.Client{Timeout: 5 * time.Second}).Do(request)
	if err != nil {
		return &playgroundFeedbackError{
			status: http.StatusBadGateway, code: "REPLAY_VERIFICATION_FAILED", message: "Could not verify the replay record.",
		}
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return &playgroundFeedbackError{
			status:  http.StatusConflict,
			code:    "REPLAY_NOT_READY",
			message: "The matching replay record is not available for feedback.",
		}
	}
	var record playgroundReplayRecord
	if err := json.NewDecoder(io.LimitReader(response.Body, playgroundFeedbackBodyLimit)).Decode(&record); err != nil {
		return &playgroundFeedbackError{
			status: http.StatusBadGateway, code: "REPLAY_VERIFICATION_FAILED", message: "Could not verify the replay record.",
		}
	}
	if strings.TrimSpace(record.ID) != payload.ReplayID {
		return &playgroundFeedbackError{
			status:  http.StatusConflict,
			code:    "REPLAY_NOT_READY",
			message: "The matching replay record is not available for feedback.",
		}
	}
	if strings.TrimSpace(record.LifecycleState) != "completed" {
		return &playgroundFeedbackError{
			status: http.StatusConflict, code: "REPLAY_IN_PROGRESS", message: "The replay is not complete yet.",
		}
	}
	if strings.TrimSpace(record.SelectedModel) != payload.TargetRef {
		return &playgroundFeedbackError{
			status:  http.StatusConflict,
			code:    "FEEDBACK_MODEL_MISMATCH",
			message: "Feedback model does not match the routed response.",
		}
	}
	return nil
}

func mapPlaygroundStoreError(err error) playgroundFeedbackError {
	switch {
	case errors.Is(err, auth.ErrPlaygroundReplayExpired):
		return playgroundFeedbackError{status: http.StatusGone, code: "PLAYGROUND_REPLAY_EXPIRED", message: err.Error()}
	case errors.Is(err, auth.ErrPlaygroundReplayInProgress):
		return playgroundFeedbackError{status: http.StatusConflict, code: "REPLAY_IN_PROGRESS", message: err.Error()}
	case errors.Is(err, auth.ErrPlaygroundReplayModelMismatch):
		return playgroundFeedbackError{status: http.StatusConflict, code: "FEEDBACK_MODEL_MISMATCH", message: err.Error()}
	case errors.Is(err, auth.ErrPlaygroundReplayDuplicate):
		return playgroundFeedbackError{status: http.StatusConflict, code: "DUPLICATE_FEEDBACK", message: err.Error()}
	case errors.Is(err, auth.ErrPlaygroundFeedbackRateLimited):
		return playgroundFeedbackError{status: http.StatusTooManyRequests, code: "FEEDBACK_RATE_LIMITED", message: err.Error()}
	case errors.Is(err, auth.ErrPlaygroundReplayNotOwned):
		return playgroundFeedbackError{status: http.StatusForbidden, code: "PLAYGROUND_REPLAY_NOT_OWNED", message: err.Error()}
	default:
		return playgroundFeedbackError{
			status: http.StatusInternalServerError, code: "FEEDBACK_STORE_ERROR", message: "Could not validate feedback ownership.",
		}
	}
}

func writePlaygroundFeedbackError(w http.ResponseWriter, feedbackErr playgroundFeedbackError) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(feedbackErr.status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{
			"code": feedbackErr.code, "message": feedbackErr.message, "type": "invalid_request_error",
		},
	})
}

type playgroundOutcomeResponseWriter struct {
	http.ResponseWriter
	status int
}

func (w *playgroundOutcomeResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *playgroundOutcomeResponseWriter) WriteHeader(status int) {
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *playgroundOutcomeResponseWriter) Write(body []byte) (int, error) {
	return w.ResponseWriter.Write(body)
}
