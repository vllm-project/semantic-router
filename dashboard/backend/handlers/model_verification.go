package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

const (
	modelVerificationPath            = "/api/models/verify"
	maxModelVerificationRequestBytes = 4 << 10
	maxModelVerificationNameBytes    = 256
	modelVerificationAuditTimeout    = time.Second
)

// ModelVerificationRequest identifies one configured provider model. The
// server resolves every transport detail from the active runtime config so a
// caller cannot turn this endpoint into an arbitrary outbound-request proxy.
type ModelVerificationRequest struct {
	Model string `json:"model"`
}

// ModelVerificationResponse proves that the selected physical model returned
// generated output. It intentionally omits endpoint URLs and credentials.
type ModelVerificationResponse struct {
	Verified      bool   `json:"verified"`
	Model         string `json:"model"`
	ProviderModel string `json:"providerModel"`
	Backend       string `json:"backend"`
	Provider      string `json:"provider"`
	Summary       string `json:"summary"`
	VerifiedAt    string `json:"verifiedAt"`
	LatencyMS     int64  `json:"latencyMs"`
}

// ModelVerificationHandler executes a bounded, non-streaming inference call
// directly against the configured backend for a provider model. Verification
// is operationally read-only: it never mutates config or Router state.
func ModelVerificationHandler(configPath string, auditors ...ModelVerificationAuditor) http.HandlerFunc {
	options := modelVerificationOptions{}
	if len(auditors) > 0 {
		options.auditor = auditors[0]
	}
	return newModelVerificationHandler(configPath, options)
}

// ModelVerificationAuditor is the narrow durable-audit seam used by the
// inference verifier. The auth service implements it without exposing its DB.
type ModelVerificationAuditor interface {
	AddAuditLog(context.Context, dashboardauth.AuditLog) error
}

func newModelVerificationHandler(configPath string, options modelVerificationOptions) http.HandlerFunc {
	service := newModelVerificationService(configPath, options)
	rateLimiter := options.rateLimiter
	if rateLimiter == nil {
		rateLimiter = newModelVerificationRateLimiter(options.now)
	}
	return func(w http.ResponseWriter, r *http.Request) {
		status := http.StatusInternalServerError
		model := ""
		errorCode := "verification_failed"
		defer func() {
			auditModelVerificationAttempt(r, options.auditor, model, status, errorCode)
		}()
		w.Header().Set("Cache-Control", "no-store")
		if r.URL.Path != modelVerificationPath {
			status = http.StatusNotFound
			errorCode = "not_found"
			http.NotFound(w, r)
			return
		}
		if r.Method != http.MethodPost {
			status = http.StatusMethodNotAllowed
			errorCode = "method_not_allowed"
			w.Header().Set("Allow", http.MethodPost)
			writeModelVerificationError(w, http.StatusMethodNotAllowed, "method_not_allowed", "Only POST is supported.")
			return
		}

		var request ModelVerificationRequest
		if err := decodeModelVerificationRequest(w, r, &request); err != nil {
			status = http.StatusBadRequest
			errorCode = "invalid_request"
			writeModelVerificationError(w, http.StatusBadRequest, "invalid_request", err.Error())
			return
		}
		model = request.Model
		identity := modelVerificationRateLimitIdentity(r, request.Model)
		allowed, retryAfter := rateLimiter.Allow(identity)
		if !allowed {
			status = http.StatusTooManyRequests
			errorCode = "rate_limited"
			retryAfterSeconds := int((retryAfter + time.Second - 1) / time.Second)
			w.Header().Set("Retry-After", strconv.Itoa(retryAfterSeconds))
			writeModelVerificationError(w, status, errorCode, "Too many live model verification requests. Try again shortly.")
			return
		}

		response, err := service.Verify(r.Context(), strings.TrimSpace(request.Model))
		if err != nil {
			var verificationErr *modelVerificationError
			if !errors.As(err, &verificationErr) {
				verificationErr = &modelVerificationError{
					status:  http.StatusInternalServerError,
					code:    "verification_failed",
					message: "Model verification could not be completed.",
				}
			}
			status = verificationErr.status
			errorCode = verificationErr.code
			writeModelVerificationError(w, verificationErr.status, verificationErr.code, verificationErr.message)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		status = http.StatusOK
		errorCode = "verified"
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(response)
	}
}

func modelVerificationUserIdentity(r *http.Request) string {
	if authContext, ok := dashboardauth.AuthFromContext(r); ok && strings.TrimSpace(authContext.UserID) != "" {
		return strings.TrimSpace(authContext.UserID)
	}
	return "unauthenticated"
}

func modelVerificationRateLimitIdentity(r *http.Request, model string) string {
	return modelVerificationUserIdentity(r) + "\x00" + strings.TrimSpace(model)
}

func auditModelVerificationAttempt(
	r *http.Request,
	auditor ModelVerificationAuditor,
	model string,
	status int,
	result string,
) {
	if auditor == nil {
		return
	}
	actor := ""
	if authContext, ok := dashboardauth.AuthFromContext(r); ok {
		actor = authContext.UserID
	}
	extra, _ := json.Marshal(map[string]string{
		"model":  model,
		"result": result,
	})
	auditContext, cancel := context.WithTimeout(context.WithoutCancel(r.Context()), modelVerificationAuditTimeout)
	defer cancel()
	_ = auditor.AddAuditLog(auditContext, dashboardauth.AuditLog{
		UserID:     actor,
		Action:     "model.inference_verify",
		Resource:   "models",
		Method:     r.Method,
		Path:       r.URL.Path,
		IP:         r.RemoteAddr,
		UserAgent:  r.UserAgent(),
		StatusCode: status,
		CreatedAt:  time.Now().Unix(),
		ExtraJSON:  string(extra),
	})
}

func decodeModelVerificationRequest(w http.ResponseWriter, r *http.Request, target *ModelVerificationRequest) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxModelVerificationRequestBytes)
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return fmt.Errorf("invalid request body: %s", modelVerificationDecodeError(err))
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return errors.New("invalid request body: expected one JSON object")
	}

	target.Model = strings.TrimSpace(target.Model)
	if target.Model == "" {
		return errors.New("model is required")
	}
	if len(target.Model) > maxModelVerificationNameBytes {
		return errors.New("model name is too large")
	}
	for _, character := range target.Model {
		if character < 0x20 || character == 0x7f {
			return errors.New("model name contains control characters")
		}
	}
	return nil
}

func modelVerificationDecodeError(err error) string {
	var maxBytesErr *http.MaxBytesError
	if errors.As(err, &maxBytesErr) {
		return "request exceeds the size limit"
	}
	if errors.Is(err, io.EOF) {
		return "request body is empty"
	}
	return "malformed JSON"
}

func writeModelVerificationError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   code,
		"message": message,
	})
}
