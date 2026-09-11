package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/evaluationplane"
)

const evaluationAPIBase = "/api/evaluation/v1"

type EvaluationPlaneHandler struct {
	service            *evaluationplane.Service
	readonly           bool
	responseStreams    *evaluationResponseStreamLimiter
	streamWriteTimeout time.Duration
}

func NewEvaluationPlaneHandler(service *evaluationplane.Service, readonly bool) *EvaluationPlaneHandler {
	return &EvaluationPlaneHandler{
		service: service, readonly: readonly,
		responseStreams:    newEvaluationResponseStreamLimiter(),
		streamWriteTimeout: evaluationResponseWriteTimeout,
	}
}

func (h *EvaluationPlaneHandler) Catalog(w http.ResponseWriter, r *http.Request) {
	if preflightOrMethod(w, r, http.MethodGet) {
		return
	}
	catalog, err := h.service.Catalog()
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, catalog)
}

func (h *EvaluationPlaneHandler) Runs(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	switch r.Method {
	case http.MethodGet:
		actor, ok := h.evaluationActor(w, r)
		if !ok {
			return
		}
		query, err := evaluationRunListQuery(r)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		ledger, err := h.service.ListRunLedgerPageAs(actor, query)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, ledger)
	case http.MethodPost:
		actor, ok := h.evaluationActor(w, r)
		if !ok {
			return
		}
		if h.denyReadonly(w) {
			return
		}
		var wire evaluationCreateRunWireRequest
		if err := decodeStrictJSON(r, &wire); err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		request, err := wire.domainRequest()
		if err != nil {
			writeEvaluationError(w, fmt.Errorf("%w: %w", evaluationplane.ErrInvalid, err))
			return
		}
		run, err := h.service.CreateRunAs(r.Context(), actor, request)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusCreated, run)
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodPost)
	}
}

func evaluationRunListQuery(r *http.Request) (evaluationplane.RunListQuery, error) {
	if !onlyQueryKeys(r, "limit", "cursor") {
		return evaluationplane.RunListQuery{}, fmt.Errorf("%w: unsupported run list query field", evaluationplane.ErrInvalid)
	}
	values := r.URL.Query()
	if len(values["limit"]) > 1 || len(values["cursor"]) > 1 {
		return evaluationplane.RunListQuery{}, fmt.Errorf("%w: run list query fields cannot be repeated", evaluationplane.ErrInvalid)
	}
	query := evaluationplane.RunListQuery{Cursor: strings.TrimSpace(values.Get("cursor"))}
	if raw := strings.TrimSpace(values.Get("limit")); raw != "" {
		limit, err := strconv.Atoi(raw)
		if err != nil {
			return evaluationplane.RunListQuery{}, fmt.Errorf("%w: run list limit must be an integer", evaluationplane.ErrInvalid)
		}
		query.Limit = limit
	}
	return query, nil
}

func (h *EvaluationPlaneHandler) RunRoute(w http.ResponseWriter, r *http.Request) {
	if evaluationCORS(w, r) {
		return
	}
	rest := strings.TrimPrefix(r.URL.Path, evaluationAPIBase+"/runs/")
	parts := strings.Split(strings.Trim(rest, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	runID := parts[0]
	if len(parts) == 1 {
		h.runResource(w, r, runID)
		return
	}
	if len(parts) == 2 {
		switch parts[1] {
		case "start":
			h.runAction(w, r, runID, "start")
		case "cancel":
			h.runAction(w, r, runID, "cancel")
		case "report":
			h.report(w, r, runID)
		case "events":
			h.events(w, r, runID)
		case "lifecycle":
			h.runLifecycle(w, r, runID)
		default:
			http.NotFound(w, r)
		}
		return
	}
	if len(parts) == 3 && parts[1] == "artifacts" {
		h.artifact(w, r, runID, parts[2])
		return
	}
	http.NotFound(w, r)
}

func (h *EvaluationPlaneHandler) Compare(w http.ResponseWriter, r *http.Request) {
	if preflightOrMethod(w, r, http.MethodGet) {
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if !onlyQueryKeys(r, "baseline_run_id", "candidate_run_id") {
		writeEvaluationError(w, fmt.Errorf("%w: unsupported comparison query field", evaluationplane.ErrInvalid))
		return
	}
	baselineID := strings.TrimSpace(r.URL.Query().Get("baseline_run_id"))
	candidateID := strings.TrimSpace(r.URL.Query().Get("candidate_run_id"))
	if baselineID == "" || candidateID == "" {
		writeEvaluationError(w, fmt.Errorf("%w: baseline_run_id and candidate_run_id are required", evaluationplane.ErrInvalid))
		return
	}
	comparison, err := h.service.CompareAs(actor, baselineID, candidateID)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, comparison)
}

func (h *EvaluationPlaneHandler) runResource(w http.ResponseWriter, r *http.Request, runID string) {
	switch r.Method {
	case http.MethodGet:
		actor, ok := h.evaluationActor(w, r)
		if !ok {
			return
		}
		run, err := h.service.GetRunAs(actor, runID)
		if err != nil {
			writeEvaluationError(w, err)
			return
		}
		writeEvaluationJSON(w, http.StatusOK, run)
	case http.MethodDelete:
		actor, ok := h.evaluationActor(w, r)
		if !ok {
			return
		}
		if h.denyReadonly(w) {
			return
		}
		if err := h.service.DeleteRunAs(actor, runID); err != nil {
			writeEvaluationError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w, http.MethodGet, http.MethodDelete)
	}
}

func (h *EvaluationPlaneHandler) runAction(w http.ResponseWriter, r *http.Request, runID, action string) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w, http.MethodPost)
		return
	}
	var (
		run evaluationplane.Run
		err error
	)
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	if h.denyReadonly(w) {
		return
	}
	if action == "start" {
		run, err = h.service.StartRunAs(r.Context(), actor, runID)
	} else {
		run, err = h.service.CancelRunAs(actor, runID)
	}
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	writeEvaluationJSON(w, http.StatusOK, run)
}

func (h *EvaluationPlaneHandler) report(w http.ResponseWriter, r *http.Request, runID string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	report, err := h.service.ReportJSONAs(actor, runID)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(report)
}

func (h *EvaluationPlaneHandler) artifact(w http.ResponseWriter, r *http.Request, runID, artifactID string) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w, http.MethodGet)
		return
	}
	actor, ok := h.evaluationActor(w, r)
	if !ok {
		return
	}
	releaseStream, streamErr := h.responseStreams.acquire(actor)
	if streamErr != nil {
		writeEvaluationError(w, streamErr)
		return
	}
	defer releaseStream()
	artifact, err := h.service.OpenArtifactAs(actor, runID, artifactID)
	if err != nil {
		writeEvaluationError(w, err)
		return
	}
	defer func() { _ = artifact.File.Close() }()
	w.Header().Set("Content-Type", artifact.MediaType)
	w.Header().Set("Content-Length", strconv.FormatInt(artifact.Size, 10))
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", artifact.Name))
	w.Header().Set("X-Content-Type-Options", "nosniff")
	deadlineWriter := newEvaluationDeadlineWriter(w, h.streamWriteTimeout)
	defer deadlineWriter.clearDeadline()
	if err := deadlineWriter.arm(); err != nil {
		return
	}
	http.ServeContent(deadlineWriter, r, artifact.Name, time.Time{}, artifact.File)
}

func (h *EvaluationPlaneHandler) denyReadonly(w http.ResponseWriter) bool {
	if !h.readonly {
		return false
	}
	writeEvaluationJSON(w, http.StatusForbidden, map[string]any{
		"error": map[string]string{"message": "Operation not allowed in readonly mode"},
	})
	return true
}

func writeEvaluationJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func writeEvaluationError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	switch {
	case errors.Is(err, evaluationplane.ErrInvalid):
		status = http.StatusBadRequest
	case errors.Is(err, evaluationplane.ErrNotFound):
		status = http.StatusNotFound
	case errors.Is(err, evaluationplane.ErrConflict):
		status = http.StatusConflict
	case errors.Is(err, evaluationplane.ErrForbidden):
		status = http.StatusForbidden
	case errors.Is(err, evaluationplane.ErrQuota):
		status = http.StatusInsufficientStorage
	}
	message := err.Error()
	if status == http.StatusInternalServerError {
		message = "Evaluation service failed"
	}
	writeEvaluationJSON(w, status, map[string]any{"error": map[string]string{"message": message}})
}

func preflightOrMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if evaluationCORS(w, r) {
		return true
	}
	if r.Method != method {
		methodNotAllowed(w, method)
		return true
	}
	return false
}

func evaluationCORS(w http.ResponseWriter, r *http.Request) bool {
	origin := strings.TrimSpace(r.Header.Get("Origin"))
	if origin == "" {
		if r.Method == http.MethodOptions {
			http.Error(w, "Cross-origin preflight requires a same-origin Origin header", http.StatusForbidden)
			return true
		}
		return false
	}
	parsed, err := url.Parse(origin)
	requestScheme := evaluationRequestScheme(r)
	if err != nil || requestScheme == "" || parsed.Scheme != requestScheme || parsed.Host != r.Host ||
		parsed.User != nil || parsed.Path != "" || parsed.RawQuery != "" || parsed.Fragment != "" {
		w.Header().Del("Access-Control-Allow-Origin")
		w.Header().Del("Access-Control-Allow-Credentials")
		http.Error(w, "Cross-origin Evaluation Plane access is forbidden", http.StatusForbidden)
		return true
	}
	w.Header().Set("Access-Control-Allow-Origin", origin)
	w.Header().Set("Access-Control-Allow-Credentials", "true")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, Last-Event-ID")
	w.Header().Set("Access-Control-Expose-Headers", "Content-Length, Content-Disposition")
	w.Header().Add("Vary", "Origin")
	w.Header().Add("Vary", "X-Forwarded-Proto")
	if r.Header.Get("Access-Control-Request-Private-Network") == "true" {
		w.Header().Set("Access-Control-Allow-Private-Network", "true")
	}
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusNoContent)
		return true
	}
	return false
}

func evaluationRequestScheme(r *http.Request) string {
	if r.TLS != nil {
		return "https"
	}
	forwarded := strings.TrimSpace(r.Header.Get("X-Forwarded-Proto"))
	if forwarded == "" {
		return "http"
	}
	if strings.Contains(forwarded, ",") || (forwarded != "http" && forwarded != "https") {
		return ""
	}
	return forwarded
}

func methodNotAllowed(w http.ResponseWriter, methods ...string) {
	w.Header().Set("Allow", strings.Join(methods, ", "))
	http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
}

func onlyQueryKeys(r *http.Request, allowed ...string) bool {
	allowlist := make(map[string]bool, len(allowed))
	for _, key := range allowed {
		allowlist[key] = true
	}
	for key := range r.URL.Query() {
		if !allowlist[key] {
			return false
		}
	}
	return true
}

func decodeEventID(id string) uint64 {
	value, _ := strconv.ParseUint(id, 10, 64)
	return value
}

func encodeSSE(event evaluationplane.Event) []byte {
	payload, _ := json.Marshal(event)
	var buffer bytes.Buffer
	fmt.Fprintf(&buffer, "id: %s\n", event.ID)
	fmt.Fprintf(&buffer, "event: %s\n", event.Type)
	fmt.Fprintf(&buffer, "data: %s\n\n", payload)
	return buffer.Bytes()
}
