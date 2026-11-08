package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strconv"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/training"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

const TrainingJSONMaxBytes int64 = 1 << 20

func trainingOwner(r *http.Request) string { ac, _ := auth.AuthFromContext(r); return ac.UserID }

// TrainingCreate decodes the shared contract without accepting client ownership
// or filesystem fields. Authorization and audit policies are registered in router.
func TrainingCreate[T, R any](status int, call func(context.Context, string, T) (R, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, TrainingJSONMaxBytes)
		decoder := json.NewDecoder(r.Body)
		decoder.DisallowUnknownFields()
		var req T
		if err := decoder.Decode(&req); err != nil {
			trainingResponse(w, 400, nil, fmt.Errorf("%w: %w", training.ErrInvalid, err))
			return
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			trainingResponse(w, 400, nil, fmt.Errorf("%w: expected one JSON request", training.ErrInvalid))
			return
		}
		value, err := call(r.Context(), trainingOwner(r), req)
		trainingResponse(w, status, value, err)
	}
}

type TrainingHandler struct{ service *training.Service }

func NewTrainingHandler(s *training.Service) *TrainingHandler { return &TrainingHandler{service: s} }

func (h *TrainingHandler) Upload(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("Content-Type") != "application/octet-stream" {
		trainingResponse(w, 400, nil, fmt.Errorf("%w: expected application/octet-stream", training.ErrInvalid))
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, training.MaxFileBytes)
	value, err := h.service.Upload(r.Context(), trainingOwner(r), r.Body)
	trainingResponse(w, 201, value, err)
}

func (h *TrainingHandler) Get(kind string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		value, err := h.service.Get(r.Context(), trainingOwner(r), kind, r.PathValue("id"))
		trainingResponse(w, 200, value, err)
	}
}

func (h *TrainingHandler) ListRuns(w http.ResponseWriter, r *http.Request) {
	value, err := h.service.ListRuns(r.Context(), trainingOwner(r), r.URL.Query().Get("experiment_id"))
	trainingResponse(w, 200, value, err)
}

func (h *TrainingHandler) Events(w http.ResponseWriter, r *http.Request) {
	var after int64
	if raw := r.URL.Query().Get("after"); raw != "" {
		var err error
		after, err = strconv.ParseInt(raw, 10, 64)
		if err != nil || after < 0 {
			trainingResponse(w, 400, nil, fmt.Errorf("%w: after must be a nonnegative integer", training.ErrInvalid))
			return
		}
	}
	value, err := h.service.Events(r.Context(), trainingOwner(r), r.PathValue("id"), after)
	trainingResponse(w, 200, value, err)
}

func (h *TrainingHandler) Action(call func(context.Context, string, string) (c.RunGraph, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		value, err := call(r.Context(), trainingOwner(r), r.PathValue("id"))
		trainingResponse(w, 202, value, err)
	}
}

func (h *TrainingHandler) Variants(w http.ResponseWriter, r *http.Request) {
	value, err := h.service.Variants(r.Context(), trainingOwner(r), r.PathValue("id"))
	trainingResponse(w, 200, value, err)
}

func (h *TrainingHandler) Download(w http.ResponseWriter, r *http.Request) {
	file, err := h.service.Download(r.Context(), trainingOwner(r), r.PathValue("id"))
	if err != nil {
		trainingResponse(w, 200, nil, err)
		return
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		trainingResponse(w, 200, nil, err)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", `attachment; filename="artifact"`)
	http.ServeContent(w, r, "artifact", info.ModTime(), file)
}

func trainingResponse(w http.ResponseWriter, status int, value any, err error) {
	if err != nil {
		status = http.StatusInternalServerError
		code, message := "internal_error", "training operation failed"
		switch {
		case errors.Is(err, training.ErrInvalid):
			status, code, message = 400, "invalid_request", err.Error()
		case errors.Is(err, training.ErrNotFound):
			status, code, message = 404, "not_found", err.Error()
		case errors.Is(err, training.ErrConflict):
			status, code, message = 409, "conflict", err.Error()
		case errors.Is(err, training.ErrTooLarge):
			status, code, message = 413, "payload_too_large", err.Error()
		case errors.Is(err, auth.ErrPermissionDenied):
			status, code, message = 403, "forbidden", "permission denied"
		}
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			status, code, message = 413, "payload_too_large", "request body too large"
		}
		if status == 500 {
			log.Printf("training API: %v", err)
		}
		value = c.APIError{Code: code, Message: message}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Printf("training response: %v", err)
	}
}
