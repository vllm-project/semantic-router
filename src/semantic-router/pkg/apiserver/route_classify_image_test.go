//go:build !windows && cgo

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

const malformedClassifyImagePayload = "request-private-payload!!!!"

func TestClassifyAndEvalImageWorkFailsFastBeforeService(t *testing.T) {
	imageURI := mustEmbeddingImageDataURI(t, "image/png")
	body, err := json.Marshal(map[string]interface{}{
		"messages": []map[string]interface{}{{
			"role": "user",
			"content": []map[string]interface{}{
				{"type": "text", "text": "describe the image"},
				{"type": "image_url", "image_url": map[string]string{"url": imageURI}},
			},
		}},
	})
	if err != nil {
		t.Fatalf("marshal image request: %v", err)
	}

	admission := newEmbeddingProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}
	defer release()

	tests := []struct {
		name    string
		path    string
		handler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)
	}{
		{name: "intent", path: "/api/v1/classify/intent", handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
			s.handleIntentClassification(w, r)
		}},
		{name: "eval", path: "/api/v1/eval", handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
			s.handleEvalClassification(w, r)
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := &evalCaptureClassificationService{}
			server := &ClassificationAPIServer{
				classificationSvc:  service,
				embeddingAdmission: admission,
			}
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(server, rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("expected 503, got %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "EMBEDDING_OVERLOADED")
			if rr.Header().Get("Retry-After") != "1" || rr.Header().Get("Cache-Control") != "no-store" {
				t.Fatalf("overload headers = %v", rr.Header())
			}
			if service.intentCalls != 0 || service.evalCalls != 0 {
				t.Fatalf("overloaded image request reached classification service: intent=%d eval=%d", service.intentCalls, service.evalCalls)
			}
		})
	}
}

func TestClassifyEvalAndCombinedTextUseNativeAdmission(t *testing.T) {
	admission := newEmbeddingProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}
	defer release()

	tests := []struct {
		name    string
		path    string
		body    string
		handler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)
	}{
		{
			name: "intent", path: "/api/v1/classify/intent", body: `{"text":"plain text classification"}`,
			handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
				s.handleIntentClassification(w, r)
			},
		},
		{
			name: "eval", path: "/api/v1/eval", body: `{"text":"plain text classification"}`,
			handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
				s.handleEvalClassification(w, r)
			},
		},
		{
			name: "combined", path: "/api/v1/classify/combined", body: `{"text":"plain text classification"}`,
			handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
				s.handleCombinedClassification(w, r)
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			service := &evalCaptureClassificationService{}
			server := &ClassificationAPIServer{
				classificationSvc:  service,
				embeddingAdmission: admission,
			}
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(server, rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("text-only status = %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "EMBEDDING_OVERLOADED")
			if service.intentCalls != 0 || service.evalCalls != 0 {
				t.Fatalf("overloaded text request reached service: intent=%d eval=%d", service.intentCalls, service.evalCalls)
			}
		})
	}
}

func TestClassifyTextSkipsAdmissionForPlaceholderService(t *testing.T) {
	admission := newEmbeddingProcessAdmission(1)
	release, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}
	defer release()

	server := &ClassificationAPIServer{
		classificationSvc:  services.NewPlaceholderClassificationService(),
		embeddingAdmission: admission,
	}
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/classify/intent",
		strings.NewReader(`{"text":"plain text classification"}`),
	)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	server.handleIntentClassification(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("placeholder status = %d: %s", rr.Code, rr.Body.String())
	}
}

func TestClassifyAndEvalRejectMalformedImageWithPlaceholderService(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	body := `{"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"describe the image"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + malformedClassifyImagePayload + `"}}]}]}`

	tests := []struct {
		name    string
		path    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{name: "intent", path: "/api/v1/classify/intent", handler: s.handleIntentClassification},
		{name: "eval", path: "/api/v1/eval", handler: s.handleEvalClassification},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(rr, req)

			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "INVALID_IMAGE")
			assertJSONErrorMessage(t, rr.Body.Bytes(), "image input must contain a decodable JPEG or PNG image within the supported limits")
			if rr.Header().Get("Cache-Control") != "no-store" || rr.Header().Get("Pragma") != "no-cache" {
				t.Fatalf("invalid image cache headers = %v", rr.Header())
			}
			if strings.Contains(rr.Body.String(), malformedClassifyImagePayload) {
				t.Fatalf("response reflected malformed image payload: %s", rr.Body.String())
			}
		})
	}
}

func TestClassifyAndEvalBoundImagePartsPerRequest(t *testing.T) {
	s := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
	}
	imageURI := mustEmbeddingImageDataURI(t, "image/png")
	parts := []map[string]interface{}{{"type": "text", "text": "describe the images"}}
	for i := 0; i <= imageurl.MaxImagePartsPerRequest; i++ {
		parts = append(parts, map[string]interface{}{
			"type":      "image_url",
			"image_url": map[string]string{"url": imageURI},
		})
	}
	body, err := json.Marshal(map[string]interface{}{
		"messages": []map[string]interface{}{{"role": "user", "content": parts}},
	})
	if err != nil {
		t.Fatalf("marshal image budget request: %v", err)
	}

	tests := []struct {
		path    string
		handler func(http.ResponseWriter, *http.Request)
	}{
		{path: "/api/v1/classify/intent", handler: s.handleIntentClassification},
		{path: "/api/v1/eval", handler: s.handleEvalClassification},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(string(body)))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()

			tt.handler(rr, req)

			if rr.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "INVALID_IMAGE")
			if rr.Header().Get("Cache-Control") != "no-store" || rr.Header().Get("Pragma") != "no-cache" {
				t.Fatalf("invalid image cache headers = %v", rr.Header())
			}
		})
	}
}

func TestWriteClassificationErrorDoesNotExposeInternalError(t *testing.T) {
	s := &ClassificationAPIServer{}
	rr := httptest.NewRecorder()

	s.writeClassificationError(rr, errors.New("private model path and runtime detail"))

	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
	assertJSONErrorCode(t, rr.Body.Bytes(), "CLASSIFICATION_ERROR")
	assertJSONErrorMessage(t, rr.Body.Bytes(), "classification failed")
	if rr.Header().Get("Cache-Control") != "" || rr.Header().Get("Pragma") != "" {
		t.Fatalf("non-image error unexpectedly received invalid-image cache headers: %v", rr.Header())
	}
	if strings.Contains(rr.Body.String(), "private model path") {
		t.Fatalf("response exposed internal error detail: %s", rr.Body.String())
	}
}

func TestClassifyAndEvalTextBackendFailureIsNonCacheable(t *testing.T) {
	tests := []struct {
		name    string
		path    string
		service *evalCaptureClassificationService
		handler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)
	}{
		{name: "intent", path: "/api/v1/classify/intent", service: &evalCaptureClassificationService{intentErr: classification.ErrTextSignalEvaluation}, handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
			s.handleIntentClassification(w, r)
		}},
		{name: "eval", path: "/api/v1/eval", service: &evalCaptureClassificationService{evalErr: classification.ErrTextSignalEvaluation}, handler: func(s *ClassificationAPIServer, w http.ResponseWriter, r *http.Request) {
			s.handleEvalClassification(w, r)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := &ClassificationAPIServer{classificationSvc: tt.service}
			req := httptest.NewRequest(http.MethodPost, tt.path, strings.NewReader(`{"text":"sensitive input"}`))
			req.Header.Set("Content-Type", "application/json")
			rr := httptest.NewRecorder()
			tt.handler(server, rr, req)

			if rr.Code != http.StatusServiceUnavailable {
				t.Fatalf("status = %d: %s", rr.Code, rr.Body.String())
			}
			assertJSONErrorCode(t, rr.Body.Bytes(), "CLASSIFICATION_ERROR")
			assertJSONErrorMessage(t, rr.Body.Bytes(), "classification temporarily unavailable")
			if rr.Header().Get("Cache-Control") != "no-store" || rr.Header().Get("Pragma") != "no-cache" || rr.Header().Get("Retry-After") != "1" {
				t.Fatalf("cache headers = %v", rr.Header())
			}
		})
	}
}

func assertJSONErrorCode(t *testing.T, body []byte, expected string) {
	t.Helper()
	var response struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("decode error response: %v", err)
	}
	if response.Error.Code != expected {
		t.Fatalf("expected error code %q, got %q: %s", expected, response.Error.Code, string(body))
	}
}
