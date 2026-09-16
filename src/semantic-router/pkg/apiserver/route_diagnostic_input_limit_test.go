//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type diagnosticErrorService struct {
	evalCaptureClassificationService
	err error
}

func (s *diagnosticErrorService) CheckSecurity(context.Context, services.SecurityRequest) (*services.SecurityResponse, error) {
	return nil, s.err
}

func TestDiagnosticInputLimitResponses(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want int
	}{
		{"input limit", fmt.Errorf("model inference: %w: at most 8192 tokens, got 9003", binding.ErrInputLimit), http.StatusBadRequest},
		{"native failure", errors.New("model inference failed"), http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider, err := embedding.NewFuncProvider("test", 768, func(context.Context, string) ([]float32, error) {
				return nil, tc.err
			})
			if err != nil {
				t.Fatal(err)
			}
			cfg := &config.RouterConfig{}
			prepared := embedding.NewSet(map[string]embedding.Provider{"mmbert": provider}, "mmbert")
			classifiers, err := classification.BuildRecipeClassifiers(cfg, nil, nil, nil, classification.RecipeRuntimeOptions{Embeddings: prepared})
			if err != nil {
				t.Fatal(err)
			}
			service := services.NewRecipeClassificationService(classifiers, cfg)
			t.Cleanup(func() { _ = service.Close() })
			embeddingAPI := &ClassificationAPIServer{config: cfg, classificationSvc: service}
			classifierAPI := &ClassificationAPIServer{classificationSvc: &diagnosticErrorService{err: tc.err}}
			for _, endpoint := range []struct {
				path    string
				body    string
				handler http.HandlerFunc
			}{
				{"classify/security", `{"text":"hello"}`, classifierAPI.handleSecurityDetection},
				{"embeddings", `{"texts":["hello"],"model":"mmbert"}`, embeddingAPI.handleEmbeddings},
				{"similarity", `{"text1":"hello","text2":"world","model":"mmbert"}`, embeddingAPI.handleSimilarity},
				{"similarity/batch", `{"query":"hello","candidates":["world"],"model":"mmbert"}`, embeddingAPI.handleBatchSimilarity},
			} {
				t.Run(endpoint.path, func(t *testing.T) {
					request := httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/"+endpoint.path, strings.NewReader(endpoint.body))
					response := httptest.NewRecorder()
					endpoint.handler(response, request)
					if response.Code != tc.want {
						t.Fatalf("status = %d, want %d: %s", response.Code, tc.want, response.Body.String())
					}
					if !strings.Contains(response.Body.String(), tc.err.Error()) {
						t.Fatalf("response lost model error details: %s", response.Body.String())
					}
					if tc.want == http.StatusBadRequest && !strings.Contains(response.Body.String(), "INVALID_INPUT") {
						t.Fatalf("input limit response = %s, want INVALID_INPUT", response.Body.String())
					}
				})
			}
		})
	}
}
