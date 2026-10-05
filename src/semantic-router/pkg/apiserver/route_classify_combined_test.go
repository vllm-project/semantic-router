//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type combinedBatchTestService struct {
	classificationService
	calls           []string
	intentOptions   *services.IntentOptions
	piiOptions      *services.PIIOptions
	securityOptions *services.SecurityOptions
	piiErrors       map[string]error
	cancel          context.CancelFunc
	cancelOnIntent  bool
}

func (s *combinedBatchTestService) ClassifyIntent(_ context.Context, req services.IntentRequest) (*services.IntentResponse, error) {
	s.calls = append(s.calls, "intent:"+req.Text)
	s.intentOptions = req.Options
	if s.cancelOnIntent && s.cancel != nil {
		s.cancel()
	}
	if req.Text == "" {
		return nil, services.ErrEmptyText
	}
	return &services.IntentResponse{Classification: services.Classification{Category: req.Text}}, nil
}

func (s *combinedBatchTestService) DetectPII(_ context.Context, req services.PIIRequest) (*services.PIIResponse, error) {
	s.calls = append(s.calls, "pii:"+req.Text)
	s.piiOptions = req.Options
	if req.Text == "" {
		return nil, services.ErrEmptyText
	}
	if err := s.piiErrors[req.Text]; err != nil {
		return nil, err
	}
	return &services.PIIResponse{Recipe: req.Recipe, Entities: []services.PIIEntity{}, SecurityRecommendation: "allow"}, nil
}

func (s *combinedBatchTestService) CheckSecurity(_ context.Context, req services.SecurityRequest) (*services.SecurityResponse, error) {
	s.calls = append(s.calls, "security:"+req.Text)
	s.securityOptions = req.Options
	if req.Text == "" {
		return nil, services.ErrEmptyText
	}
	return &services.SecurityResponse{Recipe: req.Recipe, DetectionTypes: []string{}, PatternsDetected: []string{}, Recommendation: "allow"}, nil
}

func newCombinedBatchTestService() *combinedBatchTestService {
	return &combinedBatchTestService{piiErrors: map[string]error{}}
}

type combinedBatchRecipeService struct {
	*combinedBatchTestService
	requestedRecipes []string
	recipeErr        error
}

func (s *combinedBatchRecipeService) AcquireRecipeService(recipe string) (*services.ClassificationService, func(), error) {
	s.requestedRecipes = append(s.requestedRecipes, recipe)
	return nil, func() {}, s.recipeErr
}

func TestHandleCombinedBatchClassificationReturnsOrderedResults(t *testing.T) {
	service := newCombinedBatchTestService()
	apiServer := &ClassificationAPIServer{classificationSvc: service, config: &config.RouterConfig{}}

	body := `{"texts":["first","second"],"intent_options":{"include_explanation":true},"pii_options":{"return_positions":true},"security_options":{"include_reasoning":true}}`
	recorder := httptest.NewRecorder()
	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/api/v1/diagnostics/classify/combined/batch", bytes.NewBufferString(body)))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response CombinedClassificationBatchResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Recipe != string(config.DefaultRecipeName) || response.TotalCount != 2 || len(response.Results) != 2 {
		t.Fatalf("unexpected batch response metadata: %+v", response)
	}
	if response.Results[0].Index != 0 || response.Results[1].Index != 1 {
		t.Fatalf("unexpected result indexes: %+v", response.Results)
	}
	if response.Results[0].Intent.Classification.Category != "first" || response.Results[1].Intent.Classification.Category != "second" {
		t.Fatalf("results are not aligned with inputs: %+v", response.Results)
	}
	if response.Results[0].PII == nil || response.Results[0].Security == nil || len(response.Results[0].Errors) != 0 {
		t.Fatalf("first result is incomplete: %+v", response.Results[0])
	}
	if !service.intentOptions.IncludeExplanation || !service.piiOptions.ReturnPositions || !service.securityOptions.IncludeReasoning {
		t.Fatalf("combined options were not forwarded: intent=%+v pii=%+v security=%+v", service.intentOptions, service.piiOptions, service.securityOptions)
	}
	wantCalls := []string{"intent:first", "pii:first", "security:first", "intent:second", "pii:second", "security:second"}
	if !slices.Equal(service.calls, wantCalls) {
		t.Fatalf("calls = %v, want %v", service.calls, wantCalls)
	}
}

func TestHandleCombinedBatchClassificationKeepsStageErrorsPerItem(t *testing.T) {
	service := newCombinedBatchTestService()
	service.piiErrors["bad-pii"] = errors.New("PII backend failed")
	apiServer := &ClassificationAPIServer{classificationSvc: service, config: &config.RouterConfig{}}

	recorder := httptest.NewRecorder()
	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(
		http.MethodPost,
		"/api/v1/diagnostics/classify/combined/batch",
		bytes.NewBufferString(`{"texts":["bad-pii","good"]}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response CombinedClassificationBatchResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	failed := response.Results[0]
	if failed.Intent == nil || failed.PII != nil || failed.Security == nil {
		t.Fatalf("failed stage should not discard independent results: %+v", failed)
	}
	if got := failed.Errors["pii"].Code; got != "CLASSIFICATION_ERROR" {
		t.Fatalf("pii error code = %q, want CLASSIFICATION_ERROR", got)
	}
	if len(response.Results[1].Errors) != 0 || response.Results[1].Intent == nil || response.Results[1].Security == nil {
		t.Fatalf("unrelated item was not completed: %+v", response.Results[1])
	}
}

func TestHandleCombinedBatchClassificationRejectsInvalidBatches(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
		want string
	}{
		{name: "missing texts", body: `{}`, want: "texts field is required"},
		{name: "empty texts", body: `{"texts":[]}`, want: "texts array cannot be empty"},
	} {
		t.Run(test.name, func(t *testing.T) {
			apiServer := &ClassificationAPIServer{classificationSvc: newCombinedBatchTestService(), config: &config.RouterConfig{}}
			recorder := httptest.NewRecorder()
			apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(test.body)))
			if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), test.want) {
				t.Fatalf("status/body = %d/%s, want 400 containing %q", recorder.Code, recorder.Body.String(), test.want)
			}
		})
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: newCombinedBatchTestService(),
		config: &config.RouterConfig{
			APIServer: config.APIServer{API: config.APIConfig{
				BatchClassification: config.BatchClassificationConfig{MaxBatchSize: 1},
			}},
		},
	}
	recorder := httptest.NewRecorder()
	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"texts":["one","two"]}`)))
	if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), "max_batch_size 1") {
		t.Fatalf("status/body = %d/%s, want max batch size rejection", recorder.Code, recorder.Body.String())
	}
}

func TestHandleCombinedBatchClassificationUsesOneGeneration(t *testing.T) {
	oldGeneration := newCombinedBatchTestService()
	newGeneration := newCombinedBatchTestService()
	acquires := 0
	releases := 0
	apiServer := &ClassificationAPIServer{
		classificationSvc: newLiveClassificationService(nil, nil, func() (classificationService, func(), bool) {
			acquires++
			if acquires == 1 {
				return oldGeneration, func() { releases++ }, true
			}
			return newGeneration, func() { releases++ }, true
		}),
	}

	recorder := httptest.NewRecorder()
	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(
		http.MethodPost,
		"/api/v1/diagnostics/classify/combined/batch",
		strings.NewReader(`{"texts":["one","two"]}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	if acquires != 1 || releases != 1 {
		t.Fatalf("generation acquired %d/released %d times, want once each", acquires, releases)
	}
	if len(newGeneration.calls) != 0 || len(oldGeneration.calls) != 6 {
		t.Fatalf("old/new generation calls = %v/%v", oldGeneration.calls, newGeneration.calls)
	}
}

func TestHandleCombinedBatchClassificationStopsAfterRequestCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	service := newCombinedBatchTestService()
	service.cancel = cancel
	service.cancelOnIntent = true
	apiServer := &ClassificationAPIServer{classificationSvc: service, config: &config.RouterConfig{}}
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/diagnostics/classify/combined/batch",
		strings.NewReader(`{"texts":["one","two"]}`),
	).WithContext(ctx)
	recorder := httptest.NewRecorder()

	apiServer.handleCombinedBatchClassification(recorder, req)

	if recorder.Body.Len() != 0 {
		t.Fatalf("canceled request wrote a response: %s", recorder.Body.String())
	}
	if !slices.Equal(service.calls, []string{"intent:one"}) {
		t.Fatalf("calls after cancellation = %v, want only first intent", service.calls)
	}
}

func TestHandleCombinedBatchClassificationRejectsUnknownRecipe(t *testing.T) {
	service := &combinedBatchRecipeService{
		combinedBatchTestService: newCombinedBatchTestService(),
		recipeErr:                services.ErrUnknownDiagnosticRecipe,
	}
	apiServer := &ClassificationAPIServer{classificationSvc: service, config: &config.RouterConfig{}}
	recorder := httptest.NewRecorder()

	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(
		http.MethodPost,
		"/api/v1/diagnostics/classify/combined/batch",
		strings.NewReader(`{"recipe":"foreign","texts":["one"]}`),
	))

	if recorder.Code != http.StatusBadRequest || !strings.Contains(recorder.Body.String(), "INVALID_RECIPE") {
		t.Fatalf("status/body = %d/%s, want INVALID_RECIPE", recorder.Code, recorder.Body.String())
	}
	if !slices.Equal(service.requestedRecipes, []string{"foreign"}) || len(service.calls) != 0 {
		t.Fatalf("recipe selection/calls = %v/%v", service.requestedRecipes, service.calls)
	}
}

func TestHandleCombinedBatchClassificationAcceptsConfiguredRecipe(t *testing.T) {
	cfg := &config.RouterConfig{Recipes: []config.RoutingRecipe{
		{Name: config.DefaultRecipeName},
		{Name: "private"},
	}}
	_, service := preparedInventoryService(t, native.New(nil), cfg)
	t.Cleanup(func() { _ = service.Close() })
	apiServer := &ClassificationAPIServer{classificationSvc: service, config: cfg}
	recorder := httptest.NewRecorder()

	apiServer.handleCombinedBatchClassification(recorder, httptest.NewRequest(
		http.MethodPost,
		"/api/v1/diagnostics/classify/combined/batch",
		strings.NewReader(`{"recipe":"private","texts":["hello"]}`),
	))

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusOK, recorder.Body.String())
	}
	var response CombinedClassificationBatchResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if response.Recipe != "private" || len(response.Results) != 1 {
		t.Fatalf("configured recipe was not preserved: %+v", response)
	}
	if strings.Contains(recorder.Body.String(), "INVALID_RECIPE") {
		t.Fatalf("configured recipe was rejected: %s", recorder.Body.String())
	}
}
