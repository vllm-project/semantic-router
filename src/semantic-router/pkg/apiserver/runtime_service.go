//go:build !windows && cgo

package apiserver

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type intentClassificationService interface {
	ClassifyIntent(ctx context.Context, req services.IntentRequest) (*services.IntentResponse, error)
	ClassifyIntentForEval(ctx context.Context, req services.IntentRequest) (*services.EvalResponse, error)
	DetectPII(ctx context.Context, req services.PIIRequest) (*services.PIIResponse, error)
	CheckSecurity(ctx context.Context, req services.SecurityRequest) (*services.SecurityResponse, error)
}

type batchClassificationService interface {
	ClassifyBatchUnifiedWithOptions(texts []string, options interface{}) (*services.UnifiedBatchResponse, error)
	HasUnifiedClassifier() bool
}

type auxiliaryClassificationService interface {
	ClassifyFactCheck(ctx context.Context, req services.FactCheckRequest) (*services.FactCheckResponse, error)
	ClassifyUserFeedback(ctx context.Context, req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error)
	ClassifyNLI(ctx context.Context, req services.NLIRequest) (*services.NLIResponse, error)
	IsNLIReady() bool
	HasClassifier() bool
}

type classificationReadinessService interface {
	HasFactCheckClassifier() bool
	HasHallucinationDetector() bool
	HasHallucinationExplainer() bool
	HasFeedbackDetector() bool
}

type classificationInventoryReadinessService interface {
	HasAnyFactCheckClassifier() bool
	HasAnyHallucinationDetector() bool
	HasAnyHallucinationExplainer() bool
	HasAnyFeedbackDetector() bool
}

type configUpdateService interface {
	UpdateConfig(newConfig *config.RouterConfig)
	RefreshRuntimeConfig(newConfig *config.RouterConfig)
}

type classificationService interface {
	intentClassificationService
	batchClassificationService
	auxiliaryClassificationService
	classificationReadinessService
	configUpdateService
}

type liveClassificationService struct {
	fallback classificationService
	resolver func() classificationService
}

func newLiveClassificationService(
	fallback classificationService,
	resolver func() classificationService,
) classificationService {
	return &liveClassificationService{
		fallback: fallback,
		resolver: resolver,
	}
}

func (s *liveClassificationService) current() classificationService {
	if s != nil && s.resolver != nil {
		if svc := s.resolver(); svc != nil {
			return svc
		}
	}
	if s != nil && s.fallback != nil {
		return s.fallback
	}
	return services.NewPlaceholderClassificationService()
}

func (s *liveClassificationService) ClassifyIntent(ctx context.Context, req services.IntentRequest) (*services.IntentResponse, error) {
	return s.current().ClassifyIntent(ctx, req)
}

func (s *liveClassificationService) ClassifyIntentForEval(ctx context.Context, req services.IntentRequest) (*services.EvalResponse, error) {
	return s.current().ClassifyIntentForEval(ctx, req)
}

func (s *liveClassificationService) DetectPII(ctx context.Context, req services.PIIRequest) (*services.PIIResponse, error) {
	return s.current().DetectPII(ctx, req)
}

func (s *liveClassificationService) CheckSecurity(ctx context.Context, req services.SecurityRequest) (*services.SecurityResponse, error) {
	return s.current().CheckSecurity(ctx, req)
}

func (s *liveClassificationService) ClassifyBatchUnifiedWithOptions(
	texts []string,
	options interface{},
) (*services.UnifiedBatchResponse, error) {
	return s.current().ClassifyBatchUnifiedWithOptions(texts, options)
}

func (s *liveClassificationService) ClassifyFactCheck(ctx context.Context, req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	return s.current().ClassifyFactCheck(ctx, req)
}

func (s *liveClassificationService) ClassifyUserFeedback(
	ctx context.Context,
	req services.UserFeedbackRequest,
) (*services.UserFeedbackResponse, error) {
	return s.current().ClassifyUserFeedback(ctx, req)
}

func (s *liveClassificationService) ClassifyNLI(ctx context.Context, req services.NLIRequest) (*services.NLIResponse, error) {
	return s.current().ClassifyNLI(ctx, req)
}

func (s *liveClassificationService) IsNLIReady() bool {
	return s.current().IsNLIReady()
}

func (s *liveClassificationService) HasUnifiedClassifier() bool {
	return s.current().HasUnifiedClassifier()
}

func (s *liveClassificationService) HasClassifier() bool {
	return s.current().HasClassifier()
}

func (s *liveClassificationService) HasFactCheckClassifier() bool {
	return s.current().HasFactCheckClassifier()
}

func (s *liveClassificationService) HasHallucinationDetector() bool {
	return s.current().HasHallucinationDetector()
}

func (s *liveClassificationService) HasHallucinationExplainer() bool {
	return s.current().HasHallucinationExplainer()
}

func (s *liveClassificationService) HasFeedbackDetector() bool {
	return s.current().HasFeedbackDetector()
}

func (s *liveClassificationService) HasAnyFactCheckClassifier() bool {
	current := s.current()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyFactCheckClassifier()
	}
	return current.HasFactCheckClassifier()
}

func (s *liveClassificationService) HasAnyHallucinationDetector() bool {
	current := s.current()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyHallucinationDetector()
	}
	return current.HasHallucinationDetector()
}

func (s *liveClassificationService) HasAnyHallucinationExplainer() bool {
	current := s.current()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyHallucinationExplainer()
	}
	return current.HasHallucinationExplainer()
}

func (s *liveClassificationService) HasAnyFeedbackDetector() bool {
	current := s.current()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyFeedbackDetector()
	}
	return current.HasFeedbackDetector()
}

func (s *liveClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	s.current().UpdateConfig(newConfig)
}

func (s *liveClassificationService) RefreshRuntimeConfig(newConfig *config.RouterConfig) {
	s.current().RefreshRuntimeConfig(newConfig)
}
