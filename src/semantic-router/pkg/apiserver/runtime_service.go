//go:build !windows && cgo

package apiserver

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

type intentClassificationService interface {
	ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error)
	ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error)
	DetectPII(req services.PIIRequest) (*services.PIIResponse, error)
	CheckSecurity(ctx context.Context, req services.SecurityRequest) (*services.SecurityResponse, error)
}

type batchClassificationService interface {
	ClassifyBatchUnifiedWithOptions(texts []string, options interface{}) (*services.UnifiedBatchResponse, error)
	HasUnifiedClassifier() bool
}

type auxiliaryClassificationService interface {
	ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error)
	ClassifyUserFeedback(req services.UserFeedbackRequest) (*services.UserFeedbackResponse, error)
	ClassifyNLI(req services.NLIRequest) (*services.NLIResponse, error)
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
	acquirer func() (classificationService, func(), bool)
}

func newLiveClassificationService(
	fallback classificationService,
	resolver func() classificationService,
	acquirer func() (classificationService, func(), bool),
) classificationService {
	return &liveClassificationService{
		fallback: fallback,
		resolver: resolver,
		acquirer: acquirer,
	}
}

// acquire resolves the live service and holds a reference on the generation that
// owns it until release runs. Reload retirement waits on that reference, so the
// classifier cannot be closed underneath an in-flight API call.
func (s *liveClassificationService) acquire() (classificationService, func()) {
	if s != nil && s.acquirer != nil {
		if svc, release, ok := s.acquirer(); ok && svc != nil {
			return svc, release
		}
	}
	return s.current(), func() {}
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

func (s *liveClassificationService) ClassifyIntent(req services.IntentRequest) (*services.IntentResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyIntent(req)
}

func (s *liveClassificationService) ClassifyIntentForEval(req services.IntentRequest) (*services.EvalResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyIntentForEval(req)
}

func (s *liveClassificationService) DetectPII(req services.PIIRequest) (*services.PIIResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.DetectPII(req)
}

func (s *liveClassificationService) CheckSecurity(ctx context.Context, req services.SecurityRequest) (*services.SecurityResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.CheckSecurity(ctx, req)
}

func (s *liveClassificationService) ClassifyBatchUnifiedWithOptions(
	texts []string,
	options interface{},
) (*services.UnifiedBatchResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyBatchUnifiedWithOptions(texts, options)
}

func (s *liveClassificationService) ClassifyFactCheck(req services.FactCheckRequest) (*services.FactCheckResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyFactCheck(req)
}

func (s *liveClassificationService) ClassifyUserFeedback(
	req services.UserFeedbackRequest,
) (*services.UserFeedbackResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyUserFeedback(req)
}

func (s *liveClassificationService) ClassifyNLI(req services.NLIRequest) (*services.NLIResponse, error) {
	svc, release := s.acquire()
	defer release()
	return svc.ClassifyNLI(req)
}

func (s *liveClassificationService) IsNLIReady() bool {
	svc, release := s.acquire()
	defer release()
	return svc.IsNLIReady()
}

func (s *liveClassificationService) HasUnifiedClassifier() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasUnifiedClassifier()
}

func (s *liveClassificationService) HasClassifier() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasClassifier()
}

func (s *liveClassificationService) HasFactCheckClassifier() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasFactCheckClassifier()
}

func (s *liveClassificationService) HasHallucinationDetector() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasHallucinationDetector()
}

func (s *liveClassificationService) HasHallucinationExplainer() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasHallucinationExplainer()
}

func (s *liveClassificationService) HasFeedbackDetector() bool {
	svc, release := s.acquire()
	defer release()
	return svc.HasFeedbackDetector()
}

func (s *liveClassificationService) HasAnyFactCheckClassifier() bool {
	current, release := s.acquire()
	defer release()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyFactCheckClassifier()
	}
	return current.HasFactCheckClassifier()
}

func (s *liveClassificationService) HasAnyHallucinationDetector() bool {
	current, release := s.acquire()
	defer release()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyHallucinationDetector()
	}
	return current.HasHallucinationDetector()
}

func (s *liveClassificationService) HasAnyHallucinationExplainer() bool {
	current, release := s.acquire()
	defer release()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyHallucinationExplainer()
	}
	return current.HasHallucinationExplainer()
}

func (s *liveClassificationService) HasAnyFeedbackDetector() bool {
	current, release := s.acquire()
	defer release()
	if inventory, ok := current.(classificationInventoryReadinessService); ok {
		return inventory.HasAnyFeedbackDetector()
	}
	return current.HasFeedbackDetector()
}

func (s *liveClassificationService) UpdateConfig(newConfig *config.RouterConfig) {
	svc, release := s.acquire()
	defer release()
	svc.UpdateConfig(newConfig)
}

func (s *liveClassificationService) RefreshRuntimeConfig(newConfig *config.RouterConfig) {
	svc, release := s.acquire()
	defer release()
	svc.RefreshRuntimeConfig(newConfig)
}
