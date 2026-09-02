package classification

import (
	"context"
	"errors"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const (
	admissionDeploymentPromptGuard            = "prompt_guard"
	admissionDeploymentDomainClassifier       = "domain_classifier"
	admissionDeploymentPIIClassifier          = "pii_classifier"
	admissionDeploymentFactCheckClassifier    = "fact_check_classifier"
	admissionDeploymentHallucinationDetector  = "hallucination_detector"
	admissionDeploymentHallucinationExplainer = "hallucination_explainer"
	admissionDeploymentFeedbackDetector       = "feedback_detector"
)

func buildAdmissionRegistry(cfg *config.RouterConfig) *admission.Registry {
	if cfg == nil || len(cfg.ModelAdmission) == 0 {
		return admission.NewRegistry(nil)
	}
	gates := make(map[string]admission.Admissioner, len(cfg.ModelAdmission))
	for deployment, admissionCfg := range cfg.ModelAdmission {
		gates[deployment] = admission.NewSemaphore(
			admissionCfg.MaxConcurrency,
			admissionCfg.MaxQueue,
			time.Duration(admissionCfg.QueueTimeoutMs)*time.Millisecond,
			admission.Overflow(admissionCfg.OnOverflow),
		)
	}
	return admission.NewRegistry(gates)
}

// admitModelInference runs fn under the deployment's admission gate and records
// the gate outcome. A shed request surfaces admission.ErrQueueFull to the
// caller, where the existing classify-error handling (on_error / on_unknown)
// applies.
func admitModelInference[T any](
	ctx context.Context,
	gate admission.Admissioner,
	deployment string,
	fn func() (T, error),
) (T, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if gate == nil {
		gate = admission.Noop{}
	}
	start := time.Now()
	ticket, err := gate.Acquire(ctx)
	wait := time.Since(start).Seconds()
	if err != nil {
		outcome := "canceled"
		if errors.Is(err, admission.ErrQueueFull) {
			outcome = "shed"
		}
		metrics.RecordModelAdmission(deployment, outcome, wait)
		var zero T
		return zero, err
	}
	defer ticket()
	metrics.RecordModelAdmission(deployment, "admitted", wait)
	return fn()
}

func admitNLI[T any](
	ctx context.Context,
	gate admission.Admissioner,
	classify func(premise, hypothesis string) (T, error),
	premise, hypothesis string,
) (T, error) {
	return admitModelInference(ctx, gate, admissionDeploymentHallucinationExplainer, func() (T, error) {
		return classify(premise, hypothesis)
	})
}

type admittedSequenceClassifier struct {
	backend    SequenceClassifierBackend
	gate       admission.Admissioner
	deployment string
}

func (a admittedSequenceClassifier) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	return admitModelInference(ctx, a.gate, a.deployment, func() (SequenceClassificationResult, error) {
		return a.backend.Classify(ctx, text)
	})
}

type admittedCategoryInference struct {
	backend    CategoryInference
	gate       admission.Admissioner
	deployment string
}

func (a admittedCategoryInference) Classify(ctx context.Context, text string) (candle_binding.ClassResult, error) {
	return admitModelInference(ctx, a.gate, a.deployment, func() (candle_binding.ClassResult, error) {
		return a.backend.Classify(ctx, text)
	})
}

func (a admittedCategoryInference) ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error) {
	return admitModelInference(ctx, a.gate, a.deployment, func() (candle_binding.ClassResultWithProbs, error) {
		return a.backend.ClassifyWithProbabilities(ctx, text)
	})
}

type admittedPIIInference struct {
	backend    PIIInference
	gate       admission.Admissioner
	deployment string
}

func (a admittedPIIInference) ClassifyTokens(ctx context.Context, text string) (candle_binding.TokenClassificationResult, error) {
	return admitModelInference(ctx, a.gate, a.deployment, func() (candle_binding.TokenClassificationResult, error) {
		return a.backend.ClassifyTokens(ctx, text)
	})
}

// withAdmissionRegistry shares one admission registry across classifiers. The
// underlying models are process-wide singletons, so recipe classifiers must
// share gates or every recipe multiplies the configured concurrency bound.
func withAdmissionRegistry(registry *admission.Registry) option {
	return func(c *Classifier) {
		c.admissionRegistry = registry
	}
}

func (c *Classifier) applyAdmissionGates() {
	registry := c.admissionRegistry
	if registry == nil {
		registry = buildAdmissionRegistry(c.Config)
		c.admissionRegistry = registry
	}
	if c.jailbreakInference != nil {
		c.jailbreakInference = admittedSequenceClassifier{
			backend:    c.jailbreakInference,
			gate:       registry.For(admissionDeploymentPromptGuard),
			deployment: admissionDeploymentPromptGuard,
		}
	}
	if c.categoryInference != nil {
		c.categoryInference = admittedCategoryInference{
			backend:    c.categoryInference,
			gate:       registry.For(admissionDeploymentDomainClassifier),
			deployment: admissionDeploymentDomainClassifier,
		}
	}
	if c.piiInference != nil {
		c.piiInference = admittedPIIInference{
			backend:    c.piiInference,
			gate:       registry.For(admissionDeploymentPIIClassifier),
			deployment: admissionDeploymentPIIClassifier,
		}
	}
}
