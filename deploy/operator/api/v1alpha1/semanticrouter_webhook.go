/*
Copyright 2026 vLLM Semantic Router Contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1alpha1

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config/authoring"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var semanticrouterlog = logf.Log.WithName("semanticrouter-resource")

// SetupWebhookWithManager registers the webhook with the manager
func (r *SemanticRouter) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(&semanticRouterValidator{}).
		Complete()
}

// +kubebuilder:webhook:path=/validate-vllm-ai-v1alpha1-semanticrouter,mutating=false,failurePolicy=fail,sideEffects=None,groups=vllm.ai,resources=semanticrouters,verbs=create;update,versions=v1alpha1,name=vsemanticrouter.kb.io,admissionReviewVersions=v1

var _ admission.CustomValidator = &semanticRouterValidator{}

type semanticRouterValidator struct{}

func (*semanticRouterValidator) ValidateCreate(_ context.Context, obj runtime.Object) (admission.Warnings, error) {
	sr, ok := obj.(*SemanticRouter)
	if !ok {
		return nil, fmt.Errorf("expected *SemanticRouter, got %T", obj)
	}
	semanticrouterlog.Info("validate create", "name", sr.Name)
	return nil, sr.validateSemanticRouter()
}

func (*semanticRouterValidator) ValidateUpdate(_ context.Context, _, newObj runtime.Object) (admission.Warnings, error) {
	sr, ok := newObj.(*SemanticRouter)
	if !ok {
		return nil, fmt.Errorf("expected *SemanticRouter, got %T", newObj)
	}
	semanticrouterlog.Info("validate update", "name", sr.Name)
	return nil, sr.validateSemanticRouter()
}

func (*semanticRouterValidator) ValidateDelete(_ context.Context, obj runtime.Object) (admission.Warnings, error) {
	sr, ok := obj.(*SemanticRouter)
	if !ok {
		return nil, fmt.Errorf("expected *SemanticRouter, got %T", obj)
	}
	semanticrouterlog.Info("validate delete", "name", sr.Name)
	return nil, nil
}

// ValidateCreate keeps the existing direct-call validation seam used by local tests.
func (r *SemanticRouter) ValidateCreate() (admission.Warnings, error) {
	semanticrouterlog.Info("validate create", "name", r.Name)
	return nil, r.validateSemanticRouter()
}

// ValidateUpdate keeps the existing direct-call validation seam used by local tests.
func (r *SemanticRouter) ValidateUpdate(old runtime.Object) (admission.Warnings, error) {
	semanticrouterlog.Info("validate update", "name", r.Name)
	return nil, r.validateSemanticRouter()
}

// ValidateDelete keeps the existing direct-call validation seam used by local tests.
func (r *SemanticRouter) ValidateDelete() (admission.Warnings, error) {
	semanticrouterlog.Info("validate delete", "name", r.Name)
	// No validation needed on delete
	return nil, nil
}

// validateSemanticRouter validates the SemanticRouter resource
func (r *SemanticRouter) validateSemanticRouter() error {
	// Validate autoscaling configuration
	if r.Spec.Autoscaling.Enabled != nil && *r.Spec.Autoscaling.Enabled {
		if err := r.validateAutoscaling(); err != nil {
			return err
		}
	}

	// Validate persistence configuration
	if err := r.validatePersistence(); err != nil {
		return err
	}

	// Validate probe configurations
	if err := r.validateProbes(); err != nil {
		return err
	}

	// Validate ingress configuration
	if r.Spec.Ingress.Enabled != nil && *r.Spec.Ingress.Enabled {
		if err := r.validateIngress(); err != nil {
			return err
		}
	}

	if err := r.validateConfigSources(); err != nil {
		return err
	}

	return nil
}

func (r *SemanticRouter) validateConfigSources() error {
	if r.Spec.AuthoringConfig == nil {
		return nil
	}
	if len(r.Spec.AuthoringConfig.Raw) == 0 {
		return fmt.Errorf("spec.authoringConfig must not be empty")
	}
	if len(r.Spec.VLLMEndpoints) > 0 {
		return fmt.Errorf("spec.authoringConfig cannot be combined with spec.vllmEndpoints")
	}
	if len(r.Spec.Config.Decisions) > 0 {
		return fmt.Errorf("spec.authoringConfig cannot be combined with spec.config.decisions")
	}
	if len(r.Spec.Config.ReasoningFamilies) > 0 {
		return fmt.Errorf("spec.authoringConfig cannot be combined with spec.config.reasoning_families")
	}
	if r.Spec.Config.DefaultReasoningEffort != "" {
		return fmt.Errorf("spec.authoringConfig cannot be combined with spec.config.default_reasoning_effort")
	}
	if _, err := authoring.Parse(r.Spec.AuthoringConfig.Raw); err != nil {
		return fmt.Errorf("spec.authoringConfig: %w", err)
	}
	return nil
}

// validateAutoscaling validates HPA configuration
func (r *SemanticRouter) validateAutoscaling() error {
	if r.Spec.Autoscaling.MinReplicas != nil && r.Spec.Autoscaling.MaxReplicas != nil {
		if *r.Spec.Autoscaling.MinReplicas > *r.Spec.Autoscaling.MaxReplicas {
			return fmt.Errorf("autoscaling.minReplicas (%d) must be less than or equal to maxReplicas (%d)",
				*r.Spec.Autoscaling.MinReplicas, *r.Spec.Autoscaling.MaxReplicas)
		}
	}

	// Ensure at least one metric is specified
	if r.Spec.Autoscaling.TargetCPUUtilizationPercentage == nil &&
		r.Spec.Autoscaling.TargetMemoryUtilizationPercentage == nil {
		return fmt.Errorf("autoscaling requires at least one metric (targetCPUUtilizationPercentage or targetMemoryUtilizationPercentage)")
	}

	return nil
}

// validatePersistence validates PVC configuration
func (r *SemanticRouter) validatePersistence() error {
	if r.Spec.Persistence.Enabled != nil && *r.Spec.Persistence.Enabled {
		// Cannot specify both existingClaim and storageClassName
		if r.Spec.Persistence.ExistingClaim != "" && r.Spec.Persistence.StorageClassName != "" {
			return fmt.Errorf("cannot specify both persistence.existingClaim and persistence.storageClassName")
		}
	}
	return nil
}

// validateProbes validates probe configurations
func (r *SemanticRouter) validateProbes() error {
	if err := r.validateOptionalProbe("startupProbe", r.Spec.StartupProbe); err != nil {
		return err
	}
	if err := r.validateOptionalProbe("livenessProbe", r.Spec.LivenessProbe); err != nil {
		return err
	}
	return r.validateOptionalProbe("readinessProbe", r.Spec.ReadinessProbe)
}

func (r *SemanticRouter) validateOptionalProbe(name string, probe *ProbeSpec) error {
	if probe == nil {
		return nil
	}
	if probe.Enabled != nil && !*probe.Enabled {
		return nil
	}
	return r.validateProbeSpec(name, probe)
}

// validateProbeSpec validates a single probe specification
func (r *SemanticRouter) validateProbeSpec(name string, probe *ProbeSpec) error {
	if probe.TimeoutSeconds != nil && *probe.TimeoutSeconds <= 0 {
		return fmt.Errorf("%s.timeoutSeconds must be greater than 0", name)
	}
	if probe.PeriodSeconds != nil && *probe.PeriodSeconds <= 0 {
		return fmt.Errorf("%s.periodSeconds must be greater than 0", name)
	}
	if probe.FailureThreshold != nil && *probe.FailureThreshold <= 0 {
		return fmt.Errorf("%s.failureThreshold must be greater than 0", name)
	}
	return nil
}

// validateIngress validates ingress configuration
func (r *SemanticRouter) validateIngress() error {
	if len(r.Spec.Ingress.Hosts) == 0 {
		return fmt.Errorf("ingress.hosts must be specified when ingress is enabled")
	}

	for i, host := range r.Spec.Ingress.Hosts {
		if host.Host == "" {
			return fmt.Errorf("ingress.hosts[%d].host cannot be empty", i)
		}
		if len(host.Paths) == 0 {
			return fmt.Errorf("ingress.hosts[%d].paths must have at least one path", i)
		}
	}

	return nil
}
