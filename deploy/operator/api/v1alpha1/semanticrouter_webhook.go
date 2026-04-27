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

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

// log is for logging in this package.
var semanticrouterlog = logf.Log.WithName("semanticrouter-resource")

// SetupWebhookWithManager registers the webhook with the manager
func (r *SemanticRouter) SetupWebhookWithManager(mgr ctrl.Manager) error {
	return ctrl.NewWebhookManagedBy(mgr).
		For(r).
		WithValidator(r).
		Complete()
}

// +kubebuilder:webhook:path=/validate-vllm-ai-v1alpha1-semanticrouter,mutating=false,failurePolicy=fail,sideEffects=None,groups=vllm.ai,resources=semanticrouters,verbs=create;update,versions=v1alpha1,name=vsemanticrouter.kb.io,admissionReviewVersions=v1

var _ webhook.CustomValidator = &SemanticRouter{}

// ValidateCreate implements webhook.Validator so a webhook will be registered for the type
func (r *SemanticRouter) ValidateCreate(_ context.Context, obj runtime.Object) (admission.Warnings, error) {
	semanticrouterlog.Info("validate create", "name", r.Name)
	semanticRouter, ok := obj.(*SemanticRouter)
	if !ok {
		return nil, fmt.Errorf("expected SemanticRouter for create validation, got %T", obj)
	}
	return nil, semanticRouter.validateSemanticRouter()
}

// ValidateUpdate implements webhook.Validator so a webhook will be registered for the type
func (r *SemanticRouter) ValidateUpdate(_ context.Context, _ runtime.Object, newObj runtime.Object) (admission.Warnings, error) {
	semanticrouterlog.Info("validate update", "name", r.Name)
	semanticRouter, ok := newObj.(*SemanticRouter)
	if !ok {
		return nil, fmt.Errorf("expected SemanticRouter for update validation, got %T", newObj)
	}
	return nil, semanticRouter.validateSemanticRouter()
}

// ValidateDelete implements webhook.Validator so a webhook will be registered for the type
func (r *SemanticRouter) ValidateDelete(_ context.Context, _ runtime.Object) (admission.Warnings, error) {
	semanticrouterlog.Info("validate delete", "name", r.Name)
	return nil, nil
}
