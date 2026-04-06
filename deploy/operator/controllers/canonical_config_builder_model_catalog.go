package controllers

import (
	"fmt"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

func (r *SemanticRouterReconciler) applyOperatorModelCatalog(
	canonical *routercontract.CanonicalConfig,
	spec vllmv1alpha1.ConfigSpec,
) error {
	if spec.EmbeddingModels != nil {
		embeddings, err := convertToTypedConfig[routercontract.EmbeddingModels](r, spec.EmbeddingModels)
		if err != nil {
			return fmt.Errorf("config.embedding_models: %w", err)
		}
		canonical.Global.ModelCatalog.Embeddings.Semantic = embeddings
	}

	if spec.PromptGuard != nil {
		promptGuard, err := convertToTypedConfig[routercontract.CanonicalPromptGuardModule](r, spec.PromptGuard)
		if err != nil {
			return fmt.Errorf("config.prompt_guard: %w", err)
		}
		canonical.Global.ModelCatalog.Modules.PromptGuard = promptGuard
	}

	if spec.Classifier != nil {
		classifier, err := r.convertClassifierModule(spec.Classifier)
		if err != nil {
			return fmt.Errorf("config.classifier: %w", err)
		}
		canonical.Global.ModelCatalog.Modules.Classifier = classifier
	}
	return nil
}

func (r *SemanticRouterReconciler) convertClassifierModule(
	spec *vllmv1alpha1.ClassifierConfig,
) (routercontract.CanonicalClassifierModule, error) {
	if spec == nil {
		return routercontract.CanonicalClassifierModule{}, nil
	}

	var classifier routercontract.CanonicalClassifierModule

	if spec.CategoryModel != nil {
		domain, err := convertToTypedConfig[routercontract.CanonicalCategoryModule](r, spec.CategoryModel)
		if err != nil {
			return routercontract.CanonicalClassifierModule{}, fmt.Errorf("domain: %w", err)
		}
		classifier.Domain = domain
	}

	if spec.PIIModel != nil {
		pii, err := convertToTypedConfig[routercontract.CanonicalPIIModule](r, spec.PIIModel)
		if err != nil {
			return routercontract.CanonicalClassifierModule{}, fmt.Errorf("pii: %w", err)
		}
		classifier.PII = pii
	}

	return classifier, nil
}
