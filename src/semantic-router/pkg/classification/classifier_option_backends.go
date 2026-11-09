package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (b *classifierOptionBuilder) addCategoryClassifier(categoryMapping *CategoryMapping) error {
	models := b.runtime()
	spec, ok, err := models.task("domain_classifier")
	if err != nil && !config.IsDeferredModelError(err) {
		return err
	}
	if err == nil && !ok {
		return nil
	}
	if err == nil && remoteTask(spec) {
		return b.addRemoteCategoryClassifier(models, spec, categoryMapping)
	}
	return b.addLocalCategoryClassifier(categoryMapping)
}

func (b *classifierOptionBuilder) addRemoteCategoryClassifier(models *classifierModelRuntime, spec config.ResolvedModelBinding, categoryMapping *CategoryMapping) error {
	if spec.Binding.Adapter != config.RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf("category deployment %q: the domain signal reads the classify API", spec.Binding.Deployment)
	}
	transport, err := newHTTPClassifierInference(spec.Deployment, categoryMapping)
	if err != nil {
		return err
	}
	owned, err := prepareRemoteSequence(models, spec, transport)
	if err != nil {
		return err
	}
	backend := &categoryHTTPBackend{backend: owned}
	b.options = append(b.options, withCategory(categoryMapping, nil, backend))
	return nil
}

func (b *classifierOptionBuilder) addLocalCategoryClassifier(categoryMapping *CategoryMapping) error {
	models := b.runtime()
	spec, err := models.localSpec("domain_classifier", "auto", config.RemoteClassifierContractLabelDistribution, b.cfg.CategoryModel.MaxSequenceLength)
	var labels []string
	if categoryMapping != nil {
		labels = indexedNativeLabels(categoryMapping.IdxToCategory)
	}
	backend := ownedCategoryBackend{&ownedSequenceBackend{runtime: models.runtime, spec: spec, err: err, labels: labels}}
	b.options = append(b.options, withCategory(categoryMapping, backend, backend))
	return nil
}

func (b *classifierOptionBuilder) addMCPCategoryClassifier() {
	if !b.cfg.MCPCategoryModel.Enabled {
		return
	}
	mcpInit := createMCPCategoryInitializer()
	mcpInf := createMCPCategoryInference(mcpInit)
	b.options = append(b.options, withMCPCategory(mcpInit, mcpInf))
}

func buildJailbreakDependencies(cfg *config.RouterConfig, jailbreakMapping *JailbreakMapping, models ...*classifierModelRuntime) (JailbreakInitializer, SequenceClassifierBackend, error) {
	if cfg.PromptGuard.Window != nil {
		if jailbreakMapping == nil {
			// No reachable model consumer loaded a mapping for this recipe.
			return nil, nil, nil
		}
		backend, err := newWindowedJailbreakBackend(cfg.PromptGuard, jailbreakMapping, models...)
		return backend, backend, err
	}
	if remoteModule(cfg, "prompt_guard") {
		// Remote backends have no local model to initialize.
		jailbreakInference, err := createJailbreakInference(&cfg.PromptGuard, cfg, jailbreakMapping, models...)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to create jailbreak inference: %w", err)
		}
		return nil, jailbreakInference, nil
	}
	runtime := consumerModelRuntime(models)
	spec, err := runtime.localSpec("prompt_guard", "auto", config.RemoteClassifierContractLabelDistribution, cfg.PromptGuard.MaxSequenceLength)
	var labels []string
	if jailbreakMapping != nil {
		labels = indexedNativeLabels(jailbreakMapping.IdxToLabel)
	}
	backend := &ownedSequenceBackend{runtime: runtime.runtime, spec: spec, err: err, labels: labels}
	return backend, backend, nil
}

func buildPIIDependencies(cfg *config.RouterConfig, piiMapping *PIIMapping, models ...*classifierModelRuntime) (PIIInitializer, PIIInference, error) {
	remote := remoteModule(cfg, "pii_classifier")
	if cfg.NeedsPIIMappingForRouting() && !remote && cfg.PIIModel.Window == nil {
		backend, err := prepareDecisionPII(consumerModelRuntime(models))
		if err != nil {
			return nil, nil, err
		}
		if backend != nil {
			return backend, backend, nil
		}
	}
	if cfg.PIIModel.Window != nil {
		backend, err := newWindowedPIIBackend(cfg.PIIModel, piiMapping, models...)
		if err != nil {
			return nil, nil, err
		}
		return backend, backend, nil
	}
	if remote {
		if piiMapping == nil {
			// The mapping loader is skipped on purpose when no reachable routing
			// decision consumes the PII signal. With no consumer there is nothing
			// to build: IsPIIEnabled stays false and a dormant remote backend must
			// not fail startup on the missing mapping.
			logging.ComponentEvent("classifier", "pii_detector_backend_dormant", map[string]interface{}{
				"backend": "token_spans_http",
				"reason":  "no_reachable_pii_signal",
			})
			return nil, nil, nil
		}
		// Remote inference is fully constructed here and has no local model
		// lifecycle, so the initializer is nil, as it is for a remote category
		// backend.
		modelRuntime := consumerModelRuntime(models)
		spec, _, err := modelRuntime.task("pii_classifier")
		if err != nil {
			return nil, nil, err
		}
		if spec.Binding.Adapter != config.RemoteClassifierProtocolHTTPClassify {
			return nil, nil, fmt.Errorf("PII deployment %q: the pii signal reads the classify API", spec.Binding.Deployment)
		}
		transport, err := newPIIHTTPTokenClassifierInference(spec.Deployment, piiMapping)
		if err != nil {
			return nil, nil, err
		}
		inference, err := prepareRemoteTokens(modelRuntime, spec, transport)
		if err != nil {
			return nil, nil, err
		}
		logging.ComponentEvent("classifier", "pii_detector_backend_selected", map[string]interface{}{
			"backend": "token_spans_http",
		})
		return nil, inference, nil
	}
	runtime := consumerModelRuntime(models)
	spec, err := runtime.localSpec("pii_classifier", "auto", config.RemoteClassifierContractTokenSpans, cfg.PIIModel.MaxSequenceLength)
	var labels []string
	if piiMapping != nil {
		labels = indexedNativeLabels(piiMapping.IdxToLabel)
	}
	backend := &ownedTokenBackend{runtime: runtime.runtime, spec: spec, err: err, labels: labels}
	return backend, backend, nil
}

// addComplexityBackend attaches the complexity signal's remote scorer, if one
// is configured. A nil backend leaves the local prototype path in place, so
// this is a no-op for every existing config.
//
// The contract decides which reader is built: score.v1 needs no label mapping
// and turns its number into a verdict through each rule's boundaries, while
// label_distribution.v1 reuses the shared sequence backend with the verdict
// vocabulary declared inline, exactly as the generic classifier signal does.
func (b *classifierOptionBuilder) addComplexityBackend() error {
	// The same validator globalConfigContractValidators runs at config load, so
	// a directly-built classifier cannot bypass the boundary checks either.
	if err := config.ValidateComplexityModelBackend(b.cfg); err != nil {
		return err
	}
	models := b.runtime()
	remote, ok, err := models.cfg.ComplexityRemoteTask()
	if err != nil || !ok {
		return err
	}
	spec, _, err := models.task(remote.Consumer)
	if err != nil {
		return err
	}
	switch spec.Binding.Contract {
	case config.RemoteClassifierContractScore:
		scorer, err := newScoringHTTPBackend(spec.Deployment)
		if err != nil {
			return err
		}
		logging.ComponentEvent("classifier", "complexity_backend_selected", map[string]interface{}{
			"contract": config.RemoteClassifierContractScore,
		})
		owned, err := prepareRemoteScore(models, spec, scorer)
		if err != nil {
			return err
		}
		b.options = append(b.options, withComplexityScoreBackend(owned))
	case config.RemoteClassifierContractLabelDistribution:
		labels, err := newHTTPClassifierInference(spec.Deployment, newDeclaredLabelMapping(ComplexityVerdictLabels))
		if err != nil {
			return err
		}
		logging.ComponentEvent("classifier", "complexity_backend_selected", map[string]interface{}{
			"contract": config.RemoteClassifierContractLabelDistribution,
		})
		owned, err := prepareRemoteSequence(models, spec, labels)
		if err != nil {
			return err
		}
		b.options = append(b.options, withComplexityLabelBackend(owned))
	default:
		// Unreachable: the binding plan rejects anything else. Kept so a
		// future contract cannot be silently ignored here.
		return fmt.Errorf("complexity backend contract %q has no reader", spec.Binding.Contract)
	}
	return nil
}
