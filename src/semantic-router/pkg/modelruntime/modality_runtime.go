package modelruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func modalityClassifierTask(
	cfg *config.RouterConfig,
	component string,
	initFunc func(modelPath string, useCPU bool) error,
) []Task {
	md := &cfg.ModalityDetector
	if !md.Enabled {
		return nil
	}

	method := md.GetMethod()
	if method != config.ModalityDetectionClassifier && method != config.ModalityDetectionHybrid {
		return nil
	}
	if md.Classifier == nil || md.Classifier.ModelPath == "" {
		return nil
	}

	modelPath := config.ResolveModelPath(md.Classifier.ModelPath)
	bestEffort := method == config.ModalityDetectionHybrid
	return []Task{{
		Name:       "router.modality.classifier",
		BestEffort: bestEffort,
		Run: func(context.Context) error {
			logging.ComponentEvent(component, "modality_classifier_init_started", map[string]interface{}{
				"method":    method,
				"model_ref": modelPath,
				"use_cpu":   md.Classifier.UseCPU,
			})
			if initFunc == nil {
				return fmt.Errorf("modality classifier initializer is not configured")
			}
			if err := initFunc(modelPath, md.Classifier.UseCPU); err != nil {
				event := map[string]interface{}{
					"method":    method,
					"model_ref": modelPath,
					"error":     err.Error(),
				}
				if bestEffort {
					event["fallback_to_keywords"] = true
					logging.ComponentWarnEvent(component, "modality_classifier_init_failed", event)
				} else {
					logging.ComponentErrorEvent(component, "modality_classifier_init_failed", event)
				}
				return fmt.Errorf("failed to initialize modality classifier: %w", err)
			}
			logging.ComponentEvent(component, "modality_classifier_initialized", map[string]interface{}{
				"method": method,
			})
			return nil
		},
	}}
}
