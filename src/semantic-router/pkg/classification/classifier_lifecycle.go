package classification

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// BuildClassifier creates a classifier without executing runtime initialization.
// The router assembly path uses this explicit build-then-init split so lifecycle
// ownership stays visible outside constructor call chains.
func BuildClassifier(
	cfg *config.RouterConfig,
	categoryMapping *CategoryMapping,
	piiMapping *PIIMapping,
	jailbreakMapping *JailbreakMapping,
) (*Classifier, error) {
	jailbreakInitializer, jailbreakInference, err := buildJailbreakDependencies(cfg)
	if err != nil {
		return nil, err
	}
	piiInitializer, piiInference := buildPIIDependencies(cfg)
	builder := newClassifierOptionBuilder(cfg, []option{
		withJailbreak(jailbreakMapping, jailbreakInitializer, jailbreakInference),
		withPII(piiMapping, piiInitializer, piiInference),
	})
	options, err := builder.build(categoryMapping)
	if err != nil {
		return nil, err
	}
	return newClassifierWithOptions(cfg, options...)
}

// NewClassifier preserves the legacy convenience behavior for existing callers
// by building the classifier and then explicitly initializing runtime state.
func NewClassifier(
	cfg *config.RouterConfig,
	categoryMapping *CategoryMapping,
	piiMapping *PIIMapping,
	jailbreakMapping *JailbreakMapping,
) (*Classifier, error) {
	classifier, err := BuildClassifier(cfg, categoryMapping, piiMapping, jailbreakMapping)
	if err != nil {
		return nil, err
	}
	if err := classifier.InitializeRuntime(); err != nil {
		return nil, err
	}
	return classifier, nil
}

// InitializeRuntime executes classifier-owned runtime initialization tasks after
// construction. Required and best-effort initializers are explicit task units so
// router assembly can reason about lifecycle instead of relying on constructor
// side effects.
func (c *Classifier) InitializeRuntime() error {
	if c == nil {
		return fmt.Errorf("classifier is nil")
	}

	c.logHeuristicClassifierInitialization()
	tasks := c.runtimeTasks()
	if len(tasks) == 0 {
		return nil
	}

	logging.ComponentEvent("classifier", "runtime_initialization_started", map[string]interface{}{
		"tasks": len(tasks),
	})
	_, err := modelruntime.Execute(context.Background(), tasks, modelruntime.Options{
		MaxParallelism: modelruntime.DefaultParallelism(len(tasks)),
		OnEvent:        logRuntimeInitializationEvent,
	})
	if err != nil {
		return err
	}

	logging.ComponentEvent("classifier", "runtime_initialization_completed", map[string]interface{}{
		"tasks": len(tasks),
	})
	return nil
}

func (c *Classifier) runtimeTasks() []modelruntime.Task {
	tasks := make([]modelruntime.Task, 0, 9)
	appendTask := func(name string, bestEffort bool, enabled bool, init func() error) {
		if !enabled {
			return
		}
		tasks = append(tasks, modelruntime.Task{
			Name:       name,
			BestEffort: bestEffort,
			Run: func(context.Context) error {
				return init()
			},
		})
	}

	appendTask("classifier.category", false, c.usesRoutingSignalType(config.SignalTypeDomain) && (c.IsCategoryEnabled() || c.IsMCPCategoryEnabled()), c.initializeConfiguredCategoryRuntime)
	appendTask("classifier.jailbreak", false, c.usesRoutingSignalType(config.SignalTypeJailbreak) && c.IsJailbreakEnabled(), c.initializeJailbreakClassifier)
	appendTask("classifier.pii", false, c.usesRoutingSignalType(config.SignalTypePII) && c.IsPIIEnabled(), c.initializePIIClassifier)
	appendTask("classifier.keyword_embedding", false, c.IsKeywordEmbeddingClassifierEnabled(), c.initializeKeywordEmbeddingClassifier)
	appendTask("classifier.fact_check", true, c.IsFactCheckEnabled(), c.initializeFactCheckClassifier)
	appendTask("classifier.hallucination", true, c.IsHallucinationDetectionEnabled(), c.initializeHallucinationDetector)
	appendTask("classifier.feedback", true, c.IsFeedbackDetectorEnabled(), c.initializeFeedbackDetector)
	appendTask("classifier.preference", true, c.IsPreferenceClassifierEnabled(), c.initializePreferenceClassifier)
	appendTask("classifier.language", true, len(c.Config.LanguageRules) > 0, c.initializeLanguageClassifier)

	return tasks
}

func (c *Classifier) usesRoutingSignalType(signalType string) bool {
	return c != nil && c.Config != nil && c.Config.UsesSignalTypeInRouting(signalType)
}

func (c *Classifier) initializeConfiguredCategoryRuntime() error {
	if c.IsCategoryEnabled() {
		return c.initializeCategoryClassifier()
	}
	if c.IsMCPCategoryEnabled() {
		return c.initializeMCPCategoryClassifier()
	}
	return nil
}

func logRuntimeInitializationEvent(event modelruntime.Event) {
	payload := map[string]interface{}{
		"task":        event.Task,
		"best_effort": event.BestEffort,
	}
	if event.Error != nil {
		payload["error"] = event.Error.Error()
	}

	switch event.Status {
	case modelruntime.TaskFailed:
		if event.BestEffort {
			logging.ComponentWarnEvent("classifier", "runtime_initializer_failed", payload)
			return
		}
		logging.ComponentErrorEvent("classifier", "runtime_initializer_failed", payload)
	case modelruntime.TaskSkipped:
		logging.ComponentWarnEvent("classifier", "runtime_initializer_skipped", payload)
	}
}
