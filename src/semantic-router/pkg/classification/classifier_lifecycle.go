package classification

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// BuildClassifier creates a classifier without executing runtime initialization.
// The router assembly path uses this explicit build-then-init split so lifecycle
// ownership stays visible outside constructor call chains.
func BuildClassifier(cfg *config.RouterConfig) (*Classifier, error) {
	return buildClassifier(cfg)
}

func buildClassifier(cfg *config.RouterConfig, runtimeOptions ...RecipeRuntimeOptions) (*Classifier, error) {
	if cfg != nil && cfg.RoutingScope == "" {
		cfg = cfg.ConfigForRecipe(cfg.DefaultRecipe())
	}
	var runtimeOption RecipeRuntimeOptions
	if len(runtimeOptions) > 0 {
		runtimeOption = runtimeOptions[0]
	}
	models, err := newClassifierModelRuntime(cfg, runtimeOption)
	if err != nil {
		return nil, err
	}
	cfg = models.cfg
	builder := newClassifierOptionBuilder(cfg, nil)
	transferred := false
	defer func() {
		if !transferred {
			builder.closePending()
		}
	}()
	builder.models = models
	if cfg.RoutingScope == "" || cfg.RoutingScope == config.DefaultRecipeName {
		builder.embeddingSet = runtimeOption.Embeddings
	}
	options, err := builder.build()
	if err != nil {
		return nil, err
	}
	classifier, err := newClassifierWithOptions(cfg, options...)
	if err != nil {
		return nil, err
	}
	classifier.models = models
	classifier.embeddingSet = builder.embeddingSet
	classifier.ownsEmbeddingSet = builder.ownsEmbeddingSet
	transferred = true
	return classifier, nil
}

// NewClassifier builds the classifier and then initializes its runtime state;
// a classifier that fails to initialize is closed.
func NewClassifier(cfg *config.RouterConfig, runtimeOptions ...RecipeRuntimeOptions) (*Classifier, error) {
	classifier, err := buildClassifier(cfg, runtimeOptions...)
	if err != nil {
		return nil, err
	}
	if err := classifier.InitializeRuntime(); err != nil {
		_ = classifier.Close()
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
	return c.executeRuntimeTasks(c.runtimeTasks())
}

func (c *Classifier) executeRuntimeTasks(tasks []modelruntime.Task) error {
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
		if closeErr := c.Close(); closeErr != nil {
			logging.ComponentWarnEvent("classifier", "runtime_initialization_rollback_failed", map[string]interface{}{
				"error": closeErr.Error(),
			})
		}
		return err
	}

	logging.ComponentEvent("classifier", "runtime_initialization_completed", map[string]interface{}{
		"tasks": len(tasks),
	})
	return nil
}

// Close releases the classifier's runtime resources.
func (c *Classifier) Close() error {
	if c == nil {
		return nil
	}
	c.closeOnce.Do(func() { c.closeErr = c.closeResources() })
	return c.closeErr
}

func (c *Classifier) closeResources() error {
	var closeErrors []error
	closeResource := func(name string, resource interface{}) {
		closer, ok := resource.(interface{ Close() error })
		if !ok || closer == nil {
			return
		}
		if err := closer.Close(); err != nil {
			closeErrors = append(closeErrors, fmt.Errorf("close %s: %w", name, err))
		}
	}

	if c.ownsEmbeddingSet {
		closeResource("embeddings", c.embeddingSet)
	}
	return errors.Join(closeErrors...)
}

func (c *Classifier) runtimeTasks() []modelruntime.Task {
	tasks := make([]modelruntime.Task, 0, 5)
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

	appendTask("classifier.systemone", false, len(c.preparedSystemOneRules()) > 0, c.prepareSystemOneSignals)
	appendTask("classifier.language", true, len(c.Config.LanguageRules) > 0, c.initializeLanguageClassifier)

	return tasks
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
