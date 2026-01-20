package classification

import (
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// ComplexityResult represents the output of the complexity regressor
type ComplexityResult struct {
	Score float64 `json:"score"`
}

// ComplexityClassifier handles task complexity regression using a Hugging Face model
type ComplexityClassifier struct {
	config      *config.ComplexityModelConfig
	repoID      string
	initialized bool
	mu          sync.RWMutex
}

// NewComplexityClassifier creates a new complexity classifier
func NewComplexityClassifier(cfg *config.ComplexityModelConfig) (*ComplexityClassifier, error) {
	if cfg == nil {
		return nil, nil
	}
	if cfg.ModelURL == "" {
		return nil, fmt.Errorf("complexity classifier requires model_url")
	}

	return &ComplexityClassifier{
		config: cfg,
	}, nil
}

// Initialize prepares the complexity classifier, optionally downloading the model
func (c *ComplexityClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	repoID, err := extractRepoID(c.config.ModelURL)
	if err != nil {
		return err
	}

	c.repoID = repoID

	if c.config.ModelID == "" {
		c.config.ModelID = "models/complexity-regressor"
	}

	if err := modeldownload.CheckHuggingFaceCLI(); err != nil {
		return fmt.Errorf("huggingface-cli not available for complexity model download: %w", err)
	}

	spec := modeldownload.ModelSpec{
		LocalPath:     c.config.ModelID,
		RepoID:        repoID,
		Revision:      "main",
		RequiredFiles: modeldownload.DefaultRequiredFiles,
	}
	if err := modeldownload.DownloadModel(spec, modeldownload.GetDownloadConfig()); err != nil {
		return fmt.Errorf("failed to download complexity model: %w", err)
	}

	if !candle_binding.InitDebertaV3Regressor(c.config.ModelID, c.config.UseCPU) {
		return fmt.Errorf("failed to initialize complexity regressor via Candle")
	}

	c.initialized = true
	logging.Infof("Complexity classifier initialized with HF model: %s", repoID)
	return nil
}

// IsInitialized returns whether the classifier is initialized
func (c *ComplexityClassifier) IsInitialized() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.initialized
}

// Classify returns the complexity score for the given text
func (c *ComplexityClassifier) Classify(text string) (*ComplexityResult, error) {
	c.mu.RLock()
	if !c.initialized {
		c.mu.RUnlock()
		return nil, fmt.Errorf("complexity classifier not initialized")
	}
	c.mu.RUnlock()

	if text == "" {
		return &ComplexityResult{Score: 0}, nil
	}

	start := time.Now()
	score, err := candle_binding.ClassifyDebertaV3RegressionText(text)
	metrics.RecordClassifierLatency("complexity", time.Since(start).Seconds())
	if err != nil {
		return nil, fmt.Errorf("complexity regression failed: %w", err)
	}

	return &ComplexityResult{Score: score}, nil
}

func extractRepoID(modelURL string) (string, error) {
	if strings.HasPrefix(modelURL, "http://") || strings.HasPrefix(modelURL, "https://") {
		parsed, err := url.Parse(modelURL)
		if err != nil {
			return "", fmt.Errorf("invalid model_url: %w", err)
		}
		path := strings.Trim(parsed.Path, "/")
		parts := strings.Split(path, "/")
		if len(parts) >= 3 && parts[0] == "models" {
			parts = parts[1:]
		}
		if len(parts) < 2 {
			return "", fmt.Errorf("model_url does not contain a repo id: %s", modelURL)
		}
		return parts[0] + "/" + parts[1], nil
	}

	trimmed := strings.TrimSpace(modelURL)
	if trimmed == "" {
		return "", fmt.Errorf("model_url is empty")
	}
	if strings.Count(trimmed, "/") < 1 {
		return "", fmt.Errorf("model_url must be a Hugging Face repo id (org/model)")
	}
	return trimmed, nil
}
