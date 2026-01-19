package classification

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

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
	client      *http.Client
	endpointURL string
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
		client: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

// Initialize prepares the complexity classifier, optionally downloading the model
func (c *ComplexityClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	repoID, endpoint, err := resolveHFEndpoint(c.config.ModelURL)
	if err != nil {
		return err
	}

	c.repoID = repoID
	c.endpointURL = endpoint

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
	endpoint := c.endpointURL
	c.mu.RUnlock()

	if text == "" {
		return &ComplexityResult{Score: 0}, nil
	}

	payload := map[string]string{"inputs": text}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal complexity request: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create complexity request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if token := os.Getenv("HF_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	start := time.Now()
	resp, err := c.client.Do(req)
	metrics.RecordClassifierLatency("complexity", time.Since(start).Seconds())
	if err != nil {
		return nil, fmt.Errorf("complexity inference request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("complexity inference request returned status %d", resp.StatusCode)
	}

	var decoded any
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("failed to decode complexity response: %w", err)
	}

	score, ok := extractComplexityScore(decoded)
	if !ok {
		return nil, fmt.Errorf("complexity response did not contain a score")
	}

	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return &ComplexityResult{Score: score}, nil
}

func resolveHFEndpoint(modelURL string) (string, string, error) {
	if strings.Contains(modelURL, "api-inference.huggingface.co/models/") {
		repoID, err := extractRepoID(modelURL)
		if err != nil {
			return "", "", err
		}
		return repoID, modelURL, nil
	}

	repoID, err := extractRepoID(modelURL)
	if err != nil {
		return "", "", err
	}

	return repoID, "https://api-inference.huggingface.co/models/" + repoID, nil
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

func extractComplexityScore(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case map[string]any:
		if score, ok := extractScoreFromMap(v); ok {
			return score, true
		}
	case []any:
		for _, item := range v {
			if score, ok := extractComplexityScore(item); ok {
				return score, true
			}
		}
	}
	return 0, false
}

func extractScoreFromMap(data map[string]any) (float64, bool) {
	candidates := []string{"score", "complexity", "value", "output"}
	for _, key := range candidates {
		if raw, ok := data[key]; ok {
			if score, ok := raw.(float64); ok {
				return score, true
			}
		}
	}
	if raw, ok := data["scores"]; ok {
		if scores, ok := raw.([]any); ok && len(scores) > 0 {
			if score, ok := scores[0].(float64); ok {
				return score, true
			}
		}
	}
	return 0, false
}
