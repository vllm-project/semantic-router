package classification

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// getEmbeddingWithModelType is a package-level variable for computing single embeddings.
// It exists so tests can override it.
var getEmbeddingWithModelType = candle_binding.GetEmbeddingWithModelType

// getMultiModalTextEmbedding computes a text embedding via the multimodal model.
// Package-level var so tests can override it.
var getMultiModalTextEmbedding = func(text string, targetDim int) ([]float32, error) {
	output, err := candle_binding.MultiModalEncodeText(text, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// getMultiModalImageEmbedding computes an image embedding from a base64-encoded
// image (raw base64 or data-URI) via the multimodal model.
// Also supports local file paths for preloading knowledge-base image candidates.
// Package-level var so tests can override it.
var getMultiModalImageEmbedding = func(imageRef string, targetDim int) ([]float32, error) {
	if imageRef == "" {
		return nil, fmt.Errorf("imageRef cannot be empty")
	}

	payload := imageRef

	if strings.HasPrefix(imageRef, "/") || strings.HasPrefix(imageRef, "./") {
		data, err := os.ReadFile(imageRef)
		if err != nil {
			return nil, fmt.Errorf("failed to read image file %q: %w", imageRef, err)
		}
		payload = base64.StdEncoding.EncodeToString(data)
	} else if idx := strings.Index(imageRef, ";base64,"); idx >= 0 {
		payload = imageRef[idx+len(";base64,"):]
	}

	output, err := candle_binding.MultiModalEncodeImageFromBase64(payload, targetDim)
	if err != nil {
		return nil, err
	}
	return output.Embedding, nil
}

// initMultiModalModel is a package-level var for initializing the multimodal model.
var initMultiModalModel = candle_binding.InitMultiModalEmbeddingModel

// EmbeddingClassifierInitializer initializes KeywordEmbeddingClassifier for embedding based classification
type EmbeddingClassifierInitializer interface {
	Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool, backend string, modelType string) error
}

type ExternalModelBasedEmbeddingInitializer struct{}

func (c *ExternalModelBasedEmbeddingInitializer) Init(qwen3ModelPath string, gemmaModelPath string, mmBertModelPath string, useCPU bool, backend string, modelType string) error {
	qwen3ModelPath = config.ResolveModelPath(qwen3ModelPath)
	gemmaModelPath = config.ResolveModelPath(gemmaModelPath)
	mmBertModelPath = config.ResolveModelPath(mmBertModelPath)

	backend = strings.ToLower(strings.TrimSpace(backend))
	if backend == "" {
		backend = "candle"
	}

	switch backend {
	case config.EmbeddingBackendOpenAICompatible:
		logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
			"backend":    config.EmbeddingBackendOpenAICompatible,
			"model_type": modelType,
		})
		return nil
	case "openvino":
		if err := initOpenVINOModel(modelType, mmBertModelPath, qwen3ModelPath, useCPU); err != nil {
			return err
		}
		logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
			"backend":          "openvino",
			"model_type":       modelType,
			"mmbert_model_ref": mmBertModelPath,
			"qwen3_model_ref":  qwen3ModelPath,
			"use_cpu":          useCPU,
		})
		return nil
	case "candle":
		err := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, mmBertModelPath, useCPU)
		if err != nil {
			return err
		}
		logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
			"backend":           "candle",
			"qwen3_model_ref":   qwen3ModelPath,
			"gemma_model_ref":   gemmaModelPath,
			"mmbert_model_ref":  mmBertModelPath,
			"use_cpu":           useCPU,
			"mmbert_2d_enabled": mmBertModelPath != "",
		})
		return nil
	default:
		return fmt.Errorf("unsupported embedding backend %q", backend)
	}
}

// createEmbeddingInitializer creates the appropriate keyword embedding initializer based on configuration.
func createEmbeddingInitializer() EmbeddingClassifierInitializer {
	return &ExternalModelBasedEmbeddingInitializer{}
}

// IsKeywordEmbeddingClassifierEnabled checks if keyword embedding classification rules are configured.
func (c *Classifier) IsKeywordEmbeddingClassifierEnabled() bool {
	return len(c.Config.EmbeddingRules) > 0
}

// initializeKeywordEmbeddingClassifier initializes the keyword-embedding classification model.
func (c *Classifier) initializeKeywordEmbeddingClassifier() error {
	if !c.IsKeywordEmbeddingClassifierEnabled() || c.keywordEmbeddingInitializer == nil || c.keywordEmbeddingClassifier == nil {
		return fmt.Errorf("keyword embedding similarity match is not properly configured")
	}

	modelType := strings.ToLower(strings.TrimSpace(c.Config.EmbeddingConfig.ModelType))
	if modelType == "multimodal" {
		mmPath := config.ResolveModelPath(c.Config.MultiModalModelPath)
		if mmPath == "" {
			return fmt.Errorf("embedding_rules with model_type=multimodal requires embedding_models.multimodal_model_path")
		}
		if err := initMultiModalModel(mmPath, c.Config.UseCPU); err != nil {
			return fmt.Errorf("failed to initialize multimodal model for embedding_rules: %w", err)
		}
		logging.ComponentEvent("classifier", "keyword_embedding_backend_initialized", map[string]interface{}{
			"backend":   "multimodal",
			"model_ref": mmPath,
			"use_cpu":   c.Config.UseCPU,
		})
		return c.keywordEmbeddingClassifier.WarmupCandidateEmbeddings()
	}

	if err := c.keywordEmbeddingInitializer.Init(
		c.Config.Qwen3ModelPath,
		c.Config.GemmaModelPath,
		c.Config.MmBertModelPath,
		c.Config.UseCPU,
		c.Config.EmbeddingConfig.Backend,
		c.Config.EmbeddingConfig.ModelType,
	); err != nil {
		return err
	}
	return c.keywordEmbeddingClassifier.WarmupCandidateEmbeddings()
}

func (c *EmbeddingClassifier) getBackend() string {
	if backend := embeddingBackendOverride(); backend != "" {
		logging.Infof("Embedding backend override from env: %s", backend)
		return backend
	}
	if c.backend == "" {
		return "candle"
	}
	return c.backend
}

func (c *EmbeddingClassifier) computeEmbedding(text string, modelType string, phases ...string) ([]float32, error) {
	backend := c.getBackend()
	start := time.Now()
	var embedding []float32
	var err error

	switch backend {
	case config.EmbeddingBackendOpenAICompatible:
		if c.provider == nil {
			return nil, fmt.Errorf("embedding provider is required for backend %q", backend)
		}
		embedding, err = c.provider.Embed(context.Background(), text)
	case "openvino":
		embedding, err = getOpenVINOEmbedding(modelType, text, c.optimizationConfig.TargetDimension)
	case "candle":
		var output *candle_binding.EmbeddingOutput
		output, err = getEmbeddingWithModelType(text, modelType, c.optimizationConfig.TargetDimension)
		if err == nil {
			embedding = output.Embedding
		}
	default:
		return nil, fmt.Errorf("unsupported embedding backend %q", backend)
	}

	elapsed := time.Since(start)
	dim := 0
	if embedding != nil {
		dim = len(embedding)
	}
	phase := "request"
	if len(phases) > 0 {
		phase = phases[0]
	}
	logging.Infof("[Perf] embedding inference (phase=%s, backend=%s, model=%s, dim=%d): %.3fms",
		phase, backend, modelType, dim, float64(elapsed.Microseconds())/1000.0)

	return embedding, err
}
