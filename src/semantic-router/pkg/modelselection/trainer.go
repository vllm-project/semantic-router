/*
Copyright 2025 vLLM Semantic Router.

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

package modelselection

import (
	"crypto/sha256"
	"fmt"
	"math"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// VSR's 14 categories for one-hot encoding
// IMPORTANT: This order MUST match Python training: src/training/ml_model_selection/data_loader.py
var VSRCategories = []string{
	"math",
	"physics",
	"chemistry",
	"biology",
	"computer science",
	"history",
	"economics",
	"business",
	"law",
	"health",
	"psychology",
	"philosophy",
	"other",
	"unknown",
}

// CategoryToIndex maps category name to one-hot index
var CategoryToIndex = func() map[string]int {
	m := make(map[string]int)
	for i, cat := range VSRCategories {
		m[cat] = i
	}
	return m
}()

// AlgorithmHyperparams holds configurable hyperparameters for all algorithms
type AlgorithmHyperparams struct {
	// Global weight for quality vs speed (affects all algorithms)
	// 0.0 = pure speed, 1.0 = pure quality
	QualityWeight float64 // Default: 0.9 (90% quality, 10% speed)

	// KNN hyperparameters
	KnnK int // Number of neighbors (default: 5)

	// KMeans hyperparameters
	KmeansNumClusters int // Number of clusters (default: 8)
}

// DefaultHyperparams returns the default hyperparameters
func DefaultHyperparams() AlgorithmHyperparams {
	return AlgorithmHyperparams{
		QualityWeight:     0.9, // 90% quality, 10% speed (global for all algorithms)
		KnnK:              5,   // 5 neighbors for stable, smooth routing
		KmeansNumClusters: 8,   // 8 clusters for 14 categories
	}
}

// Trainer handles training of model selection algorithms
type Trainer struct {
	// EmbeddingDim is the dimension of query embeddings (e.g., 1024 for Qwen3)
	EmbeddingDim int

	// FeatureDim is the total feature dimension (EmbeddingDim + 14 categories)
	FeatureDim int

	// LLMCandidates contains the LLM configurations
	LLMCandidates *LLMCandidatesConfig

	// RoutingData contains the training records
	RoutingData []RoutingDataRecord

	// EmbeddingModel specifies which model to use: "qwen3", "gemma", or "bert"
	EmbeddingModel string

	// UseCandle indicates whether to use Candle for real embeddings
	// If false, falls back to deterministic hash-based embeddings
	UseCandle bool

	// EmbeddingCache stores embeddings during training (query hash -> embedding)
	// This avoids regenerating the same embedding multiple times
	EmbeddingCache map[string][]float64

	// Hyperparams contains configurable algorithm hyperparameters
	Hyperparams AlgorithmHyperparams
}

// NewTrainer creates a new trainer instance
func NewTrainer(embeddingDim int) *Trainer {
	return &Trainer{
		EmbeddingDim:   embeddingDim,
		FeatureDim:     embeddingDim + len(VSRCategories), // e.g., 1024 + 14 = 1038 for Qwen3
		EmbeddingModel: "qwen3",                           // Default to Qwen3
		UseCandle:      true,                              // Default to using Candle
		EmbeddingCache: make(map[string][]float64),
		Hyperparams:    DefaultHyperparams(),
	}
}

// NewTrainerWithFallback creates a trainer that uses hash-based embeddings (no Candle required)
func NewTrainerWithFallback(embeddingDim int) *Trainer {
	return &Trainer{
		EmbeddingDim:   embeddingDim,
		FeatureDim:     embeddingDim + len(VSRCategories),
		EmbeddingModel: "",
		UseCandle:      false,
		Hyperparams:    DefaultHyperparams(),
		EmbeddingCache: make(map[string][]float64),
	}
}

// SetHyperparams sets custom algorithm hyperparameters
func (t *Trainer) SetHyperparams(params AlgorithmHyperparams) {
	t.Hyperparams = params
}

// CategoryToOneHot converts a category name to a one-hot encoded vector
func CategoryToOneHot(category string) []float64 {
	oneHot := make([]float64, len(VSRCategories))
	if idx, ok := CategoryToIndex[category]; ok {
		oneHot[idx] = 1.0
	} else {
		// Default to "other" if unknown category
		oneHot[CategoryToIndex["other"]] = 1.0
	}
	return oneHot
}

// CombineEmbeddingWithCategory creates the full feature vector
// Feature = [QueryEmbedding (768)] + [CategoryOneHot (14)] = 782 dimensions
func CombineEmbeddingWithCategory(embedding []float64, category string) []float64 {
	categoryOneHot := CategoryToOneHot(category)
	combined := make([]float64, len(embedding)+len(categoryOneHot))
	copy(combined, embedding)
	copy(combined[len(embedding):], categoryOneHot)
	return combined
}

// SetEmbeddingModel sets the embedding model to use (qwen3, gemma, or bert)
func (t *Trainer) SetEmbeddingModel(model string) {
	t.EmbeddingModel = model
}

// SetUseCandle enables or disables Candle for embedding generation
func (t *Trainer) SetUseCandle(useCandle bool) {
	t.UseCandle = useCandle
}

// GetFeatureVector generates the full feature vector for a query (embedding + category one-hot)
// This is used both during training and inference
func (t *Trainer) GetFeatureVector(query, category string) ([]float64, error) {
	embedding, err := t.GetEmbedding(query)
	if err != nil {
		return nil, err
	}
	return CombineEmbeddingWithCategory(embedding, category), nil
}

// GetEmbedding generates an embedding for a query using Candle (Qwen3/Gemma) or fallback
// Uses cache to avoid regenerating embeddings for the same query
func (t *Trainer) GetEmbedding(query string) ([]float64, error) {
	// Create cache key from query hash
	hash := sha256.Sum256([]byte(query))
	cacheKey := fmt.Sprintf("%x", hash[:16])

	// Check cache first
	if embedding, ok := t.EmbeddingCache[cacheKey]; ok {
		return embedding, nil
	}

	// Generate embedding
	var embedding []float64

	if t.UseCandle {
		// Use Candle for real embeddings (same as VSR cache/classification)
		embeddingF32, err := t.generateCandleEmbedding(query)
		if err != nil {
			logging.Warnf("Candle embedding failed, falling back to hash-based: %v", err)
			embedding = t.generateDeterministicEmbedding(query)
		} else {
			// Convert float32 to float64
			embedding = make([]float64, len(embeddingF32))
			for i, v := range embeddingF32 {
				embedding[i] = float64(v)
			}
		}
	} else {
		// Fallback to deterministic hash-based embedding (no randomness)
		embedding = t.generateDeterministicEmbedding(query)
	}

	// Cache the embedding
	t.EmbeddingCache[cacheKey] = embedding
	return embedding, nil
}

// generateCandleEmbedding generates an embedding using Candle (Qwen3/Gemma)
// This is the same approach used by VSR's semantic cache
func (t *Trainer) generateCandleEmbedding(text string) ([]float32, error) {
	switch t.EmbeddingModel {
	case "qwen3":
		// Use GetEmbeddingBatched for Qwen3 with continuous batching
		output, err := candle_binding.GetEmbeddingBatched(text, "qwen3", t.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "gemma":
		// Use GetEmbeddingWithModelType for Gemma
		output, err := candle_binding.GetEmbeddingWithModelType(text, "gemma", t.EmbeddingDim)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	case "bert", "":
		// Use traditional GetEmbedding for BERT (default)
		return candle_binding.GetEmbedding(text, t.EmbeddingDim)
	default:
		return nil, fmt.Errorf("unsupported embedding model: %s (must be 'bert', 'qwen3', or 'gemma')", t.EmbeddingModel)
	}
}

// generateDeterministicEmbedding creates a deterministic embedding from query text
// This is a fallback when no embedding provider is available
// NO random values - purely deterministic based on query content
func (t *Trainer) generateDeterministicEmbedding(query string) []float64 {
	hash := sha256.Sum256([]byte(query))
	embedding := make([]float64, t.EmbeddingDim)

	// Use multiple hash passes for better distribution
	for i := 0; i < t.EmbeddingDim; i++ {
		// Combine multiple hash bytes for each dimension
		byteIdx := i % 32
		nextByteIdx := (i + 1) % 32
		prevByteIdx := (i + 31) % 32

		// Create value from hash bytes
		val := float64(hash[byteIdx]) / 255.0
		val += float64(hash[nextByteIdx]) / 510.0
		val -= float64(hash[prevByteIdx]) / 510.0

		embedding[i] = val
	}

	// Normalize to unit vector
	var norm float64
	for _, v := range embedding {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding
}

// LoadData loads training data from files
func (t *Trainer) LoadData(candidatesPath, routingDataPath string) error {
	// Load LLM candidates
	candidates, err := LoadLLMCandidates(candidatesPath)
	if err != nil {
		return fmt.Errorf("failed to load LLM candidates: %w", err)
	}
	t.LLMCandidates = candidates

	// Load routing data
	routingData, err := LoadRoutingData(routingDataPath)
	if err != nil {
		return fmt.Errorf("failed to load routing data: %w", err)
	}
	t.RoutingData = routingData

	logging.Infof("Loaded %d LLM candidates and %d routing records",
		len(t.LLMCandidates.LLMCandidates), len(t.RoutingData))
	return nil
}

// LoadBenchmarkData loads training data from training_data_with_category.jsonl
// This is the real benchmark data with category classification
func (t *Trainer) LoadBenchmarkData(dataPath string) error {
	return t.LoadBenchmarkDataFiltered(dataPath, nil)
}

// LoadBenchmarkDataFiltered loads training data filtered by specific models
// If allowedModels is nil or empty, loads all models
func (t *Trainer) LoadBenchmarkDataFiltered(dataPath string, allowedModels []string) error {
	// Load and convert benchmark format data with model filter
	recordsByQuery, err := LoadBenchmarkDataFiltered(dataPath, allowedModels)
	if err != nil {
		return fmt.Errorf("failed to load benchmark data: %w", err)
	}

	// Convert to routing data format
	t.RoutingData = ConvertBenchmarkToRoutingData(recordsByQuery, t.Hyperparams.QualityWeight)

	// Create minimal LLM candidates from the data (for latency estimation)
	t.LLMCandidates = t.extractLLMCandidatesFromData(recordsByQuery)

	logging.Infof("Loaded %d unique queries with %d LLM models",
		len(t.RoutingData), len(t.LLMCandidates.LLMCandidates))
	return nil
}

// extractLLMCandidatesFromData builds LLM candidates from the training data
func (t *Trainer) extractLLMCandidatesFromData(recordsByQuery map[int][]BenchmarkRecord) *LLMCandidatesConfig {
	// Collect model stats
	modelLatencies := make(map[string][]float64)
	modelScores := make(map[string][]float64)

	for _, records := range recordsByQuery {
		for _, r := range records {
			modelLatencies[r.ModelName] = append(modelLatencies[r.ModelName], r.ResponseTime)
			modelScores[r.ModelName] = append(modelScores[r.ModelName], r.Performance)
		}
	}

	// Build candidates
	candidates := make(map[string]LLMCandidate)
	for model, latencies := range modelLatencies {
		avgLatency := 0.0
		for _, l := range latencies {
			avgLatency += l
		}
		if len(latencies) > 0 {
			avgLatency /= float64(len(latencies))
		}

		avgScore := 0.0
		scores := modelScores[model]
		for _, s := range scores {
			avgScore += s
		}
		if len(scores) > 0 {
			avgScore /= float64(len(scores))
		}

		candidates[model] = LLMCandidate{
			ModelID:      model,
			DisplayName:  model,
			Provider:     "nvidia",          // From benchmark data
			AvgLatencyMs: avgLatency * 1000, // Convert seconds to ms
			QualityScore: avgScore,
		}
	}

	logging.Infof("Extracted %d LLM candidates from training data", len(candidates))
	return &LLMCandidatesConfig{
		LLMCandidates: candidates,
	}
}

// ConvertToTrainingRecords converts routing data to TrainingRecord format
// Generates embeddings during training + adds category one-hot encoding
// Feature vector = [QueryEmbedding (768)] + [CategoryOneHot (14)] = 782 dimensions
func (t *Trainer) ConvertToTrainingRecords() []TrainingRecord {
	var records []TrainingRecord
	errorCount := 0

	logging.Infof("Converting to training records (queries=%d)", len(t.RoutingData))

	for i, rd := range t.RoutingData {
		category := rd.QueryType // QueryType contains the category

		// Generate embedding for this query (cached automatically)
		embedding, err := t.GetEmbedding(rd.Query)
		if err != nil {
			errorCount++
			if errorCount <= 5 {
				logging.Warnf("Failed to generate embedding for query %d: %v", i, err)
			}
			continue
		}

		// Combine embedding with category one-hot encoding
		// This creates the full feature vector (e.g., 1024 + 14 = 1038 for Qwen3)
		featureVector := CombineEmbeddingWithCategory(embedding, category)

		// Get model scores and find best model
		bestModel := rd.BestModel
		bestScore := 0.0
		if score, ok := rd.ModelScores[bestModel]; ok {
			bestScore = score
		}

		// Get real latency from LLM candidates (extracted from training data)
		latencyMs := 1000.0 // Default only if not found
		if t.LLMCandidates != nil {
			if candidate, ok := t.LLMCandidates.LLMCandidates[bestModel]; ok {
				latencyMs = candidate.AvgLatencyMs
			}
		}

		// Record for the best model - using combined feature vector (embedding + category)
		record := TrainingRecord{
			QueryEmbedding:    featureVector, // Now 782 dimensions (768 + 14)
			SelectedModel:     bestModel,
			ResponseQuality:   bestScore,
			ResponseLatencyNs: int64(time.Duration(latencyMs) * time.Millisecond),
			Success:           true,
			TimestampUnix:     time.Now().Unix(),
		}
		records = append(records, record)

		// Also add records for other models (same query, different model outcomes)
		// This is real data - same feature vector, different model performance
		for model, score := range rd.ModelScores {
			if model == bestModel {
				continue
			}

			modelLatencyMs := 1000.0
			if t.LLMCandidates != nil {
				if candidate, ok := t.LLMCandidates.LLMCandidates[model]; ok {
					modelLatencyMs = candidate.AvgLatencyMs
				}
			}

			// Same feature vector - same query + category, just different model outcome
			altRecord := TrainingRecord{
				QueryEmbedding:    featureVector, // Same feature vector (embedding + category)
				SelectedModel:     model,
				ResponseQuality:   score,
				ResponseLatencyNs: int64(time.Duration(modelLatencyMs) * time.Millisecond),
				Success:           score > 0.5,
				TimestampUnix:     time.Now().Unix(),
			}
			records = append(records, altRecord)
		}

		// Progress logging every 500 queries
		if (i+1)%500 == 0 {
			logging.Infof("Processed %d/%d queries for training records", i+1, len(t.RoutingData))
		}
	}

	if errorCount > 0 {
		logging.Warnf("Failed to generate embeddings for %d queries", errorCount)
	}
	logging.Infof("Generated %d training records from %d routing data entries", len(records), len(t.RoutingData))
	logging.Infof("Embeddings cached: %d unique queries", len(t.EmbeddingCache))
	return records
}

// TrainAllAlgorithms trains all algorithms and saves them
func (t *Trainer) TrainAllAlgorithms(outputPath string) error {
	trainingRecords := t.ConvertToTrainingRecords()

	// Train KNN
	if err := t.trainAndSaveKNN(trainingRecords, outputPath+"/knn_model.json"); err != nil {
		return fmt.Errorf("failed to train KNN: %w", err)
	}

	// Train KMeans
	if err := t.trainAndSaveKMeans(trainingRecords, outputPath+"/kmeans_model.json"); err != nil {
		return fmt.Errorf("failed to train KMeans: %w", err)
	}

	// Train SVM
	if err := t.trainAndSaveSVM(trainingRecords, outputPath+"/svm_model.json"); err != nil {
		return fmt.Errorf("failed to train SVM: %w", err)
	}

	logging.Infof("Successfully trained and saved all algorithms to %s", outputPath)
	return nil
}

func (t *Trainer) trainAndSaveKNN(records []TrainingRecord, path string) error {
	selector := NewKNNSelector(t.Hyperparams.KnnK)
	logging.Infof("Training KNN with k=%d", t.Hyperparams.KnnK)
	if err := selector.Train(records); err != nil {
		return err
	}
	return selector.Save(path)
}

func (t *Trainer) trainAndSaveKMeans(records []TrainingRecord, path string) error {
	selector := NewKMeansSelector(t.Hyperparams.KmeansNumClusters)
	logging.Infof("Training KMeans with clusters=%d", t.Hyperparams.KmeansNumClusters)
	if err := selector.Train(records); err != nil {
		return err
	}
	return selector.Save(path)
}

func (t *Trainer) trainAndSaveSVM(records []TrainingRecord, path string) error {
	selector := NewSVMSelector("rbf")
	if err := selector.Train(records); err != nil {
		return err
	}
	return selector.Save(path)
}

// LoadPretrainedSelector loads a pre-trained selector from file
func LoadPretrainedSelector(algorithm, path string) (Selector, error) {
	switch algorithm {
	case "knn":
		s := NewKNNSelector(5)
		if err := s.Load(path); err != nil {
			return nil, err
		}
		return s, nil

	case "kmeans":
		s := NewKMeansSelector(8)
		if err := s.Load(path); err != nil {
			return nil, err
		}
		return s, nil

	case "svm":
		s := NewSVMSelector("rbf")
		if err := s.Load(path); err != nil {
			return nil, err
		}
		return s, nil

	default:
		return nil, fmt.Errorf("unknown algorithm: %s", algorithm)
	}
}
