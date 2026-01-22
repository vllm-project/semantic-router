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

// Package modelselection provides ML-based model selection algorithms
// for choosing the optimal model from a set of candidates.
//
// This package uses gonum for numerical operations to ensure
// production-quality performance and numerical stability.
package modelselection

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"

	ml_binding "github.com/vllm-project/semantic-router/ml-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// UseLinfa enables Rust/Linfa implementations for KNN, KMeans, SVM.
// When true, uses ml-binding (faster, battle-tested Linfa algorithms).
// When false, uses pure Go implementations (no Rust dependency).
// MLP and Matrix Factorization always use Go (not available in Linfa).
var UseLinfa = false

// Selector interface for all model selection algorithms
type Selector interface {
	// Select chooses the best model from refs based on the selection context
	Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error)

	// Name returns the algorithm name
	Name() string

	// Train updates the model with new training data (for learning-based algorithms)
	Train(data []TrainingRecord) error
}

// SelectionContext contains information for model selection
type SelectionContext struct {
	// QueryEmbedding is the embedding vector of the user query
	QueryEmbedding []float64

	// QueryText is the raw user query text
	QueryText string

	// CategoryName is the detected category/domain
	CategoryName string

	// DecisionName is the matched decision name
	DecisionName string

	// RequestMetadata contains additional request information
	RequestMetadata *RequestMetadata
}

// RequestMetadata contains metadata about the request
type RequestMetadata struct {
	// EstimatedTokens is the estimated input token count
	EstimatedTokens int

	// MaxOutputTokens is the requested max output tokens
	MaxOutputTokens int

	// HasTools indicates if the request includes tool definitions
	HasTools bool

	// StreamingEnabled indicates if streaming is requested
	StreamingEnabled bool

	// Timestamp is when the request was received
	Timestamp time.Time
}

// TrainingRecord represents a historical request-response pair for training
type TrainingRecord struct {
	// QueryEmbedding is the embedding of the query
	QueryEmbedding []float64 `json:"query_embedding"`

	// SelectedModel is the model that was selected
	SelectedModel string `json:"selected_model"`

	// ResponseLatencyNs is how long the response took in nanoseconds
	ResponseLatencyNs int64 `json:"response_latency_ns"`

	// ResponseQuality is a quality score (0-1)
	ResponseQuality float64 `json:"response_quality"`

	// Success indicates if the request was successful
	Success bool `json:"success"`

	// TimestampUnix is when this record was created (Unix timestamp)
	TimestampUnix int64 `json:"timestamp"`
}

// ResponseLatency returns the response latency as time.Duration
func (r TrainingRecord) ResponseLatency() time.Duration {
	return time.Duration(r.ResponseLatencyNs)
}

// Timestamp returns the timestamp as time.Time
func (r TrainingRecord) Timestamp() time.Time {
	return time.Unix(r.TimestampUnix, 0)
}

// ModelStats tracks performance statistics for a model
type ModelStats struct {
	// ModelName is the model identifier
	ModelName string

	// AverageLatency in milliseconds
	AverageLatency float64

	// SuccessRate is the success rate (0-1)
	SuccessRate float64

	// QualityScore is the average quality score (0-1)
	QualityScore float64

	// RequestCount is the total number of requests
	RequestCount int64

	// LastUpdated is when stats were last updated
	LastUpdated time.Time
}

// StatsTracker tracks model performance statistics (thread-safe)
type StatsTracker struct {
	mu    sync.RWMutex
	stats map[string]*ModelStats
}

// NewStatsTracker creates a new stats tracker
func NewStatsTracker() *StatsTracker {
	return &StatsTracker{
		stats: make(map[string]*ModelStats),
	}
}

// GetStats returns stats for a model
func (t *StatsTracker) GetStats(modelName string) *ModelStats {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if stats, ok := t.stats[modelName]; ok {
		statsCopy := *stats
		return &statsCopy
	}
	return nil
}

// UpdateStats updates stats for a model (thread-safe)
func (t *StatsTracker) UpdateStats(modelName string, latency time.Duration, quality float64, success bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	stats, ok := t.stats[modelName]
	if !ok {
		stats = &ModelStats{
			ModelName:    modelName,
			SuccessRate:  1.0,
			QualityScore: quality,
		}
		t.stats[modelName] = stats
	}

	// Update running averages using Welford's algorithm for numerical stability
	stats.RequestCount++
	n := float64(stats.RequestCount)

	// Update average latency
	delta := float64(latency.Milliseconds()) - stats.AverageLatency
	stats.AverageLatency += delta / n

	// Update success rate
	successVal := 0.0
	if success {
		successVal = 1.0
	}
	deltaSuccess := successVal - stats.SuccessRate
	stats.SuccessRate += deltaSuccess / n

	// Update quality score
	deltaQuality := quality - stats.QualityScore
	stats.QualityScore += deltaQuality / n

	stats.LastUpdated = time.Now()
}

// GetAllStats returns all model stats (thread-safe copy)
func (t *StatsTracker) GetAllStats() map[string]*ModelStats {
	t.mu.RLock()
	defer t.mu.RUnlock()

	result := make(map[string]*ModelStats, len(t.stats))
	for k, v := range t.stats {
		statsCopy := *v
		result[k] = &statsCopy
	}
	return result
}

// NewSelector creates a new selector based on the configuration
// If ModelsPath is specified, loads pre-trained models from disk
func NewSelector(cfg *config.MLModelSelectionConfig) (Selector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("model selection config is nil")
	}

	// If ModelsPath is specified, try to load pre-trained model
	if cfg.ModelsPath != "" {
		return loadPretrainedSelectorFromPath(cfg.Type, cfg.ModelsPath)
	}

	// Otherwise create a new empty selector (for training mode)
	return NewEmptySelector(cfg)
}

// loadPretrainedSelectorFromPath loads a pre-trained selector from the specified path
// This is an internal helper to avoid collision with trainer.LoadPretrainedSelector
func loadPretrainedSelectorFromPath(algorithmType, modelsPath string) (Selector, error) {
	// Normalize algorithm name
	algName := algorithmType
	if algName == "matrix_factorization" {
		algName = "mf"
	}

	// Construct model file path
	modelPath := modelsPath + "/" + algName + "_model.json"

	logging.Infof("Loading pre-trained %s selector from %s", algorithmType, modelPath)

	// Load the model file
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load pre-trained model %s: %w", modelPath, err)
	}

	// Parse based on algorithm type
	switch algorithmType {
	case "knn":
		selector := NewKNNSelector(3)
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse KNN model: %w", err)
		}
		logging.Infof("Loaded KNN selector with %d training records", selector.getTrainingCount())
		return selector, nil

	case "kmeans":
		selector := NewKMeansSelector(8)
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse KMeans model: %w", err)
		}
		logging.Infof("Loaded KMeans selector with %d training records", selector.getTrainingCount())
		return selector, nil

	case "mlp":
		selector := NewMLPSelector([]int{64, 32})
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse MLP model: %w", err)
		}
		logging.Infof("Loaded MLP selector with %d training records", selector.getTrainingCount())
		return selector, nil

	case "svm":
		selector := NewSVMSelector("rbf")
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse SVM model: %w", err)
		}
		logging.Infof("Loaded SVM selector with %d training records", selector.getTrainingCount())
		return selector, nil

	case "matrix_factorization", "mf":
		selector := NewMatrixFactorizationSelector(10)
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse MatrixFactorization model: %w", err)
		}
		logging.Infof("Loaded MatrixFactorization selector with %d training records", selector.getTrainingCount())
		return selector, nil

	default:
		return nil, fmt.Errorf("unknown algorithm type: %s", algorithmType)
	}
}

// NewEmptySelector creates a new empty selector for training mode
func NewEmptySelector(cfg *config.MLModelSelectionConfig) (Selector, error) {
	switch cfg.Type {
	case "knn":
		k := cfg.K
		if k <= 0 {
			k = 3 // default
		}
		return NewKNNSelector(k), nil

	case "kmeans":
		numClusters := cfg.NumClusters
		if numClusters <= 0 {
			numClusters = 0 // will be set to number of models
		}
		// Use pointer to distinguish "not set" (nil) from "explicitly 0"
		if cfg.EfficiencyWeight != nil {
			return NewKMeansSelectorWithEfficiency(numClusters, *cfg.EfficiencyWeight), nil
		}
		return NewKMeansSelector(numClusters), nil // Uses default 0.3

	case "mlp":
		hiddenLayers := cfg.HiddenLayers
		if len(hiddenLayers) == 0 {
			hiddenLayers = []int{64, 32} // default
		}
		return NewMLPSelector(hiddenLayers), nil

	case "svm":
		kernel := cfg.Kernel
		if kernel == "" {
			kernel = "rbf" // default
		}
		return NewSVMSelector(kernel), nil

	case "matrix_factorization":
		numFactors := cfg.NumFactors
		if numFactors <= 0 {
			numFactors = 10 // default
		}
		return NewMatrixFactorizationSelector(numFactors), nil

	default:
		return nil, fmt.Errorf("unknown model selection algorithm: %s", cfg.Type)
	}
}

// =============================================================================
// Numerical Utilities using gonum for production quality
// =============================================================================

// CosineSimilarity computes cosine similarity between two vectors using gonum
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	dot := floats.Dot(a, b)
	normA := floats.Norm(a, 2)
	normB := floats.Norm(b, 2)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (normA * normB)
}

// EuclideanDistance computes Euclidean distance between two vectors using gonum
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	return floats.Distance(a, b, 2)
}

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(v []float64) []float64 {
	norm := floats.Norm(v, 2)
	if norm == 0 {
		return v
	}

	result := make([]float64, len(v))
	copy(result, v)
	floats.Scale(1/norm, result)
	return result
}

// Float32ToFloat64 converts float32 slice to float64 for gonum operations
func Float32ToFloat64(input []float32) []float64 {
	result := make([]float64, len(input))
	for i, v := range input {
		result[i] = float64(v)
	}
	return result
}

// Softmax computes softmax with numerical stability
func Softmax(x []float64) []float64 {
	if len(x) == 0 {
		return x
	}

	// Find max for numerical stability
	maxVal := floats.Max(x)

	result := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		sum += result[i]
	}

	if sum > 0 {
		floats.Scale(1/sum, result)
	}

	return result
}

// =============================================================================
// Base implementations with gonum
// =============================================================================

// baseSelector provides common functionality for all selectors
type baseSelector struct {
	mu       sync.RWMutex
	training []TrainingRecord
	maxSize  int
}

func newBaseSelector(maxSize int) baseSelector {
	return baseSelector{
		training: make([]TrainingRecord, 0, maxSize),
		maxSize:  maxSize,
	}
}

func (s *baseSelector) addTrainingData(data []TrainingRecord) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.training = append(s.training, data...)

	// Keep only recent records
	if len(s.training) > s.maxSize {
		s.training = s.training[len(s.training)-s.maxSize:]
	}
}

func (s *baseSelector) getTrainingData() []TrainingRecord {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]TrainingRecord, len(s.training))
	copy(result, s.training)
	return result
}

func (s *baseSelector) getTrainingCount() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.training)
}

// SavedModelData represents the JSON structure of saved models
type SavedModelData struct {
	Version   string           `json:"version"`
	Algorithm string           `json:"algorithm"`
	Training  []TrainingRecord `json:"training"`
	Trained   bool             `json:"trained"` // Whether the model was successfully trained
	InputDim  int              `json:"input_dim,omitempty"`
	// Algorithm-specific fields
	K              int                            `json:"k,omitempty"`
	NumClusters    int                            `json:"num_clusters,omitempty"`
	Centroids      [][]float64                    `json:"centroids,omitempty"`
	ClusterModels  []string                       `json:"cluster_models,omitempty"`
	ClusterStats   map[int]map[string]interface{} `json:"cluster_stats,omitempty"`
	HiddenLayers   []int                          `json:"hidden_layers,omitempty"`
	Kernel         string                         `json:"kernel,omitempty"`
	NumFactors     int                            `json:"num_factors,omitempty"`
	ModelToIdx     map[string]int                 `json:"model_to_idx,omitempty"`
	IdxToModel     []string                       `json:"idx_to_model,omitempty"`
	QueryFactors   [][]float64                    `json:"query_factors,omitempty"`
	Projection     [][]float64                    `json:"projection,omitempty"` // Alternative field name for query_factors
	ModelFactors   [][]float64                    `json:"model_factors,omitempty"`
	ModelBiases    []float64                      `json:"model_biases,omitempty"`
	GlobalBias     float64                        `json:"global_bias,omitempty"`
	EffWeight      float64                        `json:"efficiency_weight,omitempty"`
	SupportVectors map[int][][]float64            `json:"support_vectors,omitempty"`
	SVCoeffs       map[int][]float64              `json:"sv_coeffs,omitempty"`
	Alphas         map[int][]float64              `json:"alphas,omitempty"` // Alternative field name for sv_coeffs
	Biases         json.RawMessage                `json:"biases,omitempty"` // Can be array (MLP) or map (SVM)
	Gamma          float64                        `json:"gamma,omitempty"`
	Weights        [][][]float64                  `json:"weights,omitempty"` // MLP weights
}

// loadBaseTrainingData loads training data from JSON into the base selector
func (s *baseSelector) loadFromJSON(data []byte) (*SavedModelData, error) {
	var modelData SavedModelData
	if err := json.Unmarshal(data, &modelData); err != nil {
		return nil, fmt.Errorf("failed to parse model JSON: %w", err)
	}

	// Load training records
	if len(modelData.Training) > 0 {
		s.mu.Lock()
		s.training = modelData.Training
		s.mu.Unlock()
	}

	return &modelData, nil
}

// getModelIndex returns the index of the model in refs, using LoRA name if present
func getModelName(ref config.ModelRef) string {
	if ref.LoRAName != "" {
		return ref.LoRAName
	}
	return ref.Model
}

// buildModelIndex builds a map from model name to ref index
func buildModelIndex(refs []config.ModelRef) map[string]int {
	index := make(map[string]int, len(refs))
	for i, ref := range refs {
		index[getModelName(ref)] = i
	}
	return index
}

// =============================================================================
// KNN Selector - Linfa/Rust Implementation (via ml-binding)
// =============================================================================

// KNNSelector implements K-Nearest Neighbors using Linfa (linfa-nn)
type KNNSelector struct {
	baseSelector
	k     int
	mlKNN *ml_binding.KNNSelector
}

// NewKNNSelector creates a new KNN selector using Linfa
func NewKNNSelector(k int) *KNNSelector {
	if k <= 0 {
		k = 3
	}
	return &KNNSelector{
		baseSelector: newBaseSelector(10000),
		k:            k,
		mlKNN:        ml_binding.NewKNNSelector(k),
	}
}

func (s *KNNSelector) Name() string { return "knn" }

// LoadFromJSON loads a pre-trained KNN model from JSON data
func (s *KNNSelector) LoadFromJSON(data []byte) error {
	// Load into baseSelector for compatibility
	modelData, err := s.loadFromJSON(data)
	if err != nil {
		return err
	}
	if modelData.K > 0 {
		s.k = modelData.K
	}

	// Also load into ml-binding
	knn, err := ml_binding.KNNFromJSON(string(data))
	if err != nil {
		// Fallback: train ml-binding from loaded training data
		s.trainMLBinding()
		return nil
	}
	s.mlKNN = knn
	return nil
}

// trainMLBinding trains the ml-binding KNN from baseSelector data
func (s *KNNSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	embeddings := make([][]float64, len(training))
	labels := make([]string, len(training))
	for i, rec := range training {
		embeddings[i] = rec.QueryEmbedding
		labels[i] = rec.SelectedModel
	}

	if s.mlKNN == nil {
		s.mlKNN = ml_binding.NewKNNSelector(s.k)
	}
	_ = s.mlKNN.Train(embeddings, labels)
}

func (s *KNNSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)

	// Train ml-binding KNN
	s.trainMLBinding()

	logging.Infof("KNN (Linfa) trained with %d records", len(data))
	return nil
}

func (s *KNNSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	if s.mlKNN == nil || !s.mlKNN.IsTrained() || len(ctx.QueryEmbedding) == 0 {
		logging.Debugf("KNN: Not trained or no embedding, selecting first model")
		return &refs[0], nil
	}

	// Use ml-binding for selection
	selectedModel, err := s.mlKNN.Select(ctx.QueryEmbedding)
	if err != nil {
		logging.Debugf("KNN (Linfa) selection failed: %v, selecting first model", err)
		return &refs[0], nil
	}

	// Find the selected model in refs
	modelIndex := buildModelIndex(refs)
	if idx, ok := modelIndex[selectedModel]; ok {
		logging.Infof("KNN (Linfa) selected model %s", selectedModel)
		return &refs[idx], nil
	}

	return &refs[0], nil
}

// =============================================================================
// KMeans Selector - Linfa/Rust Implementation (via ml-binding)
// Based on Avengers-Pro framework (arXiv:2508.12631)
// =============================================================================

// KMeansSelector implements KMeans clustering using Linfa (linfa-clustering)
type KMeansSelector struct {
	baseSelector
	numClusters      int
	efficiencyWeight float64
	mlKMeans         *ml_binding.KMeansSelector
}

// NewKMeansSelector creates a new KMeans selector using Linfa
func NewKMeansSelector(numClusters int) *KMeansSelector {
	if numClusters <= 0 {
		numClusters = 4
	}
	return &KMeansSelector{
		baseSelector:     newBaseSelector(10000),
		numClusters:      numClusters,
		efficiencyWeight: 0.3, // Default: 70% performance, 30% efficiency
		mlKMeans:         ml_binding.NewKMeansSelector(numClusters),
	}
}

// NewKMeansSelectorWithEfficiency creates a KMeans selector with custom efficiency weight
func NewKMeansSelectorWithEfficiency(numClusters int, efficiencyWeight float64) *KMeansSelector {
	s := NewKMeansSelector(numClusters)
	s.efficiencyWeight = math.Max(0, math.Min(1, efficiencyWeight))
	return s
}

func (s *KMeansSelector) Name() string { return "kmeans" }

// LoadFromJSON loads a pre-trained KMeans model from JSON data
func (s *KMeansSelector) LoadFromJSON(data []byte) error {
	// Load into baseSelector for compatibility
	modelData, err := s.loadFromJSON(data)
	if err != nil {
		return err
	}
	if modelData.NumClusters > 0 {
		s.numClusters = modelData.NumClusters
	}
	if modelData.EffWeight > 0 {
		s.efficiencyWeight = modelData.EffWeight
	}

	// Also load into ml-binding
	kmeans, err := ml_binding.KMeansFromJSON(string(data))
	if err != nil {
		// Fallback: train ml-binding from loaded training data
		s.trainMLBinding()
		return nil
	}
	s.mlKMeans = kmeans
	return nil
}

// trainMLBinding trains the ml-binding KMeans from baseSelector data
func (s *KMeansSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	records := make([]ml_binding.KMeansTrainingRecord, len(training))
	for i, rec := range training {
		records[i] = ml_binding.KMeansTrainingRecord{
			Embedding: rec.QueryEmbedding,
			Label:     rec.SelectedModel,
			Quality:   rec.ResponseQuality,
			LatencyNs: rec.ResponseLatencyNs,
		}
	}

	if s.mlKMeans == nil {
		s.mlKMeans = ml_binding.NewKMeansSelector(s.numClusters)
	}
	_ = s.mlKMeans.Train(records)
}

func (s *KMeansSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)

	// Train ml-binding KMeans
	s.trainMLBinding()

	logging.Infof("KMeans (Linfa) trained with %d records", len(data))
	return nil
}

func (s *KMeansSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	if s.mlKMeans == nil || !s.mlKMeans.IsTrained() || len(ctx.QueryEmbedding) == 0 {
		logging.Debugf("KMeans: Not trained or no embedding, selecting first model")
		return &refs[0], nil
	}

	// Use ml-binding for selection
	selectedModel, err := s.mlKMeans.Select(ctx.QueryEmbedding)
	if err != nil {
		logging.Debugf("KMeans (Linfa) selection failed: %v, selecting first model", err)
		return &refs[0], nil
	}

	// Find the selected model in refs
	modelIndex := buildModelIndex(refs)
	if idx, ok := modelIndex[selectedModel]; ok {
		logging.Infof("KMeans (Linfa) selected model %s", selectedModel)
		return &refs[idx], nil
	}

	return &refs[0], nil
}

// =============================================================================
// MLP Selector - Production Implementation with gonum
// =============================================================================

// MLPSelector implements Multi-Layer Perceptron using gonum matrices
type MLPSelector struct {
	baseSelector
	hiddenLayers []int
	weights      []*mat.Dense
	biases       []*mat.Dense
	modelToIdx   map[string]int
	idxToModel   []string
	inputDim     int
	trained      bool
}

// NewMLPSelector creates a new MLP selector
func NewMLPSelector(hiddenLayers []int) *MLPSelector {
	if len(hiddenLayers) == 0 {
		hiddenLayers = []int{64, 32}
	}
	return &MLPSelector{
		baseSelector: newBaseSelector(10000),
		hiddenLayers: hiddenLayers,
		modelToIdx:   make(map[string]int),
		idxToModel:   make([]string, 0),
	}
}

func (s *MLPSelector) Name() string { return "mlp" }

// LoadFromJSON loads a pre-trained MLP model from JSON data
func (s *MLPSelector) LoadFromJSON(data []byte) error {
	modelData, err := s.loadFromJSON(data)
	if err != nil {
		return err
	}
	if len(modelData.HiddenLayers) > 0 {
		s.hiddenLayers = modelData.HiddenLayers
	}
	if len(modelData.IdxToModel) > 0 {
		s.idxToModel = modelData.IdxToModel
		s.modelToIdx = make(map[string]int)
		for i, m := range s.idxToModel {
			s.modelToIdx[m] = i
		}
	}
	// Load weights if available
	if len(modelData.Weights) > 0 {
		s.weights = make([]*mat.Dense, len(modelData.Weights))
		for i, w := range modelData.Weights {
			if len(w) > 0 && len(w[0]) > 0 {
				rows := len(w)
				cols := len(w[0])
				flatData := make([]float64, rows*cols)
				for r, row := range w {
					copy(flatData[r*cols:], row)
				}
				s.weights[i] = mat.NewDense(rows, cols, flatData)
			}
		}
	}
	// Load biases if available (stored as array of arrays)
	// Biases must be column vectors (len(b), 1) for matrix Add to work in forward pass
	if len(modelData.Biases) > 0 {
		var biasesArray [][]float64
		if err := json.Unmarshal(modelData.Biases, &biasesArray); err == nil {
			s.biases = make([]*mat.Dense, len(biasesArray))
			for i, b := range biasesArray {
				if len(b) > 0 {
					s.biases[i] = mat.NewDense(len(b), 1, b)
				}
			}
		}
	}
	// Load input dimension
	if modelData.InputDim > 0 {
		s.inputDim = modelData.InputDim
	}
	// Respect the trained flag from JSON - only set trained if actually trained
	s.trained = modelData.Trained && len(s.weights) > 0
	return nil
}

func (s *MLPSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)

	s.mu.Lock()
	// Update model mapping
	for _, record := range data {
		if _, exists := s.modelToIdx[record.SelectedModel]; !exists {
			s.modelToIdx[record.SelectedModel] = len(s.idxToModel)
			s.idxToModel = append(s.idxToModel, record.SelectedModel)
		}
	}
	s.mu.Unlock()

	training := s.getTrainingData()
	// Train if we have enough records (need at least 2 different models)
	if len(training) >= 2 && len(s.idxToModel) > 1 {
		s.trainNetwork()
	}

	logging.Infof("MLP selector trained with %d records, %d models", len(data), len(s.idxToModel))
	return nil
}

func (s *MLPSelector) trainNetwork() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find input dimension
	for _, r := range s.training {
		if len(r.QueryEmbedding) > 0 {
			s.inputDim = len(r.QueryEmbedding)
			break
		}
	}
	if s.inputDim == 0 {
		return
	}

	// Build layer sizes
	outputDim := len(s.idxToModel)
	layerSizes := append([]int{s.inputDim}, s.hiddenLayers...)
	layerSizes = append(layerSizes, outputDim)

	// Initialize weights with Xavier initialization
	s.weights = make([]*mat.Dense, len(layerSizes)-1)
	s.biases = make([]*mat.Dense, len(layerSizes)-1)

	for l := 0; l < len(layerSizes)-1; l++ {
		inSize := layerSizes[l]
		outSize := layerSizes[l+1]

		// He initialization (better for ReLU activations)
		scale := math.Sqrt(2.0 / float64(inSize))
		weights := make([]float64, outSize*inSize)
		for i := range weights {
			// Random values from normal distribution approximation
			u1 := rand.Float64()
			u2 := rand.Float64()
			// Box-Muller transform for normal distribution
			z := math.Sqrt(-2*math.Log(u1+1e-10)) * math.Cos(2*math.Pi*u2)
			weights[i] = scale * z
		}

		s.weights[l] = mat.NewDense(outSize, inSize, weights)
		// Initialize biases to small positive values to avoid dead ReLU
		biases := make([]float64, outSize)
		for i := range biases {
			biases[i] = 0.01
		}
		s.biases[l] = mat.NewDense(outSize, 1, biases)
	}

	// Full backpropagation through all layers
	learningRate := 0.05 // Moderate learning rate
	const epochs = 150   // Balanced epochs

	for epoch := 0; epoch < epochs; epoch++ {
		// Learning rate decay (cosine annealing)
		lr := learningRate * 0.5 * (1.0 + math.Cos(math.Pi*float64(epoch)/float64(epochs)))

		for _, record := range s.training {
			if len(record.QueryEmbedding) != s.inputDim || !record.Success {
				continue
			}

			targetIdx, exists := s.modelToIdx[record.SelectedModel]
			if !exists {
				continue
			}

			// Forward pass - store all activations and pre-activations
			input := mat.NewDense(s.inputDim, 1, record.QueryEmbedding)
			activations, preActivations := s.forwardWithPreActivations(input)

			// Create one-hot target (proper classification)
			target := make([]float64, outputDim)
			target[targetIdx] = 1.0 // One-hot encoding, weighted by quality later

			// Backpropagation - compute deltas for each layer
			numLayers := len(s.weights)
			deltas := make([]*mat.Dense, numLayers)

			// Output layer delta: (output - target) for cross-entropy with softmax
			output := activations[numLayers]
			outRows, _ := output.Dims()
			outputDelta := mat.NewDense(outRows, 1, nil)
			for i := 0; i < outRows; i++ {
				outputDelta.Set(i, 0, output.At(i, 0)-target[i])
			}
			deltas[numLayers-1] = outputDelta

			// Hidden layer deltas (backpropagate through ReLU)
			for l := numLayers - 2; l >= 0; l-- {
				// delta[l] = (W[l+1]^T * delta[l+1]) * ReLU'(preActivation[l])
				rows, _ := s.weights[l].Dims()
				delta := mat.NewDense(rows, 1, nil)

				// Compute W[l+1]^T * delta[l+1]
				wT := mat.DenseCopyOf(s.weights[l+1].T())
				delta.Mul(wT, deltas[l+1])

				// Multiply by ReLU derivative
				preAct := preActivations[l+1] // pre-activation of layer l+1's input = activation of layer l before ReLU
				for i := 0; i < rows; i++ {
					if preAct.At(i, 0) <= 0 {
						delta.Set(i, 0, 0) // ReLU derivative is 0 for negative values
					}
				}
				deltas[l] = delta
			}

			// Update weights and biases for all layers
			for l := 0; l < numLayers; l++ {
				wRows, wCols := s.weights[l].Dims()
				prevActivation := activations[l]

				for i := 0; i < wRows; i++ {
					for j := 0; j < wCols; j++ {
						grad := deltas[l].At(i, 0) * prevActivation.At(j, 0)
						s.weights[l].Set(i, j, s.weights[l].At(i, j)-lr*grad)
					}
					s.biases[l].Set(i, 0, s.biases[l].At(i, 0)-lr*deltas[l].At(i, 0))
				}
			}
		}
	}

	s.trained = true
}

// forwardWithPreActivations performs forward pass and stores pre-activation values for backprop
func (s *MLPSelector) forwardWithPreActivations(input *mat.Dense) ([]*mat.Dense, []*mat.Dense) {
	numLayers := len(s.weights)
	activations := make([]*mat.Dense, numLayers+1)
	preActivations := make([]*mat.Dense, numLayers+1)

	activations[0] = input
	preActivations[0] = input

	current := input
	for l := 0; l < numLayers; l++ {
		rows, _ := s.weights[l].Dims()

		// output = weights * input + bias (pre-activation)
		preAct := mat.NewDense(rows, 1, nil)
		preAct.Mul(s.weights[l], current)
		preAct.Add(preAct, s.biases[l])
		preActivations[l+1] = preAct

		// Apply activation function
		output := mat.NewDense(rows, 1, nil)
		if l < numLayers-1 {
			// ReLU for hidden layers
			for i := 0; i < rows; i++ {
				val := preAct.At(i, 0)
				if val > 0 {
					output.Set(i, 0, val)
				} else {
					output.Set(i, 0, 0)
				}
			}
		} else {
			// Softmax for output layer
			vals := make([]float64, rows)
			for i := 0; i < rows; i++ {
				vals[i] = preAct.At(i, 0)
			}
			softmaxVals := Softmax(vals)
			for i := 0; i < rows; i++ {
				output.Set(i, 0, softmaxVals[i])
			}
		}

		activations[l+1] = output
		current = output
	}

	return activations, preActivations
}

func (s *MLPSelector) forward(input *mat.Dense) []*mat.Dense {
	activations := make([]*mat.Dense, len(s.weights)+1)
	activations[0] = input

	current := input
	for l := 0; l < len(s.weights); l++ {
		rows, _ := s.weights[l].Dims()

		// output = weights * input + bias
		output := mat.NewDense(rows, 1, nil)
		output.Mul(s.weights[l], current)
		output.Add(output, s.biases[l])

		// ReLU for hidden layers
		if l < len(s.weights)-1 {
			for i := 0; i < rows; i++ {
				if output.At(i, 0) < 0 {
					output.Set(i, 0, 0)
				}
			}
		} else {
			// Softmax for output
			vals := make([]float64, rows)
			for i := 0; i < rows; i++ {
				vals[i] = output.At(i, 0)
			}
			softmaxVals := Softmax(vals)
			for i := 0; i < rows; i++ {
				output.Set(i, 0, softmaxVals[i])
			}
		}

		activations[l+1] = output
		current = output
	}

	return activations
}

func (s *MLPSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.trained || len(s.weights) == 0 || len(ctx.QueryEmbedding) == 0 || len(ctx.QueryEmbedding) != s.inputDim {
		logging.Debugf("MLP: Not trained or dimension mismatch, selecting first model")
		return &refs[0], nil
	}

	// Forward pass
	input := mat.NewDense(s.inputDim, 1, ctx.QueryEmbedding)
	activations := s.forward(input)
	output := activations[len(activations)-1]

	// Find best available model - iterate in refs order for deterministic tie-breaking
	var bestRef *config.ModelRef
	bestScore := -math.MaxFloat64

	for i := range refs {
		modelName := getModelName(refs[i])
		if modelIdx, exists := s.modelToIdx[modelName]; exists {
			score := output.At(modelIdx, 0)
			// Use > (not >=) for deterministic tie-breaking: first model wins ties
			if score > bestScore {
				bestScore = score
				bestRef = &refs[i]
			}
		}
	}

	if bestRef != nil {
		logging.Infof("MLP selected model %s with score %.3f", getModelName(*bestRef), bestScore)
		return bestRef, nil
	}

	return &refs[0], nil
}

// =============================================================================
// SVM Selector - Linfa/Rust Implementation (via ml-binding)
// =============================================================================

// SVMSelector implements SVM using Linfa (linfa-svm)
type SVMSelector struct {
	baseSelector
	kernel string
	mlSVM  *ml_binding.SVMSelector
}

// NewSVMSelector creates a new SVM selector using Linfa
func NewSVMSelector(kernel string) *SVMSelector {
	if kernel == "" {
		kernel = "rbf"
	}
	return &SVMSelector{
		baseSelector: newBaseSelector(5000),
		kernel:       kernel,
		mlSVM:        ml_binding.NewSVMSelector(),
	}
}

func (s *SVMSelector) Name() string { return "svm" }

// LoadFromJSON loads a pre-trained SVM model from JSON data
func (s *SVMSelector) LoadFromJSON(data []byte) error {
	// Load into baseSelector for compatibility
	modelData, err := s.loadFromJSON(data)
	if err != nil {
		return err
	}
	if modelData.Kernel != "" {
		s.kernel = modelData.Kernel
	}

	// Also load into ml-binding
	svm, err := ml_binding.SVMFromJSON(string(data))
	if err != nil {
		// Fallback: train ml-binding from loaded training data
		s.trainMLBinding()
		return nil
	}
	s.mlSVM = svm
	return nil
}

// trainMLBinding trains the ml-binding SVM from baseSelector data
func (s *SVMSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	embeddings := make([][]float64, len(training))
	labels := make([]string, len(training))
	for i, rec := range training {
		embeddings[i] = rec.QueryEmbedding
		labels[i] = rec.SelectedModel
	}

	if s.mlSVM == nil {
		s.mlSVM = ml_binding.NewSVMSelector()
	}
	_ = s.mlSVM.Train(embeddings, labels)
}

func (s *SVMSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)

	// Train ml-binding SVM
	s.trainMLBinding()

	logging.Infof("SVM (Linfa) trained with %d records", len(data))
	return nil
}

func (s *SVMSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	if s.mlSVM == nil || !s.mlSVM.IsTrained() || len(ctx.QueryEmbedding) == 0 {
		logging.Debugf("SVM: Not trained or no embedding, selecting first model")
		return &refs[0], nil
	}

	// Use ml-binding for selection
	selectedModel, err := s.mlSVM.Select(ctx.QueryEmbedding)
	if err != nil {
		logging.Debugf("SVM (Linfa) selection failed: %v, selecting first model", err)
		return &refs[0], nil
	}

	// Find the selected model in refs
	modelIndex := buildModelIndex(refs)
	if idx, ok := modelIndex[selectedModel]; ok {
		logging.Infof("SVM (Linfa) selected model %s", selectedModel)
		return &refs[idx], nil
	}

	return &refs[0], nil
}

// =============================================================================
// Matrix Factorization Selector - Production Implementation (RouteLLM, arXiv:2406.18665)
// Supports preference-based training using BPR (Bayesian Personalized Ranking)
// =============================================================================

// PreferenceRecord represents a pairwise preference (RouteLLM style)
// For a given query, preferredModel was chosen over otherModel
type PreferenceRecord struct {
	QueryEmbedding []float64
	PreferredModel string
	OtherModel     string
	Confidence     float64 // Optional: confidence in this preference (0-1)
}

// MatrixFactorizationSelector implements collaborative filtering using gonum
// Based on RouteLLM framework that uses preference data for training
type MatrixFactorizationSelector struct {
	baseSelector
	numFactors    int
	modelFactors  *mat.Dense // numModels x numFactors
	projection    *mat.Dense // numFactors x embDim
	modelBiases   []float64  // numModels
	globalBias    float64
	modelToIdx    map[string]int
	idxToModel    []string
	inputDim      int
	trained       bool
	preferences   []PreferenceRecord // RouteLLM-style preference data
	usePreference bool               // Whether to use preference-based training
}

// NewMatrixFactorizationSelector creates a new Matrix Factorization selector
func NewMatrixFactorizationSelector(numFactors int) *MatrixFactorizationSelector {
	if numFactors <= 0 {
		numFactors = 10
	}
	return &MatrixFactorizationSelector{
		baseSelector:  newBaseSelector(10000),
		numFactors:    numFactors,
		modelToIdx:    make(map[string]int),
		idxToModel:    make([]string, 0),
		preferences:   make([]PreferenceRecord, 0),
		usePreference: false,
	}
}

// TrainWithPreferences trains the selector using RouteLLM-style preference data
// This is the recommended training method based on arXiv:2406.18665
func (s *MatrixFactorizationSelector) TrainWithPreferences(prefs []PreferenceRecord) error {
	s.mu.Lock()
	// Add preferences
	s.preferences = append(s.preferences, prefs...)
	if len(s.preferences) > 10000 {
		s.preferences = s.preferences[len(s.preferences)-10000:]
	}

	// Update model mapping
	for _, pref := range prefs {
		if _, exists := s.modelToIdx[pref.PreferredModel]; !exists {
			s.modelToIdx[pref.PreferredModel] = len(s.idxToModel)
			s.idxToModel = append(s.idxToModel, pref.PreferredModel)
		}
		if _, exists := s.modelToIdx[pref.OtherModel]; !exists {
			s.modelToIdx[pref.OtherModel] = len(s.idxToModel)
			s.idxToModel = append(s.idxToModel, pref.OtherModel)
		}
	}
	s.usePreference = true
	s.mu.Unlock()

	if len(s.preferences) >= 50 && len(s.idxToModel) > 1 {
		s.trainWithBPR()
	}

	logging.Infof("MatrixFactorization trained with %d preference records (RouteLLM mode)", len(prefs))
	return nil
}

func (s *MatrixFactorizationSelector) Name() string { return "matrix_factorization" }

// LoadFromJSON loads a pre-trained Matrix Factorization model from JSON data
func (s *MatrixFactorizationSelector) LoadFromJSON(data []byte) error {
	modelData, err := s.loadFromJSON(data)
	if err != nil {
		return err
	}
	if modelData.NumFactors > 0 {
		s.numFactors = modelData.NumFactors
	}
	if len(modelData.IdxToModel) > 0 {
		s.idxToModel = modelData.IdxToModel
		s.modelToIdx = make(map[string]int)
		for i, m := range s.idxToModel {
			s.modelToIdx[m] = i
		}
	}
	if len(modelData.ModelBiases) > 0 {
		s.modelBiases = modelData.ModelBiases
	}
	if modelData.GlobalBias != 0 {
		s.globalBias = modelData.GlobalBias
	}
	if len(modelData.ModelFactors) > 0 {
		// Convert model factors to mat.Dense
		rows := len(modelData.ModelFactors)
		if rows > 0 {
			cols := len(modelData.ModelFactors[0])
			flatData := make([]float64, rows*cols)
			for i, row := range modelData.ModelFactors {
				copy(flatData[i*cols:], row)
			}
			s.modelFactors = mat.NewDense(rows, cols, flatData)
		}
	}
	// Load projection matrix - check both field names for compatibility
	projectionData := modelData.QueryFactors
	if len(projectionData) == 0 && len(modelData.Projection) > 0 {
		projectionData = modelData.Projection
	}
	if len(projectionData) > 0 && len(projectionData[0]) > 0 {
		// Projection matrix from query factors or projection field
		rows := len(projectionData)
		cols := len(projectionData[0])
		flatData := make([]float64, rows*cols)
		for i, row := range projectionData {
			copy(flatData[i*cols:], row)
		}
		s.projection = mat.NewDense(rows, cols, flatData)
	}
	// Load input dimension
	if modelData.InputDim > 0 {
		s.inputDim = modelData.InputDim
	}
	// Respect the trained flag from JSON - only set trained if actually trained
	s.trained = modelData.Trained && s.modelFactors != nil
	return nil
}

func (s *MatrixFactorizationSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)

	s.mu.Lock()
	for _, record := range data {
		if _, exists := s.modelToIdx[record.SelectedModel]; !exists {
			s.modelToIdx[record.SelectedModel] = len(s.idxToModel)
			s.idxToModel = append(s.idxToModel, record.SelectedModel)
		}
	}
	s.mu.Unlock()

	training := s.getTrainingData()
	// Train if we have enough records (need at least 2 different models)
	if len(training) >= 2 && len(s.idxToModel) > 1 {
		s.trainFactorization()
	}

	logging.Infof("MatrixFactorization selector trained with %d records", len(data))
	return nil
}

func (s *MatrixFactorizationSelector) trainFactorization() {
	s.mu.Lock()

	// Find input dimension
	for _, r := range s.training {
		if len(r.QueryEmbedding) > 0 {
			s.inputDim = len(r.QueryEmbedding)
			break
		}
	}
	if s.inputDim == 0 {
		s.mu.Unlock()
		return
	}

	numModels := len(s.idxToModel)
	if numModels < 2 {
		s.mu.Unlock()
		return
	}

	// Generate pseudo-preferences from training data (RouteLLM-inspired approach)
	// For each training record, create preferences where higher quality models are preferred
	type queryModelQuality struct {
		embedding []float64
		model     string
		quality   float64
	}

	// Group by similar embeddings (use cosine similarity threshold)
	embeddingGroups := make(map[int][]queryModelQuality)
	groupID := 0

	for _, r := range s.training {
		if len(r.QueryEmbedding) != s.inputDim || !r.Success {
			continue
		}

		// Find or create group based on embedding similarity
		foundGroup := -1
		for gid, group := range embeddingGroups {
			if len(group) > 0 {
				sim := CosineSimilarity(r.QueryEmbedding, group[0].embedding)
				if sim > 0.7 { // Lower threshold to create more groups with diverse samples
					foundGroup = gid
					break
				}
			}
		}

		if foundGroup < 0 {
			foundGroup = groupID
			embeddingGroups[foundGroup] = make([]queryModelQuality, 0)
			groupID++
		}

		embeddingGroups[foundGroup] = append(embeddingGroups[foundGroup], queryModelQuality{
			embedding: r.QueryEmbedding,
			model:     r.SelectedModel,
			quality:   r.ResponseQuality,
		})
	}

	// Create pseudo-preferences within each group
	pseudoPrefs := make([]PreferenceRecord, 0)
	for _, group := range embeddingGroups {
		if len(group) < 2 {
			continue
		}

		// Compare all pairs within group
		for i := 0; i < len(group); i++ {
			for j := i + 1; j < len(group); j++ {
				if group[i].model == group[j].model {
					continue
				}

				// Higher quality model is preferred
				qualityDiff := math.Abs(group[i].quality - group[j].quality)
				if qualityDiff < 0.1 {
					continue // Skip if qualities are too similar
				}

				var pref PreferenceRecord
				if group[i].quality > group[j].quality {
					pref = PreferenceRecord{
						QueryEmbedding: group[i].embedding,
						PreferredModel: group[i].model,
						OtherModel:     group[j].model,
						Confidence:     qualityDiff, // Use quality difference as confidence
					}
				} else {
					pref = PreferenceRecord{
						QueryEmbedding: group[j].embedding,
						PreferredModel: group[j].model,
						OtherModel:     group[i].model,
						Confidence:     qualityDiff,
					}
				}
				pseudoPrefs = append(pseudoPrefs, pref)
			}
		}
	}

	// If we have pseudo-preferences, use BPR training
	if len(pseudoPrefs) > 0 {
		s.preferences = pseudoPrefs
		s.mu.Unlock()
		s.trainWithBPR()
		return
	}

	// Fallback: Direct embedding-based training with contrastive learning
	// Initialize model factors with better initialization
	scale := 0.1 / math.Sqrt(float64(s.numFactors))
	modelData := make([]float64, numModels*s.numFactors)
	for i := range modelData {
		modelData[i] = scale * (rand.Float64()*2 - 1)
	}
	s.modelFactors = mat.NewDense(numModels, s.numFactors, modelData)

	// Initialize projection matrix
	projScale := 0.1 / math.Sqrt(float64(s.inputDim))
	projData := make([]float64, s.numFactors*s.inputDim)
	for i := range projData {
		projData[i] = projScale * (rand.Float64()*2 - 1)
	}
	s.projection = mat.NewDense(s.numFactors, s.inputDim, projData)

	// Initialize biases with model quality priors
	s.modelBiases = make([]float64, numModels)
	modelQualitySum := make(map[int]float64)
	modelQualityCount := make(map[int]int)

	for _, r := range s.training {
		if modelIdx, exists := s.modelToIdx[r.SelectedModel]; exists && r.Success {
			modelQualitySum[modelIdx] += r.ResponseQuality
			modelQualityCount[modelIdx]++
		}
	}

	for idx := 0; idx < numModels; idx++ {
		if modelQualityCount[idx] > 0 {
			s.modelBiases[idx] = modelQualitySum[idx] / float64(modelQualityCount[idx])
		} else {
			s.modelBiases[idx] = 0.5
		}
	}
	s.globalBias = 0.5

	// Contrastive learning: push correct model embedding closer to query
	learningRate := 0.02
	regularization := 0.01
	const epochs = 50

	for epoch := 0; epoch < epochs; epoch++ {
		lr := learningRate * (1.0 - float64(epoch)/float64(epochs)*0.5)

		for _, record := range s.training {
			if len(record.QueryEmbedding) != s.inputDim || !record.Success {
				continue
			}

			modelIdx, exists := s.modelToIdx[record.SelectedModel]
			if !exists {
				continue
			}

			// Project query to factors
			queryVec := mat.NewDense(s.inputDim, 1, record.QueryEmbedding)
			queryFactors := mat.NewDense(s.numFactors, 1, nil)
			queryFactors.Mul(s.projection, queryVec)

			qf := make([]float64, s.numFactors)
			for i := 0; i < s.numFactors; i++ {
				qf[i] = math.Tanh(queryFactors.At(i, 0))
			}

			// Contrastive update: pull correct model closer, push others away
			for m := 0; m < numModels; m++ {
				for f := 0; f < s.numFactors; f++ {
					oldVal := s.modelFactors.At(m, f)
					if m == modelIdx {
						// Positive sample: increase similarity
						grad := record.ResponseQuality * qf[f]
						newVal := oldVal + lr*(grad-regularization*oldVal)
						s.modelFactors.Set(m, f, newVal)
					} else {
						// Negative sample: decrease similarity (with smaller weight)
						grad := -0.1 * qf[f]
						newVal := oldVal + lr*(grad-regularization*oldVal)
						s.modelFactors.Set(m, f, newVal)
					}
				}
			}
		}
	}

	s.trained = true
	s.mu.Unlock()
}

// trainWithBPR implements Bayesian Personalized Ranking for preference-based training
// This is the RouteLLM approach (arXiv:2406.18665)
func (s *MatrixFactorizationSelector) trainWithBPR() {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find input dimension from preferences
	for _, p := range s.preferences {
		if len(p.QueryEmbedding) > 0 {
			s.inputDim = len(p.QueryEmbedding)
			break
		}
	}
	if s.inputDim == 0 {
		return
	}

	numModels := len(s.idxToModel)

	// Initialize model factors with small random values
	scale := 0.1 / math.Sqrt(float64(s.numFactors))
	modelData := make([]float64, numModels*s.numFactors)
	for i := range modelData {
		modelData[i] = scale * (float64(i%100)/50.0 - 1.0)
	}
	s.modelFactors = mat.NewDense(numModels, s.numFactors, modelData)

	// Initialize projection matrix
	projScale := 0.1 / math.Sqrt(float64(s.inputDim))
	projData := make([]float64, s.numFactors*s.inputDim)
	for i := range projData {
		projData[i] = projScale * (float64(i%100)/50.0 - 1.0)
	}
	s.projection = mat.NewDense(s.numFactors, s.inputDim, projData)

	// Initialize biases
	s.modelBiases = make([]float64, numModels)
	s.globalBias = 0.5

	// BPR-SGD training (RouteLLM approach)
	// Objective: maximize P(preferred > other | query)
	learningRate := 0.05
	regularization := 0.01
	const epochs = 50

	for epoch := 0; epoch < epochs; epoch++ {
		for _, pref := range s.preferences {
			if len(pref.QueryEmbedding) != s.inputDim {
				continue
			}

			prefIdx, prefExists := s.modelToIdx[pref.PreferredModel]
			otherIdx, otherExists := s.modelToIdx[pref.OtherModel]
			if !prefExists || !otherExists {
				continue
			}

			// Project query to factors
			queryVec := mat.NewDense(s.inputDim, 1, pref.QueryEmbedding)
			queryFactors := mat.NewDense(s.numFactors, 1, nil)
			queryFactors.Mul(s.projection, queryVec)

			// Apply tanh activation
			qf := make([]float64, s.numFactors)
			for i := 0; i < s.numFactors; i++ {
				qf[i] = math.Tanh(queryFactors.At(i, 0))
			}

			// Compute scores for both models
			prefRow := s.modelFactors.RowView(prefIdx)
			otherRow := s.modelFactors.RowView(otherIdx)

			var prefScore, otherScore float64
			for f := 0; f < s.numFactors; f++ {
				prefScore += prefRow.AtVec(f) * qf[f]
				otherScore += otherRow.AtVec(f) * qf[f]
			}
			prefScore += s.modelBiases[prefIdx]
			otherScore += s.modelBiases[otherIdx]

			// BPR loss gradient: sigmoid(other - preferred)
			diff := otherScore - prefScore
			sigmoid := 1.0 / (1.0 + math.Exp(-diff))

			// Apply confidence weight if available
			weight := 1.0
			if pref.Confidence > 0 {
				weight = pref.Confidence
			}

			// Update model factors using BPR gradient
			for f := 0; f < s.numFactors; f++ {
				// Preferred model: push score up
				prefOld := s.modelFactors.At(prefIdx, f)
				prefNew := prefOld + learningRate*weight*(sigmoid*qf[f]-regularization*prefOld)
				s.modelFactors.Set(prefIdx, f, prefNew)

				// Other model: push score down
				otherOld := s.modelFactors.At(otherIdx, f)
				otherNew := otherOld + learningRate*weight*(-sigmoid*qf[f]-regularization*otherOld)
				s.modelFactors.Set(otherIdx, f, otherNew)
			}

			// Update biases
			s.modelBiases[prefIdx] += learningRate * weight * (sigmoid - regularization*s.modelBiases[prefIdx])
			s.modelBiases[otherIdx] += learningRate * weight * (-sigmoid - regularization*s.modelBiases[otherIdx])
		}
	}

	s.trained = true
	logging.Infof("MatrixFactorization BPR training completed with %d preferences", len(s.preferences))
}

func (s *MatrixFactorizationSelector) predict(queryFactors []float64, modelIdx int) float64 {
	if modelIdx >= len(s.idxToModel) {
		return s.globalBias
	}

	modelRow := s.modelFactors.RowView(modelIdx)
	var dot float64
	for f := 0; f < s.numFactors && f < len(queryFactors); f++ {
		dot += modelRow.AtVec(f) * queryFactors[f]
	}

	prediction := s.globalBias + s.modelBiases[modelIdx] + dot
	return math.Max(0, math.Min(1, prediction))
}

func (s *MatrixFactorizationSelector) Select(ctx *SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, nil
	}
	if len(refs) == 1 {
		return &refs[0], nil
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	if !s.trained || s.projection == nil || len(ctx.QueryEmbedding) == 0 || len(ctx.QueryEmbedding) != s.inputDim {
		logging.Debugf("MatrixFactorization: Not trained or dimension mismatch, selecting first model")
		return &refs[0], nil
	}

	// Project query to factors
	queryVec := mat.NewDense(s.inputDim, 1, ctx.QueryEmbedding)
	queryFactors := mat.NewDense(s.numFactors, 1, nil)
	queryFactors.Mul(s.projection, queryVec)

	// Apply tanh and convert to slice
	qf := make([]float64, s.numFactors)
	for i := 0; i < s.numFactors; i++ {
		qf[i] = math.Tanh(queryFactors.At(i, 0))
	}

	// Hybrid approach: combine learned factors with embedding similarity
	modelIndex := buildModelIndex(refs)

	// Compute scores from learned factors
	factorScores := make(map[string]float64)
	maxFactorScore := 0.0
	for modelName := range modelIndex {
		if modelIdx, exists := s.modelToIdx[modelName]; exists {
			score := s.predict(qf, modelIdx)
			factorScores[modelName] = score
			if score > maxFactorScore {
				maxFactorScore = score
			}
		}
	}

	// Compute KNN-style similarity scores
	simScores := make(map[string]float64)
	simCounts := make(map[string]int)

	for _, record := range s.training {
		if len(record.QueryEmbedding) == len(ctx.QueryEmbedding) && record.Success {
			if _, exists := modelIndex[record.SelectedModel]; exists {
				sim := CosineSimilarity(ctx.QueryEmbedding, record.QueryEmbedding)
				if sim > 0.5 { // Only consider similar embeddings
					simScores[record.SelectedModel] += sim * record.ResponseQuality
					simCounts[record.SelectedModel]++
				}
			}
		}
	}

	// Normalize similarity scores
	maxSimScore := 0.0
	for model := range simScores {
		if simCounts[model] > 0 {
			simScores[model] /= float64(simCounts[model])
		}
		if simScores[model] > maxSimScore {
			maxSimScore = simScores[model]
		}
	}

	// RouteLLM-style selection: prioritize similarity-based scores for multi-model scenarios
	// When models are similar tier, use more local (KNN-style) information
	// Only use factor scores as tie-breaker or when similarity data is sparse
	var bestRef *config.ModelRef
	bestScore := -1.0

	// Check if we have meaningful similarity data
	hasSimilarityData := maxSimScore > 0.1

	for i := range refs {
		modelName := getModelName(refs[i])
		if _, exists := modelIndex[modelName]; !exists {
			continue
		}

		factorScore := 0.0
		if s, ok := factorScores[modelName]; ok && maxFactorScore > 0 {
			factorScore = s / maxFactorScore // Normalize to 0-1
		}

		simScore := 0.0
		if s, ok := simScores[modelName]; ok && maxSimScore > 0 {
			simScore = s / maxSimScore // Normalize to 0-1
		}

		// Combined score: when we have similarity data, rely more on it
		// This prevents global biases from dominating in multi-model scenarios
		var combined float64
		if hasSimilarityData {
			// 20% factor + 80% similarity when we have local data
			combined = 0.2*factorScore + 0.8*simScore
		} else {
			// 70% factor + 30% random exploration when no similarity data
			combined = 0.7*factorScore + 0.3*rand.Float64()
		}

		// Use > (not >=) for deterministic tie-breaking: first model wins ties
		if combined > bestScore {
			bestScore = combined
			bestRef = &refs[i]
		}
	}

	if bestRef != nil {
		logging.Infof("MatrixFactorization selected model %s with score %.3f", getModelName(*bestRef), bestScore)
		return bestRef, nil
	}

	return &refs[0], nil
}
