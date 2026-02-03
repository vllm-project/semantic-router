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
// This package uses Linfa (Rust) via ml-binding for ML algorithms:
// - KNN (K-Nearest Neighbors)
// - KMeans clustering
// - SVM (Support Vector Machine)
//
// Training is done in Python (src/training/ml_model_selection/).
// This package provides inference-only functionality, loading models from JSON.
package modelselection

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	ml_binding "github.com/vllm-project/semantic-router/ml-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// UseLinfa enables Rust/Linfa implementations for KNN, KMeans, SVM.
// When true, uses ml-binding (faster, battle-tested Linfa algorithms).
// When false, uses pure Go implementations (no Rust dependency).
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
	// Construct model file path
	modelPath := modelsPath + "/" + algorithmType + "_model.json"

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

	case "svm":
		selector := NewSVMSelector("rbf") // Default to RBF, will be overridden by JSON if present
		if err := selector.LoadFromJSON(data); err != nil {
			return nil, fmt.Errorf("failed to parse SVM model: %w", err)
		}
		logging.Infof("Loaded SVM selector with %d training records", selector.getTrainingCount())
		return selector, nil

	default:
		return nil, fmt.Errorf("unknown algorithm type: %s (supported: knn, kmeans, svm)", algorithmType)
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

	case "svm":
		kernel := cfg.Kernel
		if kernel == "" {
			kernel = "rbf" // Default to RBF - better for high-dimensional embeddings
		}
		return NewSVMSelector(kernel), nil

	default:
		return nil, fmt.Errorf("unknown model selection algorithm: %s (supported: knn, kmeans, svm)", cfg.Type)
	}
}

// =============================================================================
// Numerical Utilities (pure Go implementations)
// =============================================================================

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (normA * normB)
}

// EuclideanDistance computes Euclidean distance between two vectors
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	var sum float64
	for i := 0; i < len(a); i++ {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// NormalizeVector normalizes a vector to unit length
func NormalizeVector(v []float64) []float64 {
	var norm float64
	for _, val := range v {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return v
	}

	result := make([]float64, len(v))
	for i, val := range v {
		result[i] = val / norm
	}
	return result
}

// Float32ToFloat64 converts float32 slice to float64
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
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	result := make([]float64, len(x))
	var sum float64
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		sum += result[i]
	}

	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}

	return result
}

// =============================================================================
// Base implementations
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
	K             int                            `json:"k,omitempty"`
	NumClusters   int                            `json:"num_clusters,omitempty"`
	Centroids     [][]float64                    `json:"centroids,omitempty"`
	ClusterModels []string                       `json:"cluster_models,omitempty"`
	ClusterStats  map[int]map[string]interface{} `json:"cluster_stats,omitempty"`
	Kernel        string                         `json:"kernel,omitempty"`
	EffWeight     float64                        `json:"efficiency_weight,omitempty"`
	Gamma         float64                        `json:"gamma,omitempty"`
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

// trainMLBinding is deprecated - training should now be done in Python
// See: src/training/ml_model_selection/train.py
// This method logs a warning and does nothing. Use LoadPretrainedModel() instead.
func (s *KNNSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	// Training is now done in Python (src/training/ml_model_selection/)
	// The Go/Rust code only loads pretrained models via LoadFromJSON()
	logging.Warnf("KNN training in Go is deprecated. Use Python training: python src/training/ml_model_selection/train.py")
	logging.Infof("To load pretrained models, use LoadPretrainedSelector() with JSON files")

	// Initialize empty selector if needed (for compatibility)
	if s.mlKNN == nil {
		s.mlKNN = ml_binding.NewKNNSelector(s.k)
	}
}

func (s *KNNSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)
	s.trainMLBinding()
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

	// Build feature vector: embedding + category one-hot (matches Python training format)
	// Uses CombineEmbeddingWithCategory from trainer.go to ensure consistency
	featureVector := CombineEmbeddingWithCategory(ctx.QueryEmbedding, ctx.CategoryName)

	// Use ml-binding for selection
	selectedModel, err := s.mlKNN.Select(featureVector)
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

// trainMLBinding is deprecated - training should now be done in Python
// See: src/training/ml_model_selection/train.py
// This method logs a warning and does nothing. Use LoadPretrainedModel() instead.
func (s *KMeansSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	// Training is now done in Python (src/training/ml_model_selection/)
	// The Go/Rust code only loads pretrained models via LoadFromJSON()
	logging.Warnf("KMeans training in Go is deprecated. Use Python training: python src/training/ml_model_selection/train.py")
	logging.Infof("To load pretrained models, use LoadPretrainedSelector() with JSON files")

	// Initialize empty selector if needed (for compatibility)
	if s.mlKMeans == nil {
		s.mlKMeans = ml_binding.NewKMeansSelector(s.numClusters)
	}
}

func (s *KMeansSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)
	s.trainMLBinding()
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

	// Build feature vector: embedding + category one-hot (matches Python training format)
	// Uses CombineEmbeddingWithCategory from trainer.go to ensure consistency
	featureVector := CombineEmbeddingWithCategory(ctx.QueryEmbedding, ctx.CategoryName)

	// Use ml-binding for selection
	selectedModel, err := s.mlKMeans.Select(featureVector)
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
		kernel = "rbf" // Default to RBF - better for high-dimensional embeddings with quality filtering
	}

	// Create ml-binding SVM with appropriate kernel
	var mlSVM *ml_binding.SVMSelector
	switch kernel {
	case "rbf", "gaussian":
		mlSVM = ml_binding.NewSVMSelectorWithKernel(ml_binding.SVMKernelRBF, 0) // 0 = auto gamma
	default:
		mlSVM = ml_binding.NewSVMSelectorWithKernel(ml_binding.SVMKernelLinear, 0)
	}

	return &SVMSelector{
		baseSelector: newBaseSelector(5000),
		kernel:       kernel,
		mlSVM:        mlSVM,
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

// trainMLBinding is deprecated - training should now be done in Python
// See: src/training/ml_model_selection/train.py
// This method logs a warning and does nothing. Use LoadPretrainedModel() instead.
func (s *SVMSelector) trainMLBinding() {
	training := s.getTrainingData()
	if len(training) == 0 {
		return
	}

	// Training is now done in Python (src/training/ml_model_selection/)
	// The Go/Rust code only loads pretrained models via LoadFromJSON()
	logging.Warnf("SVM training in Go is deprecated. Use Python training: python src/training/ml_model_selection/train.py")
	logging.Infof("To load pretrained models, use LoadPretrainedSelector() with JSON files")

	// Initialize empty selector if needed (for compatibility)
	if s.mlSVM == nil {
		switch s.kernel {
		case "rbf", "gaussian":
			s.mlSVM = ml_binding.NewSVMSelectorWithKernel(ml_binding.SVMKernelRBF, 0)
		default:
			s.mlSVM = ml_binding.NewSVMSelectorWithKernel(ml_binding.SVMKernelLinear, 0)
		}
	}
}

func (s *SVMSelector) Train(data []TrainingRecord) error {
	s.addTrainingData(data)
	s.trainMLBinding()
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

	// Build feature vector: embedding + category one-hot (matches Python training format)
	// Uses CombineEmbeddingWithCategory from trainer.go to ensure consistency
	featureVector := CombineEmbeddingWithCategory(ctx.QueryEmbedding, ctx.CategoryName)

	// Use ml-binding for selection
	selectedModel, err := s.mlSVM.Select(featureVector)
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
