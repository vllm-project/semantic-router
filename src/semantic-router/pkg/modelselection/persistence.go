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
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	ml_binding "github.com/vllm-project/semantic-router/ml-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// SaveableSelector interface for selectors that can persist their state
type SaveableSelector interface {
	Selector
	// Save persists the trained model to a file
	Save(path string) error
	// Load restores a trained model from a file
	Load(path string) error
}

// ============================================================================
// Training Data Loading
// ============================================================================

// RoutingDataRecord represents a single training record from JSONL file
type RoutingDataRecord struct {
	Query       string             `json:"query"`
	QueryType   string             `json:"query_type"`
	BestModel   string             `json:"best_model"`
	ModelScores map[string]float64 `json:"model_scores"`
	EmbeddingID int                `json:"embedding_id"` // For pre-computed embedding lookup
}

// LLMCandidate represents an LLM model configuration
type LLMCandidate struct {
	Provider             string   `json:"provider"`
	ModelID              string   `json:"model_id"`
	DisplayName          string   `json:"display_name"`
	Category             string   `json:"category"`
	CostPerKInputTokens  float64  `json:"cost_per_1k_input_tokens"`
	CostPerKOutputTokens float64  `json:"cost_per_1k_output_tokens"`
	MaxContextLength     int      `json:"max_context_length"`
	Strengths            []string `json:"strengths"`
	AvgLatencyMs         float64  `json:"avg_latency_ms"`
	QualityScore         float64  `json:"quality_score"`
}

// LLMCandidatesConfig holds all LLM candidates configuration
type LLMCandidatesConfig struct {
	LLMCandidates map[string]LLMCandidate  `json:"llm_candidates"`
	QueryTypes    map[string]QueryTypeInfo `json:"query_types"`
}

// QueryTypeInfo defines a category of queries
type QueryTypeInfo struct {
	Description    string   `json:"description"`
	BestModels     []string `json:"best_models"`
	ExampleQueries []string `json:"example_queries"`
}

// BenchmarkRecord represents a single record from training_data_with_category.jsonl
type BenchmarkRecord struct {
	TaskName     string  `json:"task_name"`
	Query        string  `json:"query"`
	Category     string  `json:"category"`
	ModelName    string  `json:"model_name"`
	Performance  float64 `json:"performance"`
	ResponseTime float64 `json:"response_time"`
	EmbeddingID  int     `json:"embedding_id"`
	Response     string  `json:"response"`     // Model's response
	GroundTruth  string  `json:"ground_truth"` // Correct answer for quality scoring
}

// LoadBenchmarkData loads training data from training_data_with_category.jsonl
// Returns records grouped by embedding_id (unique queries)
func LoadBenchmarkData(path string) (map[int][]BenchmarkRecord, error) {
	return LoadBenchmarkDataFiltered(path, nil)
}

// LoadBenchmarkDataFiltered loads training data filtered by specific models
// If allowedModels is nil or empty, loads all models
// Returns records grouped by embedding_id (unique queries)
func LoadBenchmarkDataFiltered(path string, allowedModels []string) (map[int][]BenchmarkRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open benchmark data file: %w", err)
	}
	defer file.Close()

	// Build allowed models map for fast lookup
	allowedMap := make(map[string]bool)
	filterByModel := len(allowedModels) > 0
	for _, m := range allowedModels {
		allowedMap[m] = true
	}

	// Use larger buffer for big files
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	recordsByQuery := make(map[int][]BenchmarkRecord)
	lineNum := 0
	includedRecords := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if line == "" {
			continue
		}

		var record BenchmarkRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			logging.Warnf("Failed to parse line %d: %v", lineNum, err)
			continue
		}

		// Filter by allowed models if specified
		if filterByModel && !allowedMap[record.ModelName] {
			continue
		}

		recordsByQuery[record.EmbeddingID] = append(recordsByQuery[record.EmbeddingID], record)
		includedRecords++
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading benchmark data: %w", err)
	}

	if filterByModel {
		logging.Infof("Loaded %d unique queries (%d records from %d models) from %s",
			len(recordsByQuery), includedRecords, len(allowedModels), path)
	} else {
		logging.Infof("Loaded %d unique queries (%d total records) from %s",
			len(recordsByQuery), lineNum, path)
	}
	return recordsByQuery, nil
}

// ConvertBenchmarkToRoutingData converts benchmark format to RoutingDataRecord format
// Implements RouteLLM-style scoring: quality (from correctness) + efficiency (from latency)
// qualityWeight: 0.0 = pure speed, 1.0 = pure quality (default: 0.7)
func ConvertBenchmarkToRoutingData(recordsByQuery map[int][]BenchmarkRecord, qualityWeight float64) []RoutingDataRecord {
	var routingData []RoutingDataRecord

	for embeddingID, records := range recordsByQuery {
		if len(records) == 0 {
			continue
		}

		// Get query info from first record
		first := records[0]

		// Find min/max latency for normalization
		minLatency := math.MaxFloat64
		maxLatency := 0.0
		for _, r := range records {
			if r.ResponseTime > 0 {
				if r.ResponseTime < minLatency {
					minLatency = r.ResponseTime
				}
				if r.ResponseTime > maxLatency {
					maxLatency = r.ResponseTime
				}
			}
		}

		// Build model scores for this query
		// RouteLLM approach: combine quality with efficiency
		// Score = quality_weight * quality + efficiency_weight * (1 - normalized_latency)
		modelScores := make(map[string]float64)
		var bestModel string
		bestScore := -1.0

		// First pass: collect all quality scores
		qualityScores := make(map[string]float64)
		efficiencyScores := make(map[string]float64)

		for _, r := range records {
			// Quality score: use Performance if available, or compute from correctness
			qualityScore := r.Performance
			if qualityScore == 0 && r.GroundTruth != "" && r.Response != "" {
				// Compute quality from response correctness
				qualityScore = computeResponseQuality(r.Response, r.GroundTruth)
			}
			qualityScores[r.ModelName] = qualityScore

			// Efficiency score: inverse normalized latency (faster = better)
			efficiencyScore := 0.5 // Default if no latency data
			if r.ResponseTime > 0 && maxLatency > minLatency {
				// Normalize: 1.0 for fastest, 0.0 for slowest
				efficiencyScore = 1.0 - (r.ResponseTime-minLatency)/(maxLatency-minLatency)
			} else if r.ResponseTime > 0 && maxLatency == minLatency {
				efficiencyScore = 1.0 // All same latency
			}
			efficiencyScores[r.ModelName] = efficiencyScore
		}

		// RouteLLM approach: Quality first, then efficiency as tiebreaker
		// When quality is equal, faster model wins (correct behavior)
		// qualityWeight is passed as parameter (default 0.7 = 70% quality, 30% speed)
		efficiencyWeight := 1.0 - qualityWeight

		for _, r := range records {
			qScore := qualityScores[r.ModelName]
			eScore := efficiencyScores[r.ModelName]

			// Combined score: quality primary, efficiency tiebreaker
			combinedScore := qualityWeight*qScore + efficiencyWeight*eScore

			modelScores[r.ModelName] = combinedScore
			if combinedScore > bestScore {
				bestScore = combinedScore
				bestModel = r.ModelName
			}
		}

		// Use category as query_type, preserve embedding_id for pre-computed lookup
		routingData = append(routingData, RoutingDataRecord{
			Query:       first.Query,
			QueryType:   first.Category, // Use category field
			BestModel:   bestModel,
			ModelScores: modelScores,
			EmbeddingID: embeddingID, // Preserve for pre-computed embedding lookup
		})
	}

	logging.Infof("Converted %d routing data records", len(routingData))
	return routingData
}

// computeResponseQuality computes quality score by comparing response to ground truth
// Returns 1.0 for correct, 0.0 for incorrect, with partial matching for text responses
func computeResponseQuality(response, groundTruth string) float64 {
	// Normalize strings for comparison
	response = strings.TrimSpace(strings.ToLower(response))
	groundTruth = strings.TrimSpace(strings.ToLower(groundTruth))

	// Exact match
	if response == groundTruth {
		return 1.0
	}

	// For multiple choice, check if response contains the correct option letter
	if len(groundTruth) == 1 && (groundTruth[0] >= 'a' && groundTruth[0] <= 'z') {
		// Ground truth is a single letter (A, B, C, D)
		if strings.HasPrefix(response, groundTruth) ||
			strings.Contains(response, " "+groundTruth) ||
			strings.Contains(response, groundTruth+".") ||
			strings.Contains(response, groundTruth+")") {
			return 1.0
		}
		// Check for uppercase version
		upperGT := strings.ToUpper(groundTruth)
		if strings.HasPrefix(strings.ToUpper(response), upperGT) ||
			strings.Contains(strings.ToUpper(response), " "+upperGT) {
			return 1.0
		}
	}

	// For numeric answers, try to extract and compare numbers
	respNum := extractFirstNumber(response)
	gtNum := extractFirstNumber(groundTruth)
	if respNum != "" && gtNum != "" && respNum == gtNum {
		return 1.0
	}

	// Partial match: check if ground truth is contained in response
	if strings.Contains(response, groundTruth) {
		return 0.8
	}

	// Check word overlap for longer responses
	gtWords := strings.Fields(groundTruth)
	respWords := strings.Fields(response)
	if len(gtWords) > 3 && len(respWords) > 3 {
		matchCount := 0
		for _, gtw := range gtWords {
			for _, rw := range respWords {
				if gtw == rw && len(gtw) > 2 {
					matchCount++
					break
				}
			}
		}
		overlap := float64(matchCount) / float64(len(gtWords))
		if overlap > 0.3 {
			return overlap * 0.5 // Partial credit
		}
	}

	return 0.0
}

// extractFirstNumber extracts the first number from a string
func extractFirstNumber(s string) string {
	var num strings.Builder
	inNumber := false
	for _, c := range s {
		if c >= '0' && c <= '9' || (c == '.' && inNumber) || (c == '-' && !inNumber) {
			num.WriteRune(c)
			inNumber = true
		} else if inNumber {
			break
		}
	}
	return num.String()
}

// LoadRoutingData loads training data from a JSONL file
func LoadRoutingData(path string) ([]RoutingDataRecord, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open routing data file: %w", err)
	}
	defer file.Close()

	var records []RoutingDataRecord
	scanner := bufio.NewScanner(file)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		line := scanner.Text()
		if line == "" {
			continue
		}

		var record RoutingDataRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			logging.Warnf("Failed to parse line %d: %v", lineNum, err)
			continue
		}
		records = append(records, record)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading routing data: %w", err)
	}

	logging.Infof("Loaded %d routing data records from %s", len(records), path)
	return records, nil
}

// LoadLLMCandidates loads LLM candidate configurations
func LoadLLMCandidates(path string) (*LLMCandidatesConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read LLM candidates file: %w", err)
	}

	var cfg LLMCandidatesConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse LLM candidates: %w", err)
	}

	logging.Infof("Loaded %d LLM candidates from %s", len(cfg.LLMCandidates), path)
	return &cfg, nil
}

// ============================================================================
// Serializable Training Record (for JSON)
// ============================================================================

// SerializableTrainingRecord is a JSON-friendly version of TrainingRecord
type SerializableTrainingRecord struct {
	QueryEmbedding  []float64 `json:"query_embedding"`
	SelectedModel   string    `json:"selected_model"`
	ResponseLatency int64     `json:"response_latency_ns"` // Duration as nanoseconds
	ResponseQuality float64   `json:"response_quality"`
	Success         bool      `json:"success"`
	Timestamp       int64     `json:"timestamp"` // Unix timestamp
}

func toSerializable(r TrainingRecord) SerializableTrainingRecord {
	return SerializableTrainingRecord{
		QueryEmbedding:  r.QueryEmbedding,
		SelectedModel:   r.SelectedModel,
		ResponseLatency: r.ResponseLatencyNs,
		ResponseQuality: r.ResponseQuality,
		Success:         r.Success,
		Timestamp:       r.TimestampUnix,
	}
}

func fromSerializable(s SerializableTrainingRecord) TrainingRecord {
	return TrainingRecord{
		QueryEmbedding:    s.QueryEmbedding,
		SelectedModel:     s.SelectedModel,
		ResponseLatencyNs: s.ResponseLatency,
		ResponseQuality:   s.ResponseQuality,
		Success:           s.Success,
		TimestampUnix:     s.Timestamp,
	}
}

// ============================================================================
// KNN Model Persistence
// ============================================================================

// KNNModelData holds serializable KNN model state
type KNNModelData struct {
	Version   string                       `json:"version"`
	Algorithm string                       `json:"algorithm"`
	K         int                          `json:"k"`
	Training  []SerializableTrainingRecord `json:"training"`
	Metadata  map[string]string            `json:"metadata"`
}

// Save persists KNN model to file
func (s *KNNSelector) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Convert training records
	training := make([]SerializableTrainingRecord, len(s.training))
	for i, r := range s.training {
		training[i] = toSerializable(r)
	}

	data := KNNModelData{
		Version:   "1.0",
		Algorithm: "knn",
		K:         s.k,
		Training:  training,
		Metadata: map[string]string{
			"record_count": fmt.Sprintf("%d", len(s.training)),
		},
	}

	return saveModelJSON(path, data)
}

// Load restores KNN model from file
func (s *KNNSelector) Load(path string) error {
	// Read the raw JSON to pass to Rust binding
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// Load into Rust/Linfa binding for inference
	mlKNN, err := ml_binding.KNNFromJSON(string(jsonData))
	if err != nil {
		logging.Warnf("Failed to load KNN into Rust binding: %v (will use Go fallback)", err)
	} else {
		s.mu.Lock()
		if s.mlKNN != nil {
			s.mlKNN.Close()
		}
		s.mlKNN = mlKNN
		s.mu.Unlock()
	}

	// Also parse into Go struct for metadata
	var data KNNModelData
	if err := loadModelJSON(path, &data); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.k = data.K
	s.training = make([]TrainingRecord, len(data.Training))
	for i, r := range data.Training {
		s.training[i] = fromSerializable(r)
	}

	logging.Infof("Loaded KNN model with %d training records from %s", len(s.training), path)
	return nil
}

// ============================================================================
// KMeans Model Persistence
// ============================================================================

// KMeansModelData holds serializable KMeans model state
type KMeansModelData struct {
	Version          string                       `json:"version"`
	Algorithm        string                       `json:"algorithm"`
	NumClusters      int                          `json:"num_clusters"`
	NCluster         int                          `json:"n_clusters"` // Python uses n_clusters
	Centroids        [][]float64                  `json:"centroids"`
	ClusterModels    []string                     `json:"cluster_models"` // Now outputs as array for Rust compatibility
	ModelNames       []string                     `json:"model_names"`    // Python includes model_names
	FeatureDim       int                          `json:"feature_dim"`    // Python includes feature_dim
	EfficiencyWeight float64                      `json:"efficiency_weight"`
	Training         []SerializableTrainingRecord `json:"training"`
	Trained          bool                         `json:"trained"`
}

// Save persists KMeans model to file
func (s *KMeansSelector) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Convert training records
	training := make([]SerializableTrainingRecord, len(s.training))
	for i, r := range s.training {
		training[i] = toSerializable(r)
	}

	trained := s.mlKMeans != nil && s.mlKMeans.IsTrained()

	data := KMeansModelData{
		Version:          "1.0",
		Algorithm:        "kmeans",
		NumClusters:      s.numClusters,
		Centroids:        nil, // Not used - retrained on load from training records
		ClusterModels:    nil, // Not used - retrained on load from training records
		EfficiencyWeight: s.efficiencyWeight,
		Training:         training,
		Trained:          trained,
	}

	return saveModelJSON(path, data)
}

// Load restores KMeans model from file
func (s *KMeansSelector) Load(path string) error {
	// Read the raw JSON to pass to Rust binding
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// Load into Rust/Linfa binding for inference
	mlKMeans, err := ml_binding.KMeansFromJSON(string(jsonData))
	if err != nil {
		logging.Warnf("Failed to load KMeans into Rust binding: %v (will use Go fallback)", err)
	} else {
		s.mu.Lock()
		if s.mlKMeans != nil {
			s.mlKMeans.Close()
		}
		s.mlKMeans = mlKMeans
		s.mu.Unlock()
	}

	// Also parse into Go struct for metadata
	var data KMeansModelData
	if err := loadModelJSON(path, &data); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Support both Go (num_clusters) and Python (n_clusters) field names
	if data.NumClusters > 0 {
		s.numClusters = data.NumClusters
	} else if data.NCluster > 0 {
		s.numClusters = data.NCluster
	}
	s.efficiencyWeight = data.EfficiencyWeight

	// Convert training records
	s.training = make([]TrainingRecord, len(data.Training))
	for i, r := range data.Training {
		s.training[i] = fromSerializable(r)
	}

	logging.Infof("Loaded KMeans model with %d clusters, %d training records from %s",
		s.numClusters, len(s.training), path)
	return nil
}

// ============================================================================
// SVM Model Persistence
// ============================================================================

// SVMModelData holds serializable SVM model state
type SVMModelData struct {
	Version        string                       `json:"version"`
	Algorithm      string                       `json:"algorithm"`
	Kernel         string                       `json:"kernel"`
	Gamma          float64                      `json:"gamma"`
	C              float64                      `json:"C"`               // Python includes C parameter
	ModelNames     []string                     `json:"model_names"`     // Python uses model_names
	FeatureDim     int                          `json:"feature_dim"`     // Python includes feature_dim
	NClasses       int                          `json:"n_classes"`       // Python includes n_classes
	ModelToIdx     map[string]int               `json:"model_to_idx"`    // May not be present from Python
	IdxToModel     []string                     `json:"idx_to_model"`    // May not be present from Python
	SupportVectors [][]float64                  `json:"support_vectors"` // Python outputs 2D array
	DualCoef       [][]float64                  `json:"dual_coef"`       // Python outputs dual_coef
	Intercept      []float64                    `json:"intercept"`       // Python outputs intercept
	NSupport       []int                        `json:"n_support"`       // Python outputs n_support
	Classes        []int                        `json:"classes"`         // Python outputs classes
	Alphas         map[int][]float64            `json:"alphas"`          // Legacy field
	Biases         []float64                    `json:"biases"`          // Legacy field
	Training       []SerializableTrainingRecord `json:"training"`
	Trained        bool                         `json:"trained"`
}

// Save persists SVM model to file
func (s *SVMSelector) Save(path string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Convert training records
	training := make([]SerializableTrainingRecord, len(s.training))
	for i, r := range s.training {
		training[i] = toSerializable(r)
	}

	trained := s.mlSVM != nil && s.mlSVM.IsTrained()

	data := SVMModelData{
		Version:        "1.0",
		Algorithm:      "svm",
		Kernel:         s.kernel,
		Gamma:          0.5, // Optimized for high-dim normalized embeddings
		ModelToIdx:     nil, // Retrained on load from training records
		IdxToModel:     nil, // Retrained on load from training records
		SupportVectors: nil, // Retrained on load from training records
		Alphas:         nil, // Not used with Linfa
		Biases:         nil, // Not used with Linfa
		Training:       training,
		Trained:        trained,
	}

	return saveModelJSON(path, data)
}

// Load restores SVM model from file
func (s *SVMSelector) Load(path string) error {
	// Read the raw JSON to pass to Rust binding
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	// Load into Rust/Linfa binding for inference
	mlSVM, err := ml_binding.SVMFromJSON(string(jsonData))
	if err != nil {
		logging.Warnf("Failed to load SVM into Rust binding: %v (will use Go fallback)", err)
	} else {
		s.mu.Lock()
		if s.mlSVM != nil {
			s.mlSVM.Close()
		}
		s.mlSVM = mlSVM
		s.mu.Unlock()
	}

	// Also parse into Go struct for metadata
	var data SVMModelData
	if err := loadModelJSON(path, &data); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.kernel = data.Kernel

	// Convert training records
	s.training = make([]TrainingRecord, len(data.Training))
	for i, r := range data.Training {
		s.training[i] = fromSerializable(r)
	}

	logging.Infof("Loaded SVM model with %d training records from %s", len(s.training), path)
	return nil
}

// ============================================================================
// Helper Functions
// ============================================================================

// saveModelJSON saves model data to JSON file
func saveModelJSON(path string, data interface{}) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Marshal to JSON with pretty printing
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model data: %w", err)
	}

	// Write to file
	if err := os.WriteFile(path, jsonData, 0o644); err != nil {
		return fmt.Errorf("failed to write model file: %w", err)
	}

	logging.Infof("Saved model to %s (%d bytes)", path, len(jsonData))
	return nil
}

// loadModelJSON loads model data from JSON file
func loadModelJSON(path string, data interface{}) error {
	jsonData, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	if err := json.Unmarshal(jsonData, data); err != nil {
		return fmt.Errorf("failed to unmarshal model data: %w", err)
	}

	return nil
}

// GetDefaultModelsPath returns the default path for saved models
func GetDefaultModelsPath() string {
	return "models/model_selection"
}

// GetDefaultDataPath returns the default path for training data
func GetDefaultDataPath() string {
	return "data/model_selection"
}

// GetPretrainedPath returns the path to pre-trained models
func GetPretrainedPath() string {
	return "pretrained"
}

// ListPretrainedModels returns a list of available pre-trained models
func ListPretrainedModels(modelsPath string) ([]string, error) {
	var available []string

	algorithms := []string{"knn", "kmeans", "svm"}
	for _, alg := range algorithms {
		fileName := alg + "_model.json"
		modelPath := filepath.Join(modelsPath, fileName)
		if _, err := os.Stat(modelPath); err == nil {
			available = append(available, alg)
		}
	}

	return available, nil
}

// PretrainedModelInfo contains metadata about a pre-trained model
type PretrainedModelInfo struct {
	Algorithm       string   `json:"algorithm"`
	Version         string   `json:"version"`
	TrainingSamples int      `json:"training_samples"`
	Models          []string `json:"models"`
	FilePath        string   `json:"file_path"`
}

// GetPretrainedModelInfo returns metadata about a pre-trained model
func GetPretrainedModelInfo(algorithm, modelsPath string) (*PretrainedModelInfo, error) {
	var fileName string
	switch algorithm {
	case "knn":
		fileName = "knn_model.json"
	case "kmeans":
		fileName = "kmeans_model.json"
	case "svm":
		fileName = "svm_model.json"
	default:
		return nil, fmt.Errorf("unknown algorithm: %s (supported: knn, kmeans, svm)", algorithm)
	}

	modelPath := filepath.Join(modelsPath, fileName)

	// Read the JSON file header to get metadata
	data, err := os.ReadFile(modelPath)
	if err != nil {
		return nil, err
	}

	// Parse just the top-level fields
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	info := &PretrainedModelInfo{
		Algorithm: algorithm,
		FilePath:  modelPath,
	}

	// Extract version
	if v, ok := raw["version"]; ok {
		_ = json.Unmarshal(v, &info.Version)
	}

	// Extract training sample count
	if t, ok := raw["training"]; ok {
		var training []json.RawMessage
		_ = json.Unmarshal(t, &training)
		info.TrainingSamples = len(training)
	}

	// Extract model names from idx_to_model or cluster_models
	if m, ok := raw["idx_to_model"]; ok {
		_ = json.Unmarshal(m, &info.Models)
	} else if m, ok := raw["cluster_models"]; ok {
		_ = json.Unmarshal(m, &info.Models)
	}

	return info, nil
}
