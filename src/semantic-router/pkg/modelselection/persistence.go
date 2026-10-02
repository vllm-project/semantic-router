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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

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
