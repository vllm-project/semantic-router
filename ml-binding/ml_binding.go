// Package ml_binding provides Go bindings for Linfa-based ML algorithms.
//
// This package wraps Rust implementations of:
// - KNN (K-Nearest Neighbors) via linfa-nn
// - KMeans clustering via linfa-clustering
// - SVM (Support Vector Machine) via linfa-svm
//
// MLP and Matrix Factorization remain in Go (see modelselection package).
package ml_binding

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lml_semantic_router -lm -ldl -lpthread
#include <stdlib.h>
#include <stdint.h>

// KNN functions
void* ml_knn_new(int k);
void ml_knn_free(void* handle);
int ml_knn_train(void* handle, double* embeddings, size_t embedding_dim, char** labels, size_t num_records);
char* ml_knn_select(void* handle, double* query, size_t query_len);
int ml_knn_is_trained(void* handle);
char* ml_knn_to_json(void* handle);
void* ml_knn_from_json(char* json);

// KMeans functions
void* ml_kmeans_new(int num_clusters);
void ml_kmeans_free(void* handle);
int ml_kmeans_train(void* handle, double* embeddings, size_t embedding_dim, char** labels, double* qualities, int64_t* latencies, size_t num_records);
char* ml_kmeans_select(void* handle, double* query, size_t query_len);
int ml_kmeans_is_trained(void* handle);
char* ml_kmeans_to_json(void* handle);
void* ml_kmeans_from_json(char* json);

// SVM functions
void* ml_svm_new();
void ml_svm_free(void* handle);
int ml_svm_train(void* handle, double* embeddings, size_t embedding_dim, char** labels, size_t num_records);
char* ml_svm_select(void* handle, double* query, size_t query_len);
int ml_svm_is_trained(void* handle);
char* ml_svm_to_json(void* handle);
void* ml_svm_from_json(char* json);

// Memory management
void ml_free_string(char* ptr);
*/
import "C"

import (
	"errors"
	"sync"
	"unsafe"
)

// =============================================================================
// KNN Selector
// =============================================================================

// KNNSelector wraps the Linfa KNN implementation
type KNNSelector struct {
	handle unsafe.Pointer
	mu     sync.RWMutex
}

// NewKNNSelector creates a new KNN selector with the specified k value
func NewKNNSelector(k int) *KNNSelector {
	handle := C.ml_knn_new(C.int(k))
	if handle == nil {
		return nil
	}
	return &KNNSelector{handle: handle}
}

// Close releases the KNN selector resources
func (s *KNNSelector) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle != nil {
		C.ml_knn_free(s.handle)
		s.handle = nil
	}
}

// Train trains the KNN model with embeddings and labels
func (s *KNNSelector) Train(embeddings [][]float64, labels []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.handle == nil {
		return errors.New("selector not initialized")
	}
	if len(embeddings) == 0 || len(labels) == 0 {
		return errors.New("empty training data")
	}
	if len(embeddings) != len(labels) {
		return errors.New("embeddings and labels count mismatch")
	}

	embeddingDim := len(embeddings[0])
	numRecords := len(embeddings)

	// Flatten embeddings
	flatEmbeddings := make([]C.double, numRecords*embeddingDim)
	for i, emb := range embeddings {
		for j, v := range emb {
			flatEmbeddings[i*embeddingDim+j] = C.double(v)
		}
	}

	// Convert labels to C strings
	cLabels := make([]*C.char, numRecords)
	for i, label := range labels {
		cLabels[i] = C.CString(label)
		defer C.free(unsafe.Pointer(cLabels[i]))
	}

	result := C.ml_knn_train(
		s.handle,
		&flatEmbeddings[0],
		C.size_t(embeddingDim),
		&cLabels[0],
		C.size_t(numRecords),
	)

	if result != 0 {
		return errors.New("KNN training failed")
	}
	return nil
}

// Select selects the best model for a query embedding
func (s *KNNSelector) Select(query []float64) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	cQuery := make([]C.double, len(query))
	for i, v := range query {
		cQuery[i] = C.double(v)
	}

	result := C.ml_knn_select(s.handle, &cQuery[0], C.size_t(len(query)))
	if result == nil {
		return "", errors.New("KNN selection failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// IsTrained returns whether the model is trained
func (s *KNNSelector) IsTrained() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.handle == nil {
		return false
	}
	return C.ml_knn_is_trained(s.handle) != 0
}

// ToJSON serializes the model to JSON
func (s *KNNSelector) ToJSON() (string, error) {
	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	result := C.ml_knn_to_json(s.handle)
	if result == nil {
		return "", errors.New("JSON serialization failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// KNNFromJSON loads a KNN selector from JSON
func KNNFromJSON(json string) (*KNNSelector, error) {
	cJSON := C.CString(json)
	defer C.free(unsafe.Pointer(cJSON))

	handle := C.ml_knn_from_json(cJSON)
	if handle == nil {
		return nil, errors.New("failed to load KNN from JSON")
	}

	return &KNNSelector{handle: handle}, nil
}

// =============================================================================
// KMeans Selector
// =============================================================================

// KMeansSelector wraps the Linfa KMeans implementation
type KMeansSelector struct {
	handle unsafe.Pointer
	mu     sync.RWMutex
}

// NewKMeansSelector creates a new KMeans selector with the specified number of clusters
func NewKMeansSelector(numClusters int) *KMeansSelector {
	handle := C.ml_kmeans_new(C.int(numClusters))
	if handle == nil {
		return nil
	}
	return &KMeansSelector{handle: handle}
}

// Close releases the KMeans selector resources
func (s *KMeansSelector) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle != nil {
		C.ml_kmeans_free(s.handle)
		s.handle = nil
	}
}

// KMeansTrainingRecord represents a training record for KMeans
type KMeansTrainingRecord struct {
	Embedding []float64
	Label     string
	Quality   float64
	LatencyNs int64
}

// Train trains the KMeans model with training records
func (s *KMeansSelector) Train(records []KMeansTrainingRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.handle == nil {
		return errors.New("selector not initialized")
	}
	if len(records) == 0 {
		return errors.New("empty training data")
	}

	embeddingDim := len(records[0].Embedding)
	numRecords := len(records)

	// Flatten embeddings
	flatEmbeddings := make([]C.double, numRecords*embeddingDim)
	cLabels := make([]*C.char, numRecords)
	qualities := make([]C.double, numRecords)
	latencies := make([]C.int64_t, numRecords)

	for i, rec := range records {
		for j, v := range rec.Embedding {
			flatEmbeddings[i*embeddingDim+j] = C.double(v)
		}
		cLabels[i] = C.CString(rec.Label)
		defer C.free(unsafe.Pointer(cLabels[i]))
		qualities[i] = C.double(rec.Quality)
		latencies[i] = C.int64_t(rec.LatencyNs)
	}

	result := C.ml_kmeans_train(
		s.handle,
		&flatEmbeddings[0],
		C.size_t(embeddingDim),
		&cLabels[0],
		&qualities[0],
		&latencies[0],
		C.size_t(numRecords),
	)

	if result != 0 {
		return errors.New("KMeans training failed")
	}
	return nil
}

// Select selects the best model for a query embedding
func (s *KMeansSelector) Select(query []float64) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	cQuery := make([]C.double, len(query))
	for i, v := range query {
		cQuery[i] = C.double(v)
	}

	result := C.ml_kmeans_select(s.handle, &cQuery[0], C.size_t(len(query)))
	if result == nil {
		return "", errors.New("KMeans selection failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// IsTrained returns whether the model is trained
func (s *KMeansSelector) IsTrained() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.handle == nil {
		return false
	}
	return C.ml_kmeans_is_trained(s.handle) != 0
}

// ToJSON serializes the model to JSON
func (s *KMeansSelector) ToJSON() (string, error) {
	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	result := C.ml_kmeans_to_json(s.handle)
	if result == nil {
		return "", errors.New("JSON serialization failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// KMeansFromJSON loads a KMeans selector from JSON
func KMeansFromJSON(json string) (*KMeansSelector, error) {
	cJSON := C.CString(json)
	defer C.free(unsafe.Pointer(cJSON))

	handle := C.ml_kmeans_from_json(cJSON)
	if handle == nil {
		return nil, errors.New("failed to load KMeans from JSON")
	}

	return &KMeansSelector{handle: handle}, nil
}

// =============================================================================
// SVM Selector
// =============================================================================

// SVMSelector wraps the Linfa SVM implementation
type SVMSelector struct {
	handle unsafe.Pointer
	mu     sync.RWMutex
}

// NewSVMSelector creates a new SVM selector
func NewSVMSelector() *SVMSelector {
	handle := C.ml_svm_new()
	if handle == nil {
		return nil
	}
	return &SVMSelector{handle: handle}
}

// Close releases the SVM selector resources
func (s *SVMSelector) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle != nil {
		C.ml_svm_free(s.handle)
		s.handle = nil
	}
}

// Train trains the SVM model with embeddings and labels
func (s *SVMSelector) Train(embeddings [][]float64, labels []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.handle == nil {
		return errors.New("selector not initialized")
	}
	if len(embeddings) == 0 || len(labels) == 0 {
		return errors.New("empty training data")
	}
	if len(embeddings) != len(labels) {
		return errors.New("embeddings and labels count mismatch")
	}

	embeddingDim := len(embeddings[0])
	numRecords := len(embeddings)

	// Flatten embeddings
	flatEmbeddings := make([]C.double, numRecords*embeddingDim)
	for i, emb := range embeddings {
		for j, v := range emb {
			flatEmbeddings[i*embeddingDim+j] = C.double(v)
		}
	}

	// Convert labels to C strings
	cLabels := make([]*C.char, numRecords)
	for i, label := range labels {
		cLabels[i] = C.CString(label)
		defer C.free(unsafe.Pointer(cLabels[i]))
	}

	result := C.ml_svm_train(
		s.handle,
		&flatEmbeddings[0],
		C.size_t(embeddingDim),
		&cLabels[0],
		C.size_t(numRecords),
	)

	if result != 0 {
		return errors.New("SVM training failed")
	}
	return nil
}

// Select selects the best model for a query embedding
func (s *SVMSelector) Select(query []float64) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	cQuery := make([]C.double, len(query))
	for i, v := range query {
		cQuery[i] = C.double(v)
	}

	result := C.ml_svm_select(s.handle, &cQuery[0], C.size_t(len(query)))
	if result == nil {
		return "", errors.New("SVM selection failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// IsTrained returns whether the model is trained
func (s *SVMSelector) IsTrained() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.handle == nil {
		return false
	}
	return C.ml_svm_is_trained(s.handle) != 0
}

// ToJSON serializes the model to JSON
func (s *SVMSelector) ToJSON() (string, error) {
	if s.handle == nil {
		return "", errors.New("selector not initialized")
	}

	result := C.ml_svm_to_json(s.handle)
	if result == nil {
		return "", errors.New("JSON serialization failed")
	}
	defer C.ml_free_string(result)

	return C.GoString(result), nil
}

// SVMFromJSON loads an SVM selector from JSON
func SVMFromJSON(json string) (*SVMSelector, error) {
	cJSON := C.CString(json)
	defer C.free(unsafe.Pointer(cJSON))

	handle := C.ml_svm_from_json(cJSON)
	if handle == nil {
		return nil, errors.New("failed to load SVM from JSON")
	}

	return &SVMSelector{handle: handle}, nil
}
