//go:build windows || !cgo || (!amd64 && !arm64)

package ml_binding

import "errors"

var errUnavailable = errors.New("ml-binding: native library unavailable on this platform")

// KNNSelector wraps the Linfa KNN implementation for inference.
type KNNSelector struct{}

// NewKNNSelector creates a new KNN selector with the specified k value.
func NewKNNSelector(int) *KNNSelector { return nil }

// Close releases the KNN selector resources.
func (*KNNSelector) Close() {}

// Select selects the best model for a query embedding.
func (*KNNSelector) Select([]float64) (string, error) { return "", errUnavailable }

// IsTrained returns whether the model has been loaded.
func (*KNNSelector) IsTrained() bool { return false }

// ToJSON serializes the model to JSON.
func (*KNNSelector) ToJSON() (string, error) { return "", errUnavailable }

// KNNFromJSON loads a KNN selector from JSON.
func KNNFromJSON(string) (*KNNSelector, error) { return nil, errUnavailable }

// KMeansSelector wraps the Linfa KMeans implementation for inference.
type KMeansSelector struct{}

// NewKMeansSelector creates a new KMeans selector with the specified number of clusters.
func NewKMeansSelector(int) *KMeansSelector { return nil }

// Close releases the KMeans selector resources.
func (*KMeansSelector) Close() {}

// Select selects the best model for a query embedding.
func (*KMeansSelector) Select([]float64) (string, error) { return "", errUnavailable }

// IsTrained returns whether the model has been loaded.
func (*KMeansSelector) IsTrained() bool { return false }

// ToJSON serializes the model to JSON.
func (*KMeansSelector) ToJSON() (string, error) { return "", errUnavailable }

// KMeansFromJSON loads a KMeans selector from JSON.
func KMeansFromJSON(string) (*KMeansSelector, error) { return nil, errUnavailable }

// SVMKernelType defines the kernel type for SVM.
type SVMKernelType int

const (
	// SVMKernelLinear uses linear kernel: f(x) = w·x - b
	SVMKernelLinear SVMKernelType = 0
	// SVMKernelRBF uses RBF (Gaussian) kernel: f(x) = Σ(αᵢ·exp(-γ||x-xᵢ||²))
	SVMKernelRBF SVMKernelType = 1
)

// SVMSelector wraps the Linfa SVM implementation for inference.
type SVMSelector struct{}

// NewSVMSelector creates a new SVM selector with default (RBF) kernel.
func NewSVMSelector() *SVMSelector { return nil }

// NewSVMSelectorWithKernel creates a new SVM selector with specified kernel.
func NewSVMSelectorWithKernel(SVMKernelType, float64) *SVMSelector { return nil }

// Close releases the SVM selector resources.
func (*SVMSelector) Close() {}

// Select selects the best model for a query embedding.
func (*SVMSelector) Select([]float64) (string, error) { return "", errUnavailable }

// IsTrained returns whether the model has been loaded.
func (*SVMSelector) IsTrained() bool { return false }

// ToJSON serializes the model to JSON.
func (*SVMSelector) ToJSON() (string, error) { return "", errUnavailable }

// SVMFromJSON loads an SVM selector from JSON.
func SVMFromJSON(string) (*SVMSelector, error) { return nil, errUnavailable }
