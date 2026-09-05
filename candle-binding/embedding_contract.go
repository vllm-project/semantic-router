package candle_binding

import (
	"fmt"
	"strconv"
	"strings"
)

// EmbeddingDimensionContract is the binding-owned dimension contract for one
// embedding model family. Consumers must use this contract instead of keeping
// their own model-to-dimension table.
//
// NativeDimension is returned when the caller omits a target dimension.
// SupportedDimensions contains every width that the loaded binding may return,
// including NativeDimension.
type EmbeddingDimensionContract struct {
	Model               string
	NativeDimension     int
	SupportedDimensions []int
}

// getEmbeddingDimensionContract is replaced by the native binding during
// package initialization. The default keeps compile-only builds fail-closed
// without pretending that a model contract is available.
var getEmbeddingDimensionContract = func(string) (EmbeddingDimensionContract, error) {
	return EmbeddingDimensionContract{}, ErrBackendUnavailable
}

func normalizeEmbeddingModelType(modelType string) string {
	modelType = strings.ToLower(strings.TrimSpace(modelType))
	if modelType == "" {
		return "bert"
	}
	return modelType
}

// GetEmbeddingDimensionContract returns the contract for a loaded embedding
// model. Native builds obtain it from the binding; builds without the native
// backend return ErrBackendUnavailable.
func GetEmbeddingDimensionContract(modelType string) (EmbeddingDimensionContract, error) {
	return getEmbeddingDimensionContract(normalizeEmbeddingModelType(modelType))
}

// ResolveEmbeddingDimension resolves and validates the dimension used by an
// embedding model. A non-positive requested dimension means "use the model
// native dimension". A positive requested dimension must be listed in the
// contract.
func ResolveEmbeddingDimension(modelType string, requestedDimension int) (int, error) {
	contract, err := GetEmbeddingDimensionContract(normalizeEmbeddingModelType(modelType))
	if err != nil {
		return 0, err
	}

	if requestedDimension <= 0 {
		return contract.NativeDimension, nil
	}

	for _, supportedDimension := range contract.SupportedDimensions {
		if requestedDimension == supportedDimension {
			return requestedDimension, nil
		}
	}

	supported := make([]string, len(contract.SupportedDimensions))
	for i, dimension := range contract.SupportedDimensions {
		supported[i] = strconv.Itoa(dimension)
	}
	return 0, fmt.Errorf(
		"embedding model %q does not support dimension %d (supported dimensions: %s)",
		contract.Model,
		requestedDimension,
		strings.Join(supported, ", "),
	)
}
