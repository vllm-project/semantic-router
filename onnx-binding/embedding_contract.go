//go:build !windows && cgo && (amd64 || arm64)
// +build !windows
// +build cgo
// +build amd64 arm64

package onnx_binding

/*
#cgo LDFLAGS: -L${SRCDIR}/target/release -lonnx_semantic_router -ldl -lm -lpthread
#include <stdbool.h>
#include <stdlib.h>

typedef struct {
    char* model_name;
    int native_dimension;
    int* supported_dimensions;
    int num_supported_dimensions;
    bool error;
} EmbeddingDimensionContractResult;

extern int get_embedding_dimension_contract(
    const char* model_type,
    EmbeddingDimensionContractResult* result
);
extern int get_multimodal_embedding_dimension_contract(
    EmbeddingDimensionContractResult* result
);
extern void free_embedding_dimension_contract(
    EmbeddingDimensionContractResult* result
);
*/
import "C"

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"
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

func normalizeEmbeddingModelType(modelType string) string {
	modelType = strings.ToLower(strings.TrimSpace(modelType))
	if modelType == "" {
		return "bert"
	}
	return modelType
}

// GetEmbeddingDimensionContract returns the contract for a loaded embedding
// model. The dimensions are read from the native ONNX model instance.
func GetEmbeddingDimensionContract(modelType string) (EmbeddingDimensionContract, error) {
	normalizedModelType := normalizeEmbeddingModelType(modelType)
	var result C.EmbeddingDimensionContractResult
	var status C.int

	if normalizedModelType == "multimodal" {
		status = C.get_multimodal_embedding_dimension_contract(&result)
	} else {
		cModelType := C.CString(normalizedModelType)
		defer C.free(unsafe.Pointer(cModelType))
		status = C.get_embedding_dimension_contract(cModelType, &result)
	}
	defer C.free_embedding_dimension_contract(&result)

	if status != 0 || bool(result.error) {
		return EmbeddingDimensionContract{}, fmt.Errorf(
			"failed to get embedding dimension contract for model %q (status: %d)",
			normalizedModelType,
			status,
		)
	}

	nativeDimension := int(result.native_dimension)
	if nativeDimension <= 0 {
		return EmbeddingDimensionContract{}, fmt.Errorf(
			"embedding model %q returned invalid native dimension %d",
			normalizedModelType,
			nativeDimension,
		)
	}

	contractModel := normalizedModelType
	if result.model_name != nil {
		if loadedModel := C.GoString(result.model_name); loadedModel != "" {
			contractModel = loadedModel
		}
	}

	numSupportedDimensions := int(result.num_supported_dimensions)
	if numSupportedDimensions < 0 {
		return EmbeddingDimensionContract{}, fmt.Errorf(
			"embedding model %q returned invalid supported dimension count %d",
			normalizedModelType,
			numSupportedDimensions,
		)
	}

	supportedDimensions := make([]int, 0, numSupportedDimensions+1)
	if numSupportedDimensions > 0 {
		if result.supported_dimensions == nil {
			return EmbeddingDimensionContract{}, fmt.Errorf(
				"embedding model %q returned a nil supported dimension list",
				normalizedModelType,
			)
		}
		dimensions := (*[1 << 30]C.int)(unsafe.Pointer(result.supported_dimensions))[:numSupportedDimensions:numSupportedDimensions]
		for _, dimension := range dimensions {
			if int(dimension) > 0 {
				supportedDimensions = append(supportedDimensions, int(dimension))
			}
		}
	}

	if len(supportedDimensions) == 0 {
		supportedDimensions = append(supportedDimensions, nativeDimension)
	}

	return EmbeddingDimensionContract{
		Model:               contractModel,
		NativeDimension:     nativeDimension,
		SupportedDimensions: supportedDimensions,
	}, nil
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
