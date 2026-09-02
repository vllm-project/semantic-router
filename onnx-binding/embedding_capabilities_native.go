//go:build !windows && cgo && (amd64 || arm64)

package onnx_binding

/*
#include <stdlib.h>
#include <stdint.h>

typedef struct {
    uint32_t version;
    uint32_t struct_size;
    uint32_t backend;
    uint32_t model_type;
    uint8_t supports_batching;
    uint8_t reserved[3];
    uint32_t modalities;
    uint32_t devices;
    const uint32_t* supported_dimensions;
    size_t num_supported_dimensions;
} EmbeddingCapabilitiesV1;

extern int32_t embedding_capabilities_v1(
    const uint8_t* model_type,
    size_t model_type_len,
    EmbeddingCapabilitiesV1* result
);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// EmbeddingCapabilitiesFor returns versioned static capabilities owned by the
// compiled ONNX runtime. It rejects legacy arbitrary-name fallback aliases.
func EmbeddingCapabilitiesFor(modelType string) (EmbeddingCapabilities, error) {
	modelTypeBytes := []byte(modelType)
	cModelType := C.CBytes(modelTypeBytes)
	if cModelType != nil {
		defer C.free(cModelType)
	}

	var result C.EmbeddingCapabilitiesV1
	status := int(C.embedding_capabilities_v1(
		(*C.uint8_t)(cModelType),
		C.size_t(len(modelTypeBytes)),
		&result,
	))
	switch status {
	case 0:
		return marshalEmbeddingCapabilities(result, BackendONNX)
	case 1:
		return EmbeddingCapabilities{}, fmt.Errorf("%w: %q", ErrUnsupportedModelType, modelType)
	case 2:
		return EmbeddingCapabilities{}, fmt.Errorf("%w: invalid model type encoding", ErrUnsupportedModelType)
	default:
		return EmbeddingCapabilities{}, fmt.Errorf("%w: unexpected native status %d", ErrMalformedCapabilities, status)
	}
}

func marshalEmbeddingCapabilities(result C.EmbeddingCapabilitiesV1, expectedBackend Backend) (EmbeddingCapabilities, error) {
	if err := validateEmbeddingCapabilitiesHeader(result); err != nil {
		return EmbeddingCapabilities{}, err
	}

	backend, ok := backendFromNative(uint32(result.backend))
	if !ok || backend != expectedBackend {
		return EmbeddingCapabilities{}, fmt.Errorf("%w: backend id %d", ErrMalformedCapabilities, uint32(result.backend))
	}
	modelType, ok := modelTypeFromNative(uint32(result.model_type))
	if !ok {
		return EmbeddingCapabilities{}, fmt.Errorf("%w: model type id %d", ErrMalformedCapabilities, uint32(result.model_type))
	}
	modalities, err := modalitiesFromNative(uint32(result.modalities))
	if err != nil {
		return EmbeddingCapabilities{}, err
	}
	devices, err := devicesFromNative(uint32(result.devices))
	if err != nil {
		return EmbeddingCapabilities{}, err
	}
	dimensions, err := dimensionsFromNative(result.supported_dimensions, uint64(result.num_supported_dimensions))
	if err != nil {
		return EmbeddingCapabilities{}, err
	}

	return EmbeddingCapabilities{
		Version:             uint32(result.version),
		Backend:             backend,
		ModelType:           modelType,
		SupportsBatching:    result.supports_batching != 0,
		Modalities:          modalities,
		SupportedDimensions: dimensions,
		SupportedDevices:    devices,
	}, nil
}

func validateEmbeddingCapabilitiesHeader(result C.EmbeddingCapabilitiesV1) error {
	if uint32(result.version) != EmbeddingCapabilitiesVersionV1 {
		return fmt.Errorf("%w: version %d", ErrMalformedCapabilities, uint32(result.version))
	}
	if uint32(result.struct_size) != uint32(C.sizeof_EmbeddingCapabilitiesV1) {
		return fmt.Errorf("%w: native struct size %d, Go struct size %d", ErrMalformedCapabilities, uint32(result.struct_size), uint32(C.sizeof_EmbeddingCapabilitiesV1))
	}
	if result.reserved[0] != 0 || result.reserved[1] != 0 || result.reserved[2] != 0 {
		return fmt.Errorf("%w: reserved bytes are non-zero", ErrMalformedCapabilities)
	}
	if result.supports_batching > 1 {
		return fmt.Errorf("%w: batching flag %d", ErrMalformedCapabilities, uint8(result.supports_batching))
	}
	return nil
}

func backendFromNative(value uint32) (Backend, bool) {
	switch value {
	case 1:
		return BackendCandle, true
	case 2:
		return BackendONNX, true
	default:
		return "", false
	}
}

func modelTypeFromNative(value uint32) (ModelType, bool) {
	switch value {
	case 1:
		return ModelTypeQwen3, true
	case 2:
		return ModelTypeGemma, true
	case 3:
		return ModelTypeMmBert, true
	case 4:
		return ModelTypeMultimodal, true
	default:
		return "", false
	}
}

func modalitiesFromNative(mask uint32) ([]Modality, error) {
	const knownMask = uint32(1 | 2 | 4)
	if mask == 0 || mask&^knownMask != 0 {
		return nil, fmt.Errorf("%w: modality mask %#x", ErrMalformedCapabilities, mask)
	}
	modalities := make([]Modality, 0, 3)
	if mask&1 != 0 {
		modalities = append(modalities, ModalityText)
	}
	if mask&2 != 0 {
		modalities = append(modalities, ModalityImage)
	}
	if mask&4 != 0 {
		modalities = append(modalities, ModalityAudio)
	}
	return modalities, nil
}

func devicesFromNative(mask uint32) ([]Device, error) {
	const knownMask = uint32(1 | 2 | 4 | 8)
	if mask == 0 || mask&^knownMask != 0 {
		return nil, fmt.Errorf("%w: device mask %#x", ErrMalformedCapabilities, mask)
	}
	devices := make([]Device, 0, 4)
	if mask&1 != 0 {
		devices = append(devices, DeviceCPU)
	}
	if mask&2 != 0 {
		devices = append(devices, DeviceCUDA)
	}
	if mask&4 != 0 {
		devices = append(devices, DeviceROCm)
	}
	if mask&8 != 0 {
		devices = append(devices, DeviceMetal)
	}
	return devices, nil
}

func dimensionsFromNative(values *C.uint32_t, count uint64) ([]int, error) {
	if count == 0 {
		if values != nil {
			return nil, fmt.Errorf("%w: dimensions pointer without values", ErrMalformedCapabilities)
		}
		return []int{}, nil
	}
	if values == nil || count > 1024 {
		return nil, fmt.Errorf("%w: invalid dimensions length %d", ErrMalformedCapabilities, count)
	}

	nativeValues := unsafe.Slice(values, int(count))
	dimensions := make([]int, len(nativeValues))
	for i, value := range nativeValues {
		if value == 0 {
			return nil, fmt.Errorf("%w: zero dimension at index %d", ErrMalformedCapabilities, i)
		}
		dimensions[i] = int(value)
	}
	return dimensions, nil
}
