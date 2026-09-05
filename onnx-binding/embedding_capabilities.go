package onnx_binding

import "errors"

const EmbeddingCapabilitiesVersionV1 uint32 = 1

type Backend string

const (
	BackendCandle Backend = "candle"
	BackendONNX   Backend = "onnx"
)

type ModelType string

const (
	ModelTypeQwen3      ModelType = "qwen3"
	ModelTypeGemma      ModelType = "gemma"
	ModelTypeMmBert     ModelType = "mmbert"
	ModelTypeMultimodal ModelType = "multimodal"

	DefaultEmbeddingModelType = ModelTypeMmBert
)

type Modality string

const (
	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
	ModalityAudio Modality = "audio"
)

type Device string

const (
	DeviceCPU   Device = "cpu"
	DeviceCUDA  Device = "cuda"
	DeviceROCm  Device = "rocm"
	DeviceMetal Device = "metal"
)

type EmbeddingCapabilities struct {
	Version             uint32
	Backend             Backend
	ModelType           ModelType
	SupportsBatching    bool
	Modalities          []Modality
	SupportedDimensions []int
	SupportedDevices    []Device
}

var (
	ErrBackendUnavailable    = errors.New("onnx: native backend unavailable for this build")
	ErrUnsupportedModelType  = errors.New("unsupported embedding model type")
	ErrMalformedCapabilities = errors.New("malformed embedding capabilities")
)
