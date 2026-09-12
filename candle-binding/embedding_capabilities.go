package candle_binding

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

	DefaultEmbeddingModelType = ModelTypeQwen3
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
	Version          uint32
	Backend          Backend
	ModelType        ModelType
	SupportsBatching bool
	Modalities       []Modality
	// DimensionState distinguishes an unloaded model from an observed contract.
	DimensionState DimensionState
	// NativeDimension is the loaded model's default width, independent of list order.
	NativeDimension int
	// SupportedDimensions includes NativeDimension and model-declared widths.
	// An empty list means dimensions are not yet available, never unrestricted.
	SupportedDimensions []int
	SupportedDevices    []Device
}

// DimensionState describes availability of model-resolved dimension facts.
// Other capability fields remain usable before model initialization.
type DimensionState string

const (
	DimensionStateNotLoaded DimensionState = "not_loaded"
	DimensionStateAvailable DimensionState = "available"
)

var (
	ErrUnsupportedModelType  = errors.New("unsupported embedding model type")
	ErrMalformedCapabilities = errors.New("malformed embedding capabilities")
)
