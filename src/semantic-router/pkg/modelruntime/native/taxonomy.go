package native

type Backend string
type Capability string
type Family string
type ArtifactFormat string
type Modality string

const (
	BackendCandle   Backend = "candle"
	BackendONNX     Backend = "onnxruntime"
	BackendOpenVINO Backend = "openvino"

	CapabilityEmbedding              Capability = "embedding"
	CapabilitySequenceClassification Capability = "sequence_classification"

	FamilyModernBERT Family = "modernbert"
	FamilyLlama      Family = "llama"
	FamilyQwen       Family = "qwen"
	FamilyGemma      Family = "gemma"

	ArtifactFormatSafetensors ArtifactFormat = "safetensors"
	ArtifactFormatONNX        ArtifactFormat = "onnx"
	ArtifactFormatOpenVINO    ArtifactFormat = "openvino"

	ModalityText  Modality = "text"
	ModalityImage Modality = "image"
)
