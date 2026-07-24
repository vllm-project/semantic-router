package native

// Backend represents a target runtime provider (e.g., candle, onnxruntime, openvino).
type Backend string

const (
	BackendCandle   Backend = "candle"
	BackendONNX     Backend = "onnxruntime"
	BackendOpenVINO Backend = "openvino"
	BackendRemote   Backend = "remote"
)

// Capability defines what a model is capable of (e.g., embedding, sequence classification).
type Capability string

const (
	CapabilityEmbedding                Capability = "embedding"
	CapabilitySequenceClassification   Capability = "sequence_classification"
	CapabilityTokenClassification      Capability = "token_classification"
	CapabilityMultimodalEmbedding      Capability = "multimodal_embedding"
	CapabilityModalityRouting          Capability = "modality_routing"
	CapabilityReranking                Capability = "reranking"
	CapabilityGuard                    Capability = "guard"
	CapabilityGenerativeClassification Capability = "generative_classification"
	CapabilityLoRAAdapterServing       Capability = "lora_adapter_serving"
	CapabilityMLPSelector              Capability = "mlp_selector"
)

// Family represents the architecture of the model.
type Family string

const (
	FamilyBERT       Family = "BERT"
	FamilyModernBERT Family = "ModernBERT"
	FamilymmBERT     Family = "mmBERT"
	FamilyQwen3      Family = "Qwen3"
	FamilyGemma      Family = "Gemma"
	FamilyMiniLM     Family = "MiniLM"
	FamilyQwen3Guard Family = "Qwen3Guard"
)

// ArtifactFormat specifies the on-disk or in-memory format.
type ArtifactFormat string

const (
	ArtifactFullModel  ArtifactFormat = "full_model"
	ArtifactMergedLoRA ArtifactFormat = "merged_lora"
	ArtifactAdapter    ArtifactFormat = "adapter"
	ArtifactONNXIR     ArtifactFormat = "onnx_ir"
	ArtifactOpenVINOIR ArtifactFormat = "openvino_ir"
	ArtifactQuantized  ArtifactFormat = "quantized"
)

// Modality specifies the data type the model operates on.
type Modality string

const (
	ModalityText       Modality = "text"
	ModalityImage      Modality = "image"
	ModalityAudio      Modality = "audio"
	ModalityMultimodal Modality = "multimodal"
)
