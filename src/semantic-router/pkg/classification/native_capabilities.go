package classification

// NativeBackendCapabilities describes runtime-visible native backend features.
// It is intentionally owned by the router classification package so callers do
// not need to infer feature support from build tags or backend-specific stubs.
type NativeBackendCapabilities struct {
	Name                       string `json:"name"`
	UnifiedBatchClassification bool   `json:"unified_batch_classification"`
	LoRABatchClassification    bool   `json:"lora_batch_classification"`
	BatchedEmbedding           bool   `json:"batched_embedding"`
	MultimodalEmbedding        bool   `json:"multimodal_embedding"`
	ModalityRouting            bool   `json:"modality_routing"`
	MLPSelector                bool   `json:"mlp_selector"`
	ExplicitReset              bool   `json:"explicit_reset"`
}

// CurrentNativeBackendCapabilities returns the feature contract for the native
// backend selected by the current Go build tags.
func CurrentNativeBackendCapabilities() NativeBackendCapabilities {
	return nativeBackendCapabilities
}
