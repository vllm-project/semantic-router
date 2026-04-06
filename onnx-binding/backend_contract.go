package onnx_binding

import "fmt"

type NativeBackend string

const (
	NativeBackendCandle NativeBackend = "candle"
	NativeBackendONNX   NativeBackend = "onnx"
)

type BackendFeature string

const (
	FeatureEmbeddings            BackendFeature = "embeddings"
	FeatureUnifiedClassification BackendFeature = "unified_classification"
	FeatureLoRABatchInference    BackendFeature = "lora_batch_inference"
	FeatureMultimodalEmbeddings  BackendFeature = "multimodal_embeddings"
	FeatureExplicitRuntimeReset  BackendFeature = "explicit_runtime_reset"
)

type EmbeddingFamily string

const (
	EmbeddingFamilyQwen3  EmbeddingFamily = "qwen3"
	EmbeddingFamilyGemma  EmbeddingFamily = "gemma"
	EmbeddingFamilyMMBert EmbeddingFamily = "mmbert"
)

type ReloadSemantics string

const (
	ReloadRequiresProcessRestart ReloadSemantics = "process_restart_required"
)

type EmbeddingContract struct {
	SupportedFamilies  []EmbeddingFamily
	BatchedFamilies    []EmbeddingFamily
	SupportsMultimodal bool
}

func (c EmbeddingContract) SupportsFamily(family EmbeddingFamily) bool {
	for _, supported := range c.SupportedFamilies {
		if supported == family {
			return true
		}
	}
	return false
}

func (c EmbeddingContract) SupportsBatchedFamily(family EmbeddingFamily) bool {
	for _, supported := range c.BatchedFamilies {
		if supported == family {
			return true
		}
	}
	return false
}

type UnifiedClassificationContract struct {
	Supported                  bool
	SupportsLoRABatchInference bool
}

type LifecycleContract struct {
	SupportsBestEffortCleanup bool
	SupportsExplicitReset     bool
	ReloadSemantics           ReloadSemantics
}

type BackendContract struct {
	Backend               NativeBackend
	Embedding             EmbeddingContract
	UnifiedClassification UnifiedClassificationContract
	Lifecycle             LifecycleContract
}

type UnsupportedFeatureError struct {
	Backend NativeBackend
	Feature BackendFeature
}

func (err UnsupportedFeatureError) Error() string {
	return fmt.Sprintf("%s backend does not support %s", err.Backend, err.Feature)
}

type UnsupportedEmbeddingFamilyError struct {
	Backend NativeBackend
	Family  EmbeddingFamily
	Batched bool
}

func (err UnsupportedEmbeddingFamilyError) Error() string {
	if err.Batched {
		return fmt.Sprintf("%s backend does not support batched %s embeddings", err.Backend, err.Family)
	}
	return fmt.Sprintf("%s backend does not support %s embeddings", err.Backend, err.Family)
}

var currentBackendContract = BackendContract{
	Backend: NativeBackendONNX,
	Embedding: EmbeddingContract{
		SupportedFamilies:  []EmbeddingFamily{EmbeddingFamilyMMBert},
		BatchedFamilies:    []EmbeddingFamily{EmbeddingFamilyMMBert},
		SupportsMultimodal: true,
	},
	UnifiedClassification: UnifiedClassificationContract{
		Supported:                  false,
		SupportsLoRABatchInference: false,
	},
	Lifecycle: LifecycleContract{
		SupportsBestEffortCleanup: false,
		SupportsExplicitReset:     false,
		ReloadSemantics:           ReloadRequiresProcessRestart,
	},
}

func CurrentBackendContract() BackendContract {
	return currentBackendContract
}

func (c BackendContract) SupportsFeature(feature BackendFeature) bool {
	switch feature {
	case FeatureEmbeddings:
		return len(c.Embedding.SupportedFamilies) > 0
	case FeatureUnifiedClassification:
		return c.UnifiedClassification.Supported
	case FeatureLoRABatchInference:
		return c.UnifiedClassification.SupportsLoRABatchInference
	case FeatureMultimodalEmbeddings:
		return c.Embedding.SupportsMultimodal
	case FeatureExplicitRuntimeReset:
		return c.Lifecycle.SupportsExplicitReset
	default:
		return false
	}
}

func (c BackendContract) RequireFeature(feature BackendFeature) error {
	if c.SupportsFeature(feature) {
		return nil
	}
	return UnsupportedFeatureError{Backend: c.Backend, Feature: feature}
}

func (c BackendContract) RequireEmbeddingFamily(family EmbeddingFamily) error {
	if c.Embedding.SupportsFamily(family) {
		return nil
	}
	return UnsupportedEmbeddingFamilyError{Backend: c.Backend, Family: family}
}

func (c BackendContract) RequireBatchedEmbeddingFamily(family EmbeddingFamily) error {
	if c.Embedding.SupportsBatchedFamily(family) {
		return nil
	}
	return UnsupportedEmbeddingFamilyError{Backend: c.Backend, Family: family, Batched: true}
}
