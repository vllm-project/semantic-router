// Package tasks defines inference results independently of model engines and
// router signals. Providers adapt their native or wire results at the boundary;
// recipe bindings supply label mappings and task-specific interpretation.
package tasks

// TextPairRequest retains the ordered premise and hypothesis of an NLI task.
type TextPairRequest struct {
	Premise    string
	Hypothesis string
}

// EmbeddingResult preserves the vector and execution metadata supplied by the
// model. Dimension, pooling, normalization and modality belong to the prepared
// binding's capability; missing metadata must not be guessed from ModelType.
type EmbeddingResult struct {
	Input            *InputUsage
	Embedding        []float32
	ModelType        string
	SequenceLength   int
	ProcessingTimeMs float32
}

// InputUsage records tokenizer facts, including templates and special tokens.
// A nil usage means unavailable, rather than a guessed character-based count.
type InputUsage struct {
	OriginalTokens  int
	ProcessedTokens int
	Truncated       bool
}

func (r EmbeddingResult) InputMetadata() *InputUsage { return r.Input }
