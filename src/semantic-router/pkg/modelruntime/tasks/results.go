// Package tasks defines inference results independently of model engines and
// router signals. Providers adapt their native or wire results at the boundary;
// recipe bindings supply label mappings and task-specific interpretation.
package tasks

// ClassResult is the historical top-1 classification result. A model reporting
// only this result does not imply a complete label distribution.
type ClassResult struct {
	Class      int
	Confidence float32
	Categories []string
}

// ClassResultWithProbs preserves top-1 information and any model-reported full
// distribution. A nil Probabilities slice means no distribution was supplied;
// consumers must not fill missing probabilities from Class or Confidence.
type ClassResultWithProbs struct {
	Class         int
	Confidence    float32
	Probabilities []float32
	NumClasses    int
}

// LabelDistribution contains a complete distribution in the label order of
// the prepared binding. It deliberately has no independently supplied argmax.
type LabelDistribution struct {
	Probabilities []float32
	Input         *InputUsage
}

// TokenEntity is a labeled span in the exact input text. Start and End are
// UTF-8 byte offsets, with End exclusive. Adapters convert other offset units
// once, before returning a result. Confidence is a probability in [0,1] only
// when the containing result explicitly declares ScoresAvailable=true.
type TokenEntity struct {
	EntityType  string
	Start       int
	End         int
	Text        string
	Confidence  float32
	Subtype     string
	Explanation string
}

// TokenClassificationResult retains valid spans even when the provider reports
// a partial scan. TruncatedAt, when known, is a UTF-8 byte boundary in the input.
// The accompanying partial-result error remains visible to the caller.
type TokenClassificationResult struct {
	Input       *InputUsage
	Entities    []TokenEntity
	TruncatedAt *int
	// Summary preserves a provider's aggregate span score when supplied.
	// Its semantics are explicit; an empty span list does not imply confidence 1.
	Summary          *ScoreResult
	SummarySemantics *ScoreSemantics
	// ScoresAvailable distinguishes reported probabilities from models that
	// select spans without scores. False means unavailable; nil means the
	// adapter has not declared score semantics. Neither implies probability 0.
	ScoresAvailable *bool
}

// HasScores reports whether every span carries a model-reported probability.
func (r TokenClassificationResult) HasScores() bool {
	return r.ScoresAvailable != nil && *r.ScoresAvailable
}

// GroundedTextRequest preserves the inputs of a grounding task independently
// from its shared token-spans result. Returned offsets refer to Answer, never
// to a flattened prompt containing the context and question.
type GroundedTextRequest struct {
	Context  string
	Question string
	Answer   string
}

// TextPairRequest retains the ordered premise and hypothesis of an NLI task.
type TextPairRequest struct {
	Premise    string
	Hypothesis string
}

// TokenLabelSet declares entity and outside labels for a task. Unlike PII's
// historical mapping, a general spans task need not reserve label index zero.
// A prepared adapter copies and validates these labels before serving calls.
type TokenLabelSet struct {
	Labels  []string
	Outside []string
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

func (r LabelDistribution) InputMetadata() *InputUsage         { return r.Input }
func (r TokenClassificationResult) InputMetadata() *InputUsage { return r.Input }
func (r EmbeddingResult) InputMetadata() *InputUsage           { return r.Input }
