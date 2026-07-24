//go:build windows || !cgo

package candle_binding

// This file is the compile-only stub for the Candle backend. It is selected on
// Windows or whenever CGO is disabled, i.e. whenever the native Candle library
// cannot be linked. It deliberately does NOT emulate the native backend.
//
// Contract (see issue #2491): the stub fails closed. Every inference or
// mutation API returns a typed ErrBackendUnavailable error (or the neutral
// "unavailable" value for the few APIs without an error return) instead of a
// plausible synthetic success. This guarantees that a build without the native
// backend cannot be packaged as a working production router that returns fake
// safety, classification, similarity, or adapter results.

import "errors"

// ErrBackendUnavailable is returned by every inference and mutation API in the
// non-CGO build to signal that the native Candle backend is not linked. It is a
// typed sentinel so callers can detect the unavailable-backend condition with
// errors.Is.
var ErrBackendUnavailable = errors.New("candle: native backend unavailable (built without cgo)")

// TokenizeResult represents the result of tokenization
type TokenizeResult struct {
	TokenIDs []int32  // Token IDs
	Tokens   []string // String representation of tokens
}

// SimResult represents the result of a similarity search
type SimResult struct {
	Index int     // Index of the most similar text
	Score float32 // Similarity score
}

// ClassResult represents the result of a text classification
type ClassResult struct {
	Categories []string
	Class      int     // Class index
	Confidence float32 // Confidence score
}

// ClassResultWithProbs represents the result of a text classification with full probability distribution
type ClassResultWithProbs struct {
	Class         int       // Class index
	Confidence    float32   // Confidence score
	Probabilities []float32 // Full probability distribution
	NumClasses    int       // Number of classes
}

// TokenEntity represents a single detected entity in token classification
type TokenEntity struct {
	EntityType string  // Type of entity (e.g., "PERSON", "EMAIL", "PHONE")
	Start      int     // Start character position in original text
	End        int     // End character position in original text
	Text       string  // Actual entity text
	Confidence float32 // Confidence score (0.0 to 1.0)
}

// TokenClassificationResult represents the result of token classification
type TokenClassificationResult struct {
	Entities []TokenEntity // Array of detected entities
}

// LoRA Unified Classifier structures
type LoRAIntentResult struct {
	Category   string
	Confidence float32
}

type LoRAPIIResult struct {
	HasPII     bool
	PIITypes   []string
	Confidence float32
}

type LoRASecurityResult struct {
	IsJailbreak bool
	ThreatType  string
	Confidence  float32
}

type LoRABatchResult struct {
	IntentResults   []LoRAIntentResult
	PIIResults      []LoRAPIIResult
	SecurityResults []LoRASecurityResult
	BatchSize       int
	AvgConfidence   float32
}

// Qwen3LoRAResult represents the classification result from Qwen3 LoRA generative classifier
type Qwen3LoRAResult struct {
	ClassID       int
	Confidence    float32
	CategoryName  string
	Probabilities []float32
	NumCategories int
}

// SafetyClassificationResult represents the result of safety classification
type SafetyClassificationResult struct {
	SafetyLabel string   // "Safe", "Unsafe", or "Controversial"
	Categories  []string // List of detected categories
	RawOutput   string   // Raw model output
}

// EmbeddingOutput represents the complete embedding generation result with metadata
type EmbeddingOutput struct {
	Embedding        []float32 // The embedding vector
	ModelType        string    // Model used: "qwen3", "gemma", or "unknown"
	SequenceLength   int       // Sequence length in tokens
	ProcessingTimeMs float32   // Processing time in milliseconds
}

// SimilarityOutput represents the result of embedding similarity calculation
type SimilarityOutput struct {
	Similarity       float32 // Cosine similarity score (-1.0 to 1.0)
	ModelType        string  // Model used: "qwen3", "gemma", or "unknown"
	ProcessingTimeMs float32 // Processing time in milliseconds
}

// BatchSimilarityMatch represents a single match in batch similarity matching
type BatchSimilarityMatch struct {
	Index      int     // Index of the candidate in the input array
	Similarity float32 // Cosine similarity score
}

// BatchSimilarityOutput holds the result of batch similarity matching
type BatchSimilarityOutput struct {
	Matches          []BatchSimilarityMatch // Top-k matches, sorted by similarity (descending)
	ModelType        string                 // Model used: "qwen3", "gemma", or "unknown"
	ProcessingTimeMs float32                // Processing time in milliseconds
}

// ModelInfo represents information about a single embedding model
type ModelInfo struct {
	ModelName         string // "qwen3" or "gemma"
	IsLoaded          bool   // Whether the model is loaded
	MaxSequenceLength int    // Maximum sequence length
	DefaultDimension  int    // Default embedding dimension
	ModelPath         string // Model path
}

// ModelsInfoOutput holds information about all embedding models
type ModelsInfoOutput struct {
	Models []ModelInfo // Array of model information
}

// MultiModalEmbeddingOutput represents the result of a multi-modal embedding.
type MultiModalEmbeddingOutput struct {
	Embedding        []float32 // The embedding vector (384-dim by default)
	Modality         string    // "text", "image", or "audio"
	ProcessingTimeMs float32   // Processing time in milliseconds
}

// InitModel initializes the BERT model
func InitModel(modelID string, useCPU bool) error {
	return ErrBackendUnavailable
}

// TokenizeText tokenizes the given text
func TokenizeText(text string, maxLength int) (TokenizeResult, error) {
	return TokenizeResult{}, ErrBackendUnavailable
}

func TokenizeTextDefault(text string) (TokenizeResult, error) {
	return TokenizeResult{}, ErrBackendUnavailable
}

// GetEmbedding gets the embedding vector for a text
func GetEmbedding(text string, maxLength int) ([]float32, error) {
	return nil, ErrBackendUnavailable
}

func GetEmbeddingDefault(text string) ([]float32, error) {
	return nil, ErrBackendUnavailable
}

// GetEmbeddingSmart intelligently selects the optimal embedding model
func GetEmbeddingSmart(text string, qualityPriority, latencyPriority float32) ([]float32, error) {
	return nil, ErrBackendUnavailable
}

// InitEmbeddingModelsBatched initializes Qwen3 embedding model
func InitEmbeddingModelsBatched(qwen3ModelPath string, maxBatchSize int, maxWaitMs uint64, useCPU bool) error {
	return ErrBackendUnavailable
}

// GetEmbeddingBatched generates an embedding using the continuous batching model
func GetEmbeddingBatched(text string, modelType string, targetDim int) (*EmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// InitEmbeddingModels initializes Qwen3 and/or Gemma embedding models
func InitEmbeddingModels(qwen3ModelPath, gemmaModelPath string, mmBertModelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// GetEmbeddingWithDim generates an embedding with intelligent model selection
func GetEmbeddingWithDim(text string, qualityPriority, latencyPriority float32, targetDim int) ([]float32, error) {
	return nil, ErrBackendUnavailable
}

// GetEmbeddingWithMetadata generates an embedding with full metadata
func GetEmbeddingWithMetadata(text string, qualityPriority, latencyPriority float32, targetDim int) (*EmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// GetEmbeddingWithModelType generates an embedding with a manually specified model type
func GetEmbeddingWithModelType(text string, modelType string, targetDim int) (*EmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// InitMmBertEmbeddingModel initializes mmBERT embedding model
func InitMmBertEmbeddingModel(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitMultiModalEmbeddingModel initializes multi-modal embedding model
func InitMultiModalEmbeddingModel(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// MultiModalEncodeText encodes text using multi-modal model
func MultiModalEncodeText(text string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// MultiModalEncodeImage encodes image using multi-modal model
func MultiModalEncodeImage(pixelData []float32, height, width, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// MultiModalEncodeAudio encodes audio using multi-modal model
func MultiModalEncodeAudio(melData []float32, nMels, timeFrames, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// MultiModalEncodeImageFromBytes decodes image bytes and encodes to embedding
func MultiModalEncodeImageFromBytes(imageBytes []byte, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// MultiModalEncodeImageFromBase64 decodes a base64-encoded image and encodes to embedding
func MultiModalEncodeImageFromBase64(base64Str string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// MultiModalEncodeImageFromURL downloads and encodes an image from URL
func MultiModalEncodeImageFromURL(url string, targetDim int) (*MultiModalEmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// GetEmbedding2DMatryoshka generates an embedding using the 2D Matryoshka API
func GetEmbedding2DMatryoshka(text string, modelType string, targetLayer int, targetDim int) (*EmbeddingOutput, error) {
	return nil, ErrBackendUnavailable
}

// CalculateSimilarity calculates the similarity between two texts. Without the
// native backend it returns -1.0, an out-of-range sentinel that native code
// uses to mark a failed similarity computation.
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	return -1.0
}

func CalculateSimilarityDefault(text1, text2 string) float32 {
	return -1.0
}

// CalculateEmbeddingSimilarity calculates cosine similarity
func CalculateEmbeddingSimilarity(text1, text2 string, modelType string, targetDim int) (*SimilarityOutput, error) {
	return nil, ErrBackendUnavailable
}

// CalculateSimilarityBatch finds top-k most similar candidates
func CalculateSimilarityBatch(query string, candidates []string, topK int, modelType string, targetDim int) (*BatchSimilarityOutput, error) {
	return nil, ErrBackendUnavailable
}

// GetEmbeddingModelsInfo retrieves information about all loaded embedding models
func GetEmbeddingModelsInfo() (*ModelsInfoOutput, error) {
	return nil, ErrBackendUnavailable
}

// FindMostSimilar finds the most similar text. Without the native backend it
// returns the native "no match" sentinel result rather than a fake hit.
func FindMostSimilar(query string, candidates []string, maxLength int) SimResult {
	return SimResult{Index: -1, Score: -1.0}
}

func FindMostSimilarDefault(query string, candidates []string) SimResult {
	return SimResult{Index: -1, Score: -1.0}
}

// SetMemoryCleanupHandler sets up a finalizer. No-op without the native backend.
func SetMemoryCleanupHandler() {}

// IsModelInitialized returns whether the model has been successfully
// initialized. The stub is never initialized because no backend is linked.
func IsModelInitialized() (rustState bool, goState bool) {
	return false, false
}

// InitClassifier initializes the BERT classifier
func InitClassifier(modelPath string, numClasses int, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitPIIClassifier initializes the PII classifier
func InitPIIClassifier(modelPath string, numClasses int, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitJailbreakClassifier initializes the jailbreak classifier
func InitJailbreakClassifier(modelPath string, numClasses int, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyText classifies the provided text
func ClassifyText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyTextWithProbabilities classifies with probabilities
func ClassifyTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	return ClassResultWithProbs{}, ErrBackendUnavailable
}

// ClassifyPIIText classifies the provided text for PII
func ClassifyPIIText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyJailbreakText classifies the provided text for jailbreak
func ClassifyJailbreakText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitModernBertClassifier initializes ModernBERT
func InitModernBertClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitModernBertPIIClassifier initializes ModernBERT PII
func InitModernBertPIIClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitModernBertJailbreakClassifier initializes ModernBERT Jailbreak
func InitModernBertJailbreakClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// InitModernBertPIITokenClassifier initializes ModernBERT PII Token
func InitModernBertPIITokenClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyModernBertText classifies using ModernBERT
func ClassifyModernBertText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyModernBertTextWithProbabilities classifies using ModernBERT with probs
func ClassifyModernBertTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	return ClassResultWithProbs{}, ErrBackendUnavailable
}

// ClassifyModernBertPIIText classifies PII using ModernBERT
func ClassifyModernBertPIIText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyModernBertJailbreakText classifies Jailbreak using ModernBERT
func ClassifyModernBertJailbreakText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitDebertaJailbreakClassifier initializes DeBERTa
func InitDebertaJailbreakClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyDebertaJailbreakText classifies using DeBERTa
func ClassifyDebertaJailbreakText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyModernBertPIITokens classifies tokens using ModernBERT
func ClassifyModernBertPIITokens(text string, modelConfigPath string) (TokenClassificationResult, error) {
	return TokenClassificationResult{}, ErrBackendUnavailable
}

// InitBertTokenClassifier initializes BERT token classifier
func InitBertTokenClassifier(modelPath string, numClasses int, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyBertPIITokens classifies tokens using BERT
func ClassifyBertPIITokens(text string, id2labelJson string) (TokenClassificationResult, error) {
	return TokenClassificationResult{}, ErrBackendUnavailable
}

// ClassifyBertText classifies using BERT
func ClassifyBertText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitCandleBertClassifier initializes Candle BERT. Returns false because the
// native backend is unavailable.
func InitCandleBertClassifier(modelPath string, numClasses int, useCPU bool) bool {
	return false
}

// InitCandleBertTokenClassifier initializes Candle BERT Token. Returns false
// because the native backend is unavailable.
func InitCandleBertTokenClassifier(modelPath string, numClasses int, useCPU bool) bool {
	return false
}

// ClassifyCandleBertText classifies using Candle BERT
func ClassifyCandleBertText(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyCandleBertTokens classifies tokens using Candle BERT
func ClassifyCandleBertTokens(text string) (TokenClassificationResult, error) {
	return TokenClassificationResult{}, ErrBackendUnavailable
}

// ClassifyCandleBertTokensWithLabels classifies tokens using Candle BERT with labels
func ClassifyCandleBertTokensWithLabels(text string, id2labelJSON string) (TokenClassificationResult, error) {
	return TokenClassificationResult{}, ErrBackendUnavailable
}

// InitLoRAUnifiedClassifier initializes LoRA Unified Classifier
func InitLoRAUnifiedClassifier(intentModelPath, piiModelPath, securityModelPath, architecture string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyBatchWithLoRA performs batch classification using LoRA
func ClassifyBatchWithLoRA(texts []string) (LoRABatchResult, error) {
	return LoRABatchResult{}, ErrBackendUnavailable
}

// InitQwen3MultiLoRAClassifier initializes Qwen3 Multi-LoRA
func InitQwen3MultiLoRAClassifier(baseModelPath string) error {
	return ErrBackendUnavailable
}

// LoadQwen3LoRAAdapter loads a LoRA adapter
func LoadQwen3LoRAAdapter(adapterName, adapterPath string) error {
	return ErrBackendUnavailable
}

// ClassifyWithQwen3Adapter classifies using Qwen3 adapter
func ClassifyWithQwen3Adapter(text, adapterName string) (*Qwen3LoRAResult, error) {
	return nil, ErrBackendUnavailable
}

// GetQwen3LoadedAdapters returns loaded adapters
func GetQwen3LoadedAdapters() ([]string, error) {
	return nil, ErrBackendUnavailable
}

// ClassifyZeroShotQwen3 classifies zero-shot
func ClassifyZeroShotQwen3(text string, categories []string) (*Qwen3LoRAResult, error) {
	return nil, ErrBackendUnavailable
}

// InitQwen3Guard initializes Qwen3Guard
func InitQwen3Guard(modelPath string) error {
	return ErrBackendUnavailable
}

// ClassifyPromptSafety classifies prompt safety
func ClassifyPromptSafety(text string) (*SafetyClassificationResult, error) {
	return nil, ErrBackendUnavailable
}

// ClassifyResponseSafety classifies response safety
func ClassifyResponseSafety(text string) (*SafetyClassificationResult, error) {
	return nil, ErrBackendUnavailable
}

// GetGuardRawOutput gets raw guard output
func GetGuardRawOutput(text string, mode string) (string, error) {
	return "", ErrBackendUnavailable
}

// IsQwen3GuardInitialized checks initialization
func IsQwen3GuardInitialized() bool {
	return false
}

// IsQwen3MultiLoRAInitialized checks initialization
func IsQwen3MultiLoRAInitialized() bool {
	return false
}

// IsMmBert32KModel checks if a model is mmBERT-32K
func IsMmBert32KModel(configPath string) bool {
	return false
}

// InitMmBert32KIntentClassifier initializes mmBERT-32K intent classifier
func InitMmBert32KIntentClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyMmBert32KIntent classifies text with mmBERT-32K intent classifier
func ClassifyMmBert32KIntent(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitMmBert32KFactcheckClassifier initializes mmBERT-32K fact-check classifier
func InitMmBert32KFactcheckClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyMmBert32KFactcheck classifies text with mmBERT-32K fact-check classifier
func ClassifyMmBert32KFactcheck(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitMmBert32KJailbreakClassifier initializes mmBERT-32K jailbreak classifier
func InitMmBert32KJailbreakClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyMmBert32KJailbreak classifies text with mmBERT-32K jailbreak classifier
func ClassifyMmBert32KJailbreak(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// ClassifyMmBert32KJailbreakWithProbs classifies text with mmBERT-32K jailbreak
// classifier and returns the full probability distribution
func ClassifyMmBert32KJailbreakWithProbs(text string) (ClassResultWithProbs, error) {
	return ClassResultWithProbs{}, ErrBackendUnavailable
}

// InitMmBert32KFeedbackClassifier initializes mmBERT-32K feedback classifier
func InitMmBert32KFeedbackClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyMmBert32KFeedback classifies text with mmBERT-32K feedback classifier
func ClassifyMmBert32KFeedback(text string) (ClassResult, error) {
	return ClassResult{}, ErrBackendUnavailable
}

// InitMmBert32KPIIClassifier initializes mmBERT-32K PII classifier
func InitMmBert32KPIIClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyMmBert32KPII classifies text with mmBERT-32K PII classifier
func ClassifyMmBert32KPII(text string) ([]TokenEntity, error) {
	return nil, ErrBackendUnavailable
}

// NLI constants and types
type NLILabel int

const (
	NLIEntailment    NLILabel = 0
	NLINeutral       NLILabel = 1
	NLIContradiction NLILabel = 2
	NLIError         NLILabel = -1
)

// InitHallucinationModel initializes the hallucination model
func InitHallucinationModel(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// FactCheckResult represents the result of fact checking
type FactCheckResult struct {
	Class      int
	Label      NLILabel
	Confidence float32
}

// InitFactCheckClassifier initializes the fact check classifier
func InitFactCheckClassifier(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyFactCheckText classifies text for fact checking
func ClassifyFactCheckText(text string) (FactCheckResult, error) {
	return FactCheckResult{}, ErrBackendUnavailable
}

// FeedbackResult represents the result of feedback detection
type FeedbackResult struct {
	Class      int
	Label      string
	Confidence float32
}

// InitFeedbackDetector initializes the feedback detector
func InitFeedbackDetector(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// ClassifyFeedbackText classifies feedback text
func ClassifyFeedbackText(text string) (FeedbackResult, error) {
	return FeedbackResult{}, ErrBackendUnavailable
}

// DetectHallucinations detects hallucinations
func DetectHallucinations(text, context, modelPath string, threshold float32) (HallucinationResult, error) {
	return HallucinationResult{}, ErrBackendUnavailable
}

// InitNLIModel initializes NLI model
func InitNLIModel(modelPath string, useCPU bool) error {
	return ErrBackendUnavailable
}

// NLIResult represents NLI classification result
type NLIResult struct {
	LabelStr       string
	EntailmentProb float32
	NeutralProb    float32
	ContradictProb float32
	Label          NLILabel
	Confidence     float32
}

// ClassifyNLI classifies NLI
func ClassifyNLI(premise, hypothesis string) (NLIResult, error) {
	return NLIResult{}, ErrBackendUnavailable
}

// DetectHallucinationsWithNLI detects hallucinations using NLI
func DetectHallucinationsWithNLI(text, context, modelPath string, threshold float32) (HallucinationResult, error) {
	return HallucinationResult{}, ErrBackendUnavailable
}

// HallucinationResult represents the result of hallucination detection
type HallucinationResult struct {
	HasHallucination bool
	Confidence       float32
	Spans            []HallucinationSpan
}

// HallucinationSpan represents a detected hallucination span
type HallucinationSpan struct {
	Text                    string
	Start                   int
	End                     int
	HallucinationConfidence float32
	NLILabel                NLILabel
	NLILabelStr             string
	NLIConfidence           float32
	Severity                int
	Explanation             string
}
