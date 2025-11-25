//go:build !windows && cgo
// +build !windows,cgo

package openvino_binding

import (
	"fmt"
	"log"
	"runtime"
	"sync"
	"unsafe"
)

/*
#cgo CFLAGS: -I${SRCDIR}/cpp/include
#cgo LDFLAGS: -L${SRCDIR}/build -lopenvino_semantic_router -lstdc++ -lm
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/build

#include <stdlib.h>
#include <stdbool.h>
#include "openvino_semantic_router.h"
*/
import "C"

var (
	initOnce         sync.Once
	initErr          error
	modelInitialized bool

	classifierInitOnce sync.Once
	classifierInitErr  error

	embeddingInitOnce sync.Once
	embeddingInitErr  error

	tokenClassifierInitOnce sync.Once
	tokenClassifierInitErr  error
)

// ================================================================================================
// GO DATA STRUCTURES
// ================================================================================================

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

// EmbeddingOutput represents the complete embedding generation result with metadata
type EmbeddingOutput struct {
	Embedding        []float32 // The embedding vector
	ProcessingTimeMs float32   // Processing time in milliseconds
}

// SimilarityOutput represents the result of embedding similarity calculation
type SimilarityOutput struct {
	Similarity       float32 // Cosine similarity score (-1.0 to 1.0)
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
	ProcessingTimeMs float32                // Processing time in milliseconds
}

// ================================================================================================
// INITIALIZATION FUNCTIONS
// ================================================================================================

// InitModel initializes the BERT similarity model with the specified model path
//
// Parameters:
//   - modelPath: Path to OpenVINO IR model (.xml file)
//   - device: Device name ("CPU", "GPU", "AUTO", etc.)
//
// Returns:
//   - error: Non-nil if initialization fails
//
// Example:
//
//	err := InitModel("models/bert-base-uncased.xml", "CPU")
//	if err != nil {
//	    log.Fatal(err)
//	}
func InitModel(modelPath string, device string) error {
	var err error
	initOnce.Do(func() {
		if modelPath == "" {
			err = fmt.Errorf("model path cannot be empty")
			return
		}

		if device == "" {
			device = "CPU"
		}

		log.Printf("Initializing OpenVINO similarity model: %s on %s", modelPath, device)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_similarity_model(cModelPath, cDevice)
		if !bool(success) {
			err = fmt.Errorf("failed to initialize OpenVINO similarity model")
			return
		}

		modelInitialized = true
	})

	// Reset the once so we can try again if needed
	if err != nil {
		initOnce = sync.Once{}
		modelInitialized = false
	}

	return err
}

// IsModelInitialized returns whether the similarity model has been successfully initialized
func IsModelInitialized() bool {
	return bool(C.ov_is_similarity_model_initialized())
}

// InitClassifier initializes the BERT classifier with the specified model path and number of classes
//
// Parameters:
//   - modelPath: Path to OpenVINO IR model (.xml file)
//   - numClasses: Number of classification classes
//   - device: Device name ("CPU", "GPU", "AUTO", etc.)
//
// Returns:
//   - error: Non-nil if initialization fails
func InitClassifier(modelPath string, numClasses int, device string) error {
	var err error
	classifierInitOnce.Do(func() {
		if modelPath == "" {
			err = fmt.Errorf("model path cannot be empty")
			return
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		if device == "" {
			device = "CPU"
		}

		log.Printf("Initializing OpenVINO classifier: %s on %s with %d classes", modelPath, device, numClasses)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_classifier(cModelPath, C.int(numClasses), cDevice)
		if !bool(success) {
			err = fmt.Errorf("failed to initialize OpenVINO classifier")
			return
		}
	})

	classifierInitErr = err
	return err
}

// InitEmbeddingModel initializes the embedding model
//
// Parameters:
//   - modelPath: Path to OpenVINO IR model (.xml file)
//   - device: Device name ("CPU", "GPU", "AUTO", etc.)
//
// Returns:
//   - error: Non-nil if initialization fails
func InitEmbeddingModel(modelPath string, device string) error {
	var err error
	embeddingInitOnce.Do(func() {
		if modelPath == "" {
			err = fmt.Errorf("model path cannot be empty")
			return
		}

		if device == "" {
			device = "CPU"
		}

		log.Printf("Initializing OpenVINO embedding model: %s on %s", modelPath, device)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_embedding_model(cModelPath, cDevice)
		if !bool(success) {
			err = fmt.Errorf("failed to initialize OpenVINO embedding model")
			return
		}
	})

	embeddingInitErr = err
	return err
}

// IsEmbeddingModelInitialized returns whether the embedding model has been successfully initialized
func IsEmbeddingModelInitialized() bool {
	return bool(C.ov_is_embedding_model_initialized())
}

// InitTokenClassifier initializes the BERT token classifier
//
// Parameters:
//   - modelPath: Path to OpenVINO IR model (.xml file)
//   - numClasses: Number of token classes
//   - device: Device name ("CPU", "GPU", "AUTO", etc.)
//
// Returns:
//   - error: Non-nil if initialization fails
func InitTokenClassifier(modelPath string, numClasses int, device string) error {
	var err error
	tokenClassifierInitOnce.Do(func() {
		if modelPath == "" {
			err = fmt.Errorf("model path cannot be empty")
			return
		}

		if numClasses < 2 {
			err = fmt.Errorf("number of classes must be at least 2, got %d", numClasses)
			return
		}

		if device == "" {
			device = "CPU"
		}

		log.Printf("Initializing OpenVINO token classifier: %s on %s with %d classes", modelPath, device, numClasses)

		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_token_classifier(cModelPath, C.int(numClasses), cDevice)
		if !bool(success) {
			err = fmt.Errorf("failed to initialize OpenVINO token classifier")
			return
		}
	})

	tokenClassifierInitErr = err
	return err
}

// ================================================================================================
// TOKENIZATION FUNCTIONS
// ================================================================================================

// TokenizeText tokenizes the given text into tokens and their IDs with maxLength parameter
func TokenizeText(text string, maxLength int) (TokenizeResult, error) {
	if !IsModelInitialized() && !IsEmbeddingModelInitialized() {
		return TokenizeResult{}, fmt.Errorf("no model initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_tokenize_text(cText, C.int(maxLength))
	defer C.ov_free_tokenization_result(result)

	if bool(result.error) {
		return TokenizeResult{}, fmt.Errorf("failed to tokenize text")
	}

	tokenCount := int(result.token_count)
	tokenIDs := make([]int32, tokenCount)
	tokens := make([]string, tokenCount)

	if tokenCount > 0 && result.token_ids != nil {
		cTokenIDs := (*[1 << 30]C.int)(unsafe.Pointer(result.token_ids))[:tokenCount:tokenCount]
		for i := 0; i < tokenCount; i++ {
			tokenIDs[i] = int32(cTokenIDs[i])
		}
	}

	if tokenCount > 0 && result.tokens != nil {
		cTokens := (*[1 << 30]*C.char)(unsafe.Pointer(result.tokens))[:tokenCount:tokenCount]
		for i := 0; i < tokenCount; i++ {
			tokens[i] = C.GoString(cTokens[i])
		}
	}

	return TokenizeResult{
		TokenIDs: tokenIDs,
		Tokens:   tokens,
	}, nil
}

// TokenizeTextDefault tokenizes text with default max length (512)
func TokenizeTextDefault(text string) (TokenizeResult, error) {
	return TokenizeText(text, 512)
}

// ================================================================================================
// EMBEDDING FUNCTIONS
// ================================================================================================

// GetEmbedding gets the embedding vector for a text
func GetEmbedding(text string, maxLength int) ([]float32, error) {
	if !IsEmbeddingModelInitialized() {
		return nil, fmt.Errorf("embedding model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_get_text_embedding(cText, C.int(maxLength))

	if bool(result.error) {
		return nil, fmt.Errorf("failed to generate embedding")
	}

	length := int(result.length)
	embedding := make([]float32, length)

	if length > 0 && result.data != nil {
		cFloats := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]
		for i := 0; i < length; i++ {
			embedding[i] = float32(cFloats[i])
		}
		C.ov_free_embedding(result.data, result.length)
	}

	return embedding, nil
}

// GetEmbeddingDefault gets the embedding vector for a text with default max length (512)
func GetEmbeddingDefault(text string) ([]float32, error) {
	return GetEmbedding(text, 512)
}

// GetEmbeddingWithMetadata generates an embedding with full metadata
func GetEmbeddingWithMetadata(text string, maxLength int) (*EmbeddingOutput, error) {
	if !IsEmbeddingModelInitialized() {
		return nil, fmt.Errorf("embedding model not initialized")
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_get_text_embedding(cText, C.int(maxLength))

	if bool(result.error) {
		return nil, fmt.Errorf("failed to generate embedding")
	}

	length := int(result.length)
	embedding := make([]float32, length)

	if length > 0 && result.data != nil {
		cArray := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:length:length]
		for i := 0; i < length; i++ {
			embedding[i] = float32(cArray[i])
		}
		C.ov_free_embedding(result.data, result.length)
	}

	return &EmbeddingOutput{
		Embedding:        embedding,
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// ================================================================================================
// SIMILARITY FUNCTIONS
// ================================================================================================

// CalculateSimilarity calculates the similarity between two texts with maxLength parameter
func CalculateSimilarity(text1, text2 string, maxLength int) float32 {
	if !IsModelInitialized() && !IsEmbeddingModelInitialized() {
		log.Printf("No model initialized")
		return -1.0
	}

	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	result := C.ov_calculate_similarity(cText1, cText2, C.int(maxLength))
	return float32(result)
}

// CalculateSimilarityDefault calculates the similarity between two texts with default max length (512)
func CalculateSimilarityDefault(text1, text2 string) float32 {
	return CalculateSimilarity(text1, text2, 512)
}

// FindMostSimilar finds the most similar text from a list of candidates with maxLength parameter
func FindMostSimilar(query string, candidates []string, maxLength int) SimResult {
	if !IsModelInitialized() && !IsEmbeddingModelInitialized() {
		log.Printf("No model initialized")
		return SimResult{Index: -1, Score: -1.0}
	}

	if len(candidates) == 0 {
		return SimResult{Index: -1, Score: -1.0}
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	cCandidatesPtr := (**C.char)(unsafe.Pointer(&cCandidates[0]))

	result := C.ov_find_most_similar(cQuery, cCandidatesPtr, C.int(len(candidates)), C.int(maxLength))

	return SimResult{
		Index: int(result.index),
		Score: float32(result.score),
	}
}

// FindMostSimilarDefault finds the most similar text with default max length (512)
func FindMostSimilarDefault(query string, candidates []string) SimResult {
	return FindMostSimilar(query, candidates, 512)
}

// CalculateEmbeddingSimilarity calculates cosine similarity between two texts using embedding models
func CalculateEmbeddingSimilarity(text1, text2 string, maxLength int) (*SimilarityOutput, error) {
	if !IsEmbeddingModelInitialized() {
		return nil, fmt.Errorf("embedding model not initialized")
	}

	cText1 := C.CString(text1)
	defer C.free(unsafe.Pointer(cText1))

	cText2 := C.CString(text2)
	defer C.free(unsafe.Pointer(cText2))

	var result C.OVEmbeddingSimilarityResult
	status := C.ov_calculate_embedding_similarity(
		cText1,
		cText2,
		C.int(maxLength),
		&result,
	)

	if status != 0 || bool(result.error) {
		return nil, fmt.Errorf("failed to calculate similarity")
	}

	return &SimilarityOutput{
		Similarity:       float32(result.similarity),
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// CalculateSimilarityBatch finds top-k most similar candidates for a query
func CalculateSimilarityBatch(query string, candidates []string, topK int, maxLength int) (*BatchSimilarityOutput, error) {
	if !IsEmbeddingModelInitialized() && !IsModelInitialized() {
		return nil, fmt.Errorf("no model initialized")
	}

	if len(candidates) == 0 {
		return nil, fmt.Errorf("candidates array cannot be empty")
	}

	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))

	cCandidates := make([]*C.char, len(candidates))
	for i, candidate := range candidates {
		cCandidates[i] = C.CString(candidate)
		defer C.free(unsafe.Pointer(cCandidates[i]))
	}

	var result C.OVBatchSimilarityResult
	status := C.ov_calculate_similarity_batch(
		cQuery,
		(**C.char)(unsafe.Pointer(&cCandidates[0])),
		C.int(len(candidates)),
		C.int(topK),
		C.int(maxLength),
		&result,
	)

	if status != 0 || bool(result.error) {
		return nil, fmt.Errorf("failed to calculate batch similarity")
	}

	numMatches := int(result.num_matches)
	matches := make([]BatchSimilarityMatch, numMatches)

	if numMatches > 0 && result.matches != nil {
		matchesSlice := (*[1 << 30]C.OVSimilarityMatch)(unsafe.Pointer(result.matches))[:numMatches:numMatches]
		for i := 0; i < numMatches; i++ {
			matches[i] = BatchSimilarityMatch{
				Index:      int(matchesSlice[i].index),
				Similarity: float32(matchesSlice[i].similarity),
			}
		}
	}

	C.ov_free_batch_similarity_result(&result)

	return &BatchSimilarityOutput{
		Matches:          matches,
		ProcessingTimeMs: float32(result.processing_time_ms),
	}, nil
}

// ================================================================================================
// CLASSIFICATION FUNCTIONS
// ================================================================================================

// ClassifyText classifies the provided text and returns the predicted class and confidence
func ClassifyText(text string) (ClassResult, error) {
	if classifierInitErr != nil {
		return ClassResult{}, fmt.Errorf("classifier not initialized: %v", classifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_classify_text(cText)

	if result.predicted_class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text")
	}

	return ClassResult{
		Class:      int(result.predicted_class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyTextWithProbabilities classifies the provided text and returns the predicted class, confidence, and full probability distribution
func ClassifyTextWithProbabilities(text string) (ClassResultWithProbs, error) {
	if classifierInitErr != nil {
		return ClassResultWithProbs{}, fmt.Errorf("classifier not initialized: %v", classifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_classify_text_with_probabilities(cText)

	if result.predicted_class < 0 {
		return ClassResultWithProbs{}, fmt.Errorf("failed to classify text with probabilities")
	}

	probabilities := make([]float32, int(result.num_classes))
	if result.probabilities != nil && result.num_classes > 0 {
		probsSlice := (*[1 << 30]C.float)(unsafe.Pointer(result.probabilities))[:result.num_classes:result.num_classes]
		for i, prob := range probsSlice {
			probabilities[i] = float32(prob)
		}
		C.ov_free_probabilities(result.probabilities, result.num_classes)
	}

	return ClassResultWithProbs{
		Class:         int(result.predicted_class),
		Confidence:    float32(result.confidence),
		Probabilities: probabilities,
		NumClasses:    int(result.num_classes),
	}, nil
}

// ================================================================================================
// TOKEN CLASSIFICATION FUNCTIONS
// ================================================================================================

// ClassifyTokens performs token classification for PII detection
func ClassifyTokens(text string, id2labelJson string) (TokenClassificationResult, error) {
	if tokenClassifierInitErr != nil {
		return TokenClassificationResult{}, fmt.Errorf("token classifier not initialized: %v", tokenClassifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cId2Label := C.CString(id2labelJson)
	defer C.free(unsafe.Pointer(cId2Label))

	result := C.ov_classify_tokens(cText, cId2Label)
	defer C.ov_free_token_result(result)

	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify tokens")
	}

	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	numEntities := int(result.num_entities)
	entities := make([]TokenEntity, numEntities)

	cEntities := (*[1 << 20]C.OVTokenEntity)(unsafe.Pointer(result.entities))[:numEntities:numEntities]

	for i := 0; i < numEntities; i++ {
		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntities[i].entity_type),
			Start:      int(cEntities[i].start),
			End:        int(cEntities[i].end),
			Text:       C.GoString(cEntities[i].text),
			Confidence: float32(cEntities[i].confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// ================================================================================================
// MODERNBERT SUPPORT
// ================================================================================================

// ModernBERT-specific initialization and sync.Once variables
var (
	modernbertEmbeddingInitOnce sync.Once
	modernbertEmbeddingInitErr  error

	modernbertClassifierInitOnce sync.Once
	modernbertClassifierInitErr  error

	modernbertTokenClassifierInitOnce sync.Once
	modernbertTokenClassifierInitErr  error
)

// InitModernBertEmbedding initializes the ModernBERT embedding model (optimized BERT)
func InitModernBertEmbedding(modelPath string, device string) error {
	modernbertEmbeddingInitOnce.Do(func() {
		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_modernbert_embedding(cModelPath, cDevice)
		if !success {
			modernbertEmbeddingInitErr = fmt.Errorf("failed to initialize ModernBERT embedding model")
		} else {
			log.Printf("ModernBERT embedding model initialized: %s on %s", modelPath, device)
		}
	})
	return modernbertEmbeddingInitErr
}

// IsModernBertEmbeddingInitialized checks if ModernBERT embedding model is initialized
func IsModernBertEmbeddingInitialized() bool {
	return bool(C.ov_is_modernbert_embedding_initialized())
}

// InitModernBertClassifier initializes the ModernBERT classifier
func InitModernBertClassifier(modelPath string, numClasses int, device string) error {
	modernbertClassifierInitOnce.Do(func() {
		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_modernbert_classifier(cModelPath, C.int(numClasses), cDevice)
		if !success {
			modernbertClassifierInitErr = fmt.Errorf("failed to initialize ModernBERT classifier")
		} else {
			log.Printf("ModernBERT classifier initialized: %s on %s with %d classes", modelPath, device, numClasses)
		}
	})
	return modernbertClassifierInitErr
}

// IsModernBertClassifierInitialized checks if ModernBERT classifier is initialized
func IsModernBertClassifierInitialized() bool {
	return bool(C.ov_is_modernbert_classifier_initialized())
}

// InitModernBertTokenClassifier initializes the ModernBERT token classifier (for PII, NER, etc.)
func InitModernBertTokenClassifier(modelPath string, numClasses int, device string) error {
	modernbertTokenClassifierInitOnce.Do(func() {
		cModelPath := C.CString(modelPath)
		defer C.free(unsafe.Pointer(cModelPath))

		cDevice := C.CString(device)
		defer C.free(unsafe.Pointer(cDevice))

		success := C.ov_init_modernbert_token_classifier(cModelPath, C.int(numClasses), cDevice)
		if !success {
			modernbertTokenClassifierInitErr = fmt.Errorf("failed to initialize ModernBERT token classifier")
		} else {
			log.Printf("ModernBERT token classifier initialized: %s on %s with %d classes", modelPath, device, numClasses)
		}
	})
	return modernbertTokenClassifierInitErr
}

// IsModernBertTokenClassifierInitialized checks if ModernBERT token classifier is initialized
func IsModernBertTokenClassifierInitialized() bool {
	return bool(C.ov_is_modernbert_token_classifier_initialized())
}

// ClassifyModernBert performs text classification using ModernBERT
func ClassifyModernBert(text string) (ClassResult, error) {
	if modernbertClassifierInitErr != nil {
		return ClassResult{}, fmt.Errorf("ModernBERT classifier not initialized: %v", modernbertClassifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_classify_modernbert(cText)

	if result.predicted_class < 0 {
		return ClassResult{}, fmt.Errorf("failed to classify text with ModernBERT")
	}

	return ClassResult{
		Class:      int(result.predicted_class),
		Confidence: float32(result.confidence),
	}, nil
}

// ClassifyModernBertTokens performs token classification with BIO tagging using ModernBERT
func ClassifyModernBertTokens(text string, id2labelJson string) (TokenClassificationResult, error) {
	if modernbertTokenClassifierInitErr != nil {
		return TokenClassificationResult{}, fmt.Errorf("ModernBERT token classifier not initialized: %v", modernbertTokenClassifierInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	cId2Label := C.CString(id2labelJson)
	defer C.free(unsafe.Pointer(cId2Label))

	result := C.ov_classify_modernbert_tokens(cText, cId2Label)
	defer C.ov_free_token_result(result)

	if result.num_entities < 0 {
		return TokenClassificationResult{}, fmt.Errorf("failed to classify tokens with ModernBERT")
	}

	if result.num_entities == 0 {
		return TokenClassificationResult{Entities: []TokenEntity{}}, nil
	}

	numEntities := int(result.num_entities)
	entities := make([]TokenEntity, numEntities)

	cEntities := (*[1 << 20]C.OVTokenEntity)(unsafe.Pointer(result.entities))[:numEntities:numEntities]

	for i := 0; i < numEntities; i++ {
		entities[i] = TokenEntity{
			EntityType: C.GoString(cEntities[i].entity_type),
			Start:      int(cEntities[i].start),
			End:        int(cEntities[i].end),
			Text:       C.GoString(cEntities[i].text),
			Confidence: float32(cEntities[i].confidence),
		}
	}

	return TokenClassificationResult{
		Entities: entities,
	}, nil
}

// GetModernBertEmbedding generates an embedding using ModernBERT
func GetModernBertEmbedding(text string, maxLength int) ([]float32, error) {
	if modernbertEmbeddingInitErr != nil {
		return nil, fmt.Errorf("ModernBERT embedding model not initialized: %v", modernbertEmbeddingInitErr)
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	result := C.ov_get_modernbert_embedding(cText, C.int(maxLength))

	if result.error {
		return nil, fmt.Errorf("failed to get ModernBERT embedding")
	}

	if result.data == nil || result.length <= 0 {
		return nil, fmt.Errorf("invalid ModernBERT embedding result")
	}

	defer C.ov_free_embedding(result.data, result.length)

	embedding := make([]float32, int(result.length))
	embeddingSlice := (*[1 << 30]C.float)(unsafe.Pointer(result.data))[:result.length:result.length]
	for i, val := range embeddingSlice {
		embedding[i] = float32(val)
	}

	return embedding, nil
}

// ================================================================================================
// UTILITY FUNCTIONS
// ================================================================================================

// SetMemoryCleanupHandler sets up a finalizer to clean up memory when the Go GC runs
func SetMemoryCleanupHandler() {
	runtime.GC()
}

// GetVersion returns the OpenVINO version
func GetVersion() string {
	version := C.ov_get_version()
	return C.GoString(version)
}

// GetAvailableDevices returns a list of available devices
func GetAvailableDevices() []string {
	cDevices := C.ov_get_available_devices()
	if cDevices == nil {
		return []string{}
	}
	defer C.ov_free_cstring(cDevices)

	devicesStr := C.GoString(cDevices)
	if devicesStr == "" {
		return []string{}
	}

	// Split by comma
	var devices []string
	start := 0
	for i := 0; i < len(devicesStr); i++ {
		if devicesStr[i] == ',' {
			devices = append(devices, devicesStr[start:i])
			start = i + 1
		}
	}
	if start < len(devicesStr) {
		devices = append(devices, devicesStr[start:])
	}

	return devices
}
