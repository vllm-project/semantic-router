//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultEmbeddingDimension = 768
	defaultEmbeddingPriority  = 0.5
)

// invalidDimensionMessage matches the dimensions accepted by isValidDimension,
// including 64 (which the similarity error messages previously omitted).
const invalidDimensionMessage = "dimension must be one of: 64, 128, 256, 512, 768, 1024 (got %d)"

// validatePriority rejects a priority weight outside the documented [0.0, 1.0]
// range; out-of-range values were previously accepted and passed to the model.
func validatePriority(name string, value float32) (string, string, bool) {
	if value < 0 || value > 1 {
		return "INVALID_PARAMETER", fmt.Sprintf("%s must be between 0.0 and 1.0 (got %g)", name, value), false
	}
	return "", "", true
}

// handleEmbeddings handles embedding generation requests
func (s *ClassificationAPIServer) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parseEmbeddingRequest(w, r)
	if !ok {
		return
	}

	results, totalProcessingTime, err := buildEmbeddingResults(req)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
			fmt.Sprintf("failed to generate embedding: %v", err))
		return
	}

	avgProcessingTime := float64(totalProcessingTime) / float64(len(req.Texts))
	response := EmbeddingResponse{
		Embeddings:            results,
		TotalCount:            len(results),
		TotalProcessingTimeMs: totalProcessingTime,
		AvgProcessingTimeMs:   avgProcessingTime,
	}

	logging.Infof("Generated %d embeddings in %dms (avg: %.2fms)",
		len(results), totalProcessingTime, avgProcessingTime)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseEmbeddingRequest(w http.ResponseWriter, r *http.Request) (EmbeddingRequest, bool) {
	var req EmbeddingRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return EmbeddingRequest{}, false
	}

	applyEmbeddingDefaults(&req)
	if code, message, ok := validateEmbeddingRequest(req); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
		return EmbeddingRequest{}, false
	}

	return req, true
}

func applyEmbeddingDefaults(req *EmbeddingRequest) {
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimension
	}
	if req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func validateEmbeddingRequest(req EmbeddingRequest) (string, string, bool) {
	if len(req.Texts) == 0 {
		return "INVALID_INPUT", "texts array cannot be empty", false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf("dimension must be one of: 64, 128, 256, 512, 768, 1024 (got %d)", req.Dimension), false
	}
	if req.TargetLayer != 0 && req.Model != "mmbert" {
		return "INVALID_PARAMETER", "target_layer is only supported for model='mmbert'", false
	}
	if req.Model == "mmbert" && req.TargetLayer != 0 && !isValidMmBertLayer(req.TargetLayer) {
		return "INVALID_LAYER", fmt.Sprintf("target_layer must be one of: 3, 6, 11, 22 (got %d)", req.TargetLayer), false
	}
	return "", "", true
}

func buildEmbeddingResults(req EmbeddingRequest) ([]EmbeddingResult, int64, error) {
	results := make([]EmbeddingResult, 0, len(req.Texts))
	var totalProcessingTime int64

	for _, text := range req.Texts {
		output, err := embeddingOutput(req, text)
		if err != nil {
			return nil, 0, err
		}

		processingTime := int64(output.ProcessingTimeMs)
		results = append(results, EmbeddingResult{
			Text:             text,
			Embedding:        output.Embedding,
			Dimension:        len(output.Embedding),
			ModelUsed:        output.ModelType,
			ProcessingTimeMs: processingTime,
		})

		totalProcessingTime += processingTime
	}

	return results, totalProcessingTime, nil
}

func embeddingOutput(req EmbeddingRequest, text string) (*candle_binding.EmbeddingOutput, error) {
	switch req.Model {
	case "auto", "":
		return candle_binding.GetEmbeddingWithMetadata(text, req.QualityPriority, req.LatencyPriority, req.Dimension)
	case "mmbert":
		return candle_binding.GetEmbedding2DMatryoshka(text, req.Model, req.TargetLayer, req.Dimension)
	default:
		return candle_binding.GetEmbeddingWithModelType(text, req.Model, req.Dimension)
	}
}

// handleSimilarity handles text similarity calculation requests
func (s *ClassificationAPIServer) handleSimilarity(w http.ResponseWriter, r *http.Request) {
	var req SimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	applySimilarityDefaults(&req)
	if code, message, ok := validateSimilarityRequest(req); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
		return
	}

	// Calculate similarity
	result, err := candle_binding.CalculateEmbeddingSimilarity(
		req.Text1,
		req.Text2,
		req.Model,
		req.Dimension,
	)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "SIMILARITY_CALCULATION_FAILED",
			fmt.Sprintf("failed to calculate similarity: %v", err))
		return
	}

	response := SimilarityResponse{
		Similarity:       result.Similarity,
		ModelUsed:        result.ModelType,
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	logging.Infof("Calculated similarity: %.4f (model: %s, took: %.2fms)",
		result.Similarity, result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func applySimilarityDefaults(req *SimilarityRequest) {
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimension
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func validateSimilarityRequest(req SimilarityRequest) (string, string, bool) {
	if strings.TrimSpace(req.Text1) == "" || strings.TrimSpace(req.Text2) == "" {
		return "INVALID_INPUT", "both text1 and text2 must be provided", false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(invalidDimensionMessage, req.Dimension), false
	}
	if code, message, ok := validatePriority("quality_priority", req.QualityPriority); !ok {
		return code, message, false
	}
	if code, message, ok := validatePriority("latency_priority", req.LatencyPriority); !ok {
		return code, message, false
	}
	return "", "", true
}

// handleBatchSimilarity handles batch similarity matching requests
func (s *ClassificationAPIServer) handleBatchSimilarity(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parseBatchSimilarityRequest(w, r)
	if !ok {
		return
	}

	// Calculate batch similarity
	result, err := candle_binding.CalculateSimilarityBatch(
		req.Query,
		req.Candidates,
		req.TopK,
		req.Model,
		req.Dimension,
	)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_FAILED",
			fmt.Sprintf("failed to calculate batch similarity: %v", err))
		return
	}

	matches, err := buildBatchSimilarityMatches(result, req.Candidates)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_INVALID_RESULT", err.Error())
		return
	}

	response := BatchSimilarityResponse{
		Matches:          matches,
		TotalCandidates:  len(req.Candidates),
		ModelUsed:        result.ModelType,
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	logging.Infof("Calculated batch similarity: query='%s', %d candidates, top-%d matches (model: %s, took: %.2fms)",
		req.Query, len(req.Candidates), len(matches), result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseBatchSimilarityRequest(w http.ResponseWriter, r *http.Request) (BatchSimilarityRequest, bool) {
	var req BatchSimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return BatchSimilarityRequest{}, false
	}

	applyBatchSimilarityDefaults(&req)
	if code, message, ok := validateBatchSimilarityRequest(req); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
		return BatchSimilarityRequest{}, false
	}
	normalizeBatchSimilarityLimit(&req)

	return req, true
}

func applyBatchSimilarityDefaults(req *BatchSimilarityRequest) {
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimension
	}
	if req.TopK == 0 {
		req.TopK = len(req.Candidates)
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func validateBatchSimilarityRequest(req BatchSimilarityRequest) (string, string, bool) {
	if strings.TrimSpace(req.Query) == "" {
		return "INVALID_INPUT", "query must be provided", false
	}
	if len(req.Candidates) == 0 {
		return "INVALID_INPUT", "candidates array cannot be empty", false
	}
	for i, c := range req.Candidates {
		if strings.TrimSpace(c) == "" {
			return "INVALID_INPUT", fmt.Sprintf("candidates[%d] must not be empty or whitespace", i), false
		}
	}
	if req.TopK < 0 {
		return "INVALID_INPUT", "top_k cannot be negative", false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(invalidDimensionMessage, req.Dimension), false
	}
	if code, message, ok := validatePriority("quality_priority", req.QualityPriority); !ok {
		return code, message, false
	}
	if code, message, ok := validatePriority("latency_priority", req.LatencyPriority); !ok {
		return code, message, false
	}
	return "", "", true
}

func normalizeBatchSimilarityLimit(req *BatchSimilarityRequest) {
	if req.TopK > len(req.Candidates) {
		req.TopK = len(req.Candidates)
	}
}

func buildBatchSimilarityMatches(result *candle_binding.BatchSimilarityOutput, candidates []string) ([]BatchSimilarityMatch, error) {
	if result == nil {
		return nil, fmt.Errorf("batch similarity result is nil")
	}

	matches := make([]BatchSimilarityMatch, len(result.Matches))
	for i, match := range result.Matches {
		if match.Index < 0 || match.Index >= len(candidates) {
			return nil, fmt.Errorf("match index %d is out of range for %d candidates", match.Index, len(candidates))
		}
		matches[i] = BatchSimilarityMatch{
			Index:      match.Index,
			Similarity: match.Similarity,
			Text:       candidates[match.Index],
		}
	}
	return matches, nil
}

// isValidDimension checks if the provided dimension is valid
func isValidDimension(dim int) bool {
	validDimensions := map[int]bool{64: true, 128: true, 256: true, 512: true, 768: true, 1024: true}
	return validDimensions[dim]
}

// isValidMmBertLayer checks if the provided layer is valid for mmBERT early exit
func isValidMmBertLayer(layer int) bool {
	validLayers := map[int]bool{3: true, 6: true, 11: true, 22: true}
	return validLayers[layer]
}
