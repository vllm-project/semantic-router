//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// handleEmbeddings handles embedding generation requests
func (s *ClassificationAPIServer) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req EmbeddingRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if len(req.Texts) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "texts array cannot be empty")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	if !isValidDimension(req.Dimension) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
		return
	}

	// Generate embeddings for each text
	results := make([]EmbeddingResult, 0, len(req.Texts))
	var totalProcessingTime int64

	for _, text := range req.Texts {
		var output *candle_binding.EmbeddingOutput
		var err error

		// Choose between manual model selection or automatic routing
		if req.Model == "auto" || req.Model == "" {
			// Automatic routing based on quality/latency priorities
			output, err = candle_binding.GetEmbeddingWithMetadata(
				text,
				req.QualityPriority,
				req.LatencyPriority,
				req.Dimension,
			)
		} else {
			// Manual model selection ("qwen3" or "gemma")
			output, err = candle_binding.GetEmbeddingWithModelType(
				text,
				req.Model,
				req.Dimension,
			)
		}

		if err != nil {
			s.writeErrorResponse(w, http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
				fmt.Sprintf("failed to generate embedding: %v", err))
			return
		}

		// Use metadata directly from Rust layer
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

	// Calculate statistics
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

// handleSimilarity handles text similarity calculation requests
func (s *ClassificationAPIServer) handleSimilarity(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req SimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if req.Text1 == "" || req.Text2 == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "both text1 and text2 must be provided")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	if !isValidDimension(req.Dimension) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
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

// handleBatchSimilarity handles batch similarity matching requests
func (s *ClassificationAPIServer) handleBatchSimilarity(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var req BatchSimilarityRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
		return
	}

	// Validate input
	if req.Query == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "query must be provided")
		return
	}
	if len(req.Candidates) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "candidates array cannot be empty")
		return
	}

	// Set defaults
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.TopK == 0 {
		req.TopK = len(req.Candidates) // Default to all candidates
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}

	// Validate dimension
	if !isValidDimension(req.Dimension) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_DIMENSION",
			fmt.Sprintf("dimension must be one of: 128, 256, 512, 768, 1024 (got %d)", req.Dimension))
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

	// Build response with matched text included
	matches := make([]BatchSimilarityMatch, len(result.Matches))
	for i, match := range result.Matches {
		matches[i] = BatchSimilarityMatch{
			Index:      match.Index,
			Similarity: match.Similarity,
			Text:       req.Candidates[match.Index],
		}
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

// isValidDimension checks if the provided dimension is valid
func isValidDimension(dim int) bool {
	validDimensions := map[int]bool{128: true, 256: true, 512: true, 768: true, 1024: true}
	return validDimensions[dim]
}
