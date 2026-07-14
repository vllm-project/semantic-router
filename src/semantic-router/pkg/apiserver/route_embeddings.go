//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Native entry points are package variables so admission and error-path tests
// can remain deterministic without loading models.
var (
	calculateEmbeddingSimilarityNative = candle_binding.CalculateEmbeddingSimilarityWithOptions
	calculateSimilarityBatchNative     = candle_binding.CalculateSimilarityBatchWithOptions
	validateImagesAfterAdmission       = validateEmbeddingImages
)

const (
	defaultEmbeddingDimension = 768
	defaultEmbeddingPriority  = 0.5
)

// handleEmbeddings handles embedding generation requests
func (s *ClassificationAPIServer) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parseEmbeddingRequest(w, r)
	if !ok {
		return
	}
	release, admitted := s.admitEmbeddingNative(w, r.Context())
	if !admitted {
		return
	}
	defer release()
	if code, message, valid := validateImagesAfterAdmission(req.Images); !valid {
		s.writeErrorResponse(w, embeddingValidationStatus(code), code, message)
		return
	}

	results, totalProcessingTime, err := buildEmbeddingResults(req)
	if err != nil {
		status, code, message := classifyEmbeddingError(err)
		s.writeErrorResponse(w, status, code, message)
		return
	}

	avgProcessingTime := averageEmbeddingProcessingTime(totalProcessingTime, req)
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
	if err := s.parseInferenceJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return EmbeddingRequest{}, false
	}

	applyEmbeddingDefaults(&req)
	mmbertPath := ""
	if s.config != nil {
		mmbertPath = s.config.EmbeddingModels.MmBertModelPath
	}
	availableLayers := config.MmBertAvailableLayers(mmbertPath)
	if code, message, ok := validateEmbeddingRequestShape(req, availableLayers); !ok {
		s.writeErrorResponse(w, embeddingValidationStatus(code), code, message)
		return EmbeddingRequest{}, false
	}

	return req, true
}

func averageEmbeddingProcessingTime(totalProcessingTime int64, req EmbeddingRequest) float64 {
	inputCount := len(req.Texts) + len(req.Images)
	if inputCount == 0 {
		return 0
	}
	return float64(totalProcessingTime) / float64(inputCount)
}

func applyEmbeddingDefaults(req *EmbeddingRequest) {
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimensionForRequest(*req)
	}
	if shouldDefaultEmbeddingPriorities(*req) {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func defaultEmbeddingDimensionForRequest(req EmbeddingRequest) int {
	switch {
	case isMixedEmbeddingRequest(req):
		return defaultMixedEmbeddingDimension
	case len(req.Images) > 0:
		return defaultImageEmbeddingDimension
	default:
		return defaultEmbeddingDimension
	}
}

func shouldDefaultEmbeddingPriorities(req EmbeddingRequest) bool {
	return len(req.Texts) > 0 && len(req.Images) == 0 && req.Model == "auto" &&
		nativeEmbeddingBackendCapabilities().autoSupportsPriorities &&
		req.QualityPriority == 0 && req.LatencyPriority == 0
}

// handleSimilarity handles text similarity calculation requests
func (s *ClassificationAPIServer) handleSimilarity(w http.ResponseWriter, r *http.Request) {
	var req SimilarityRequest
	if err := s.parseInferenceJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}

	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = 768 // Default to full dimension
	}
	if req.Model == "auto" && nativeEmbeddingBackendCapabilities().autoSupportsPriorities &&
		req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = 0.5
		req.LatencyPriority = 0.5
	}
	if code, message, ok := validateSimilarityRequest(req, s.mmBertAvailableLayers()); !ok {
		s.writeErrorResponse(w, embeddingValidationStatus(code), code, message)
		return
	}
	release, admitted := s.admitEmbeddingNative(w, r.Context())
	if !admitted {
		return
	}
	defer release()

	result, err := calculateEmbeddingSimilarityNative(
		req.Text1,
		req.Text2,
		candle_binding.SimilarityOptions{
			ModelType:       req.Model,
			TargetLayer:     req.TargetLayer,
			TargetDim:       req.Dimension,
			QualityPriority: req.QualityPriority,
			LatencyPriority: req.LatencyPriority,
		},
	)
	if err != nil {
		if errors.Is(err, candle_binding.ErrEmbeddingInputTooLong) {
			s.writeErrorResponse(w, http.StatusRequestEntityTooLarge, embeddingInputTooLargeCode,
				"embedding input exceeds the selected model context")
			return
		}
		logging.Errorf("Similarity calculation failed")
		s.writeErrorResponse(w, http.StatusInternalServerError, "SIMILARITY_CALCULATION_FAILED",
			"failed to calculate similarity")
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
	req, ok := s.parseBatchSimilarityRequest(w, r)
	if !ok {
		return
	}
	release, admitted := s.admitEmbeddingNative(w, r.Context())
	if !admitted {
		return
	}
	defer release()

	result, err := calculateSimilarityBatchNative(
		req.Query,
		req.Candidates,
		req.TopK,
		candle_binding.SimilarityOptions{
			ModelType:       req.Model,
			TargetLayer:     req.TargetLayer,
			TargetDim:       req.Dimension,
			QualityPriority: req.QualityPriority,
			LatencyPriority: req.LatencyPriority,
		},
	)
	if err != nil {
		if errors.Is(err, candle_binding.ErrEmbeddingInputTooLong) {
			s.writeErrorResponse(w, http.StatusRequestEntityTooLarge, embeddingInputTooLargeCode,
				"embedding input exceeds the selected model context")
			return
		}
		logging.Errorf("Batch similarity calculation failed")
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_FAILED",
			"failed to calculate batch similarity")
		return
	}

	matches, err := buildBatchSimilarityMatches(result, req.Candidates)
	if err != nil {
		logging.Errorf("Batch similarity returned an invalid result")
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_INVALID_RESULT",
			"batch similarity returned an invalid result")
		return
	}

	response := BatchSimilarityResponse{
		Matches:          matches,
		TotalCandidates:  len(req.Candidates),
		ModelUsed:        result.ModelType,
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	logging.Infof("Calculated batch similarity: %d candidates, top-%d matches (model: %s, took: %.2fms)",
		len(req.Candidates), len(matches), result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseBatchSimilarityRequest(w http.ResponseWriter, r *http.Request) (BatchSimilarityRequest, bool) {
	var req BatchSimilarityRequest
	if err := s.parseInferenceJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return BatchSimilarityRequest{}, false
	}

	applyBatchSimilarityDefaults(&req)
	if code, message, ok := validateBatchSimilarityRequest(req, s.mmBertAvailableLayers()); !ok {
		s.writeErrorResponse(w, embeddingValidationStatus(code), code, message)
		return BatchSimilarityRequest{}, false
	}
	normalizeBatchSimilarityLimit(&req)

	return req, true
}

func (s *ClassificationAPIServer) mmBertAvailableLayers() []int {
	modelPath := ""
	if s.config != nil {
		modelPath = s.config.EmbeddingModels.MmBertModelPath
	}
	return config.MmBertAvailableLayers(modelPath)
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
	if req.Model == "auto" && nativeEmbeddingBackendCapabilities().autoSupportsPriorities &&
		req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
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

// formatLayerList renders a layer set as a comma-separated string for error
// messages, e.g. [6 11 16 22] -> "6, 11, 16, 22".
func formatLayerList(layers []int) string {
	parts := make([]string, len(layers))
	for i, l := range layers {
		parts[i] = strconv.Itoa(l)
	}
	return strings.Join(parts, ", ")
}
