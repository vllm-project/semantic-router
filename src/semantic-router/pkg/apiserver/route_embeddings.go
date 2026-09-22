//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// mediaEncodeError locates a failed media item without echoing its payload.
type mediaEncodeError struct {
	modality string
	index    int
	err      error
}

func (e *mediaEncodeError) Error() string {
	return fmt.Sprintf("%s[%d]: %v", e.modality, e.index, e.err)
}
func (e *mediaEncodeError) Unwrap() error { return e.err }

// classifyEmbeddingError maps an owned inference error to the HTTP status,
// error code, and client message. Model input limits and input-caused image
// failures are client errors; other inference failures remain internal errors.
func classifyEmbeddingError(err error) (int, string, string) {
	inputError := errors.Is(err, binding.ErrInvalidInput) || errors.Is(err, binding.ErrInputLimit) || errors.Is(err, binding.ErrCapability)
	var mediaErr *mediaEncodeError
	if inputError && errors.As(err, &mediaErr) {
		return http.StatusBadRequest, "INVALID_" + strings.ToUpper(mediaErr.modality), fmt.Sprintf("%s[%d] could not be encoded", mediaErr.modality, mediaErr.index)
	}
	if inputError {
		return http.StatusBadRequest, "INVALID_INPUT", err.Error()
	}

	return http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
		fmt.Sprintf("failed to generate embedding: %v", err)
}

const (
	defaultEmbeddingDimension = 0
	defaultEmbeddingPriority  = 0.5
	// maxImagesPerRequest bounds images per request; each is a full SigLIP
	// forward pass and the body-size cap alone admits very many minimal images.
	maxImagesPerRequest = 8
	maxAudiosPerRequest = 8
)

// Shape validation is independent of the selected provider's capabilities.
const invalidDimensionMessage = "dimension must be a nonnegative int32 supported by the selected model (got %d)"

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
	var request EmbeddingRequest
	if err := s.parseJSONRequest(r, &request); err != nil {
		s.writeJSONRequestError(w, err)
		return
	}
	cfg, prepared, release, prepareErr := s.acquireEmbeddingRuntimeForRecipe(request.Recipe)
	defer release()
	req, ok := s.prepareEmbeddingRequest(w, request, prepared)
	if !ok {
		return
	}
	if prepareErr != nil {
		s.writeEmbeddingRuntimeError(w, prepareErr)
		return
	}
	results, totalProcessingTime, err := buildOwnedEmbeddingResults(r.Context(), prepared, req)
	if err != nil {
		status, code, message := classifyEmbeddingError(err)
		s.writeErrorResponse(w, status, code, message)
		return
	}

	avgProcessingTime := averageEmbeddingProcessingTime(totalProcessingTime, req)
	response := EmbeddingResponse{
		Recipe:                embeddingRecipeName(req.Recipe, cfg),
		Embeddings:            results,
		TotalCount:            len(results),
		TotalProcessingTimeMs: totalProcessingTime,
		AvgProcessingTimeMs:   avgProcessingTime,
	}

	logging.Infof("Generated %d embeddings in %dms (avg: %.2fms)",
		len(results), totalProcessingTime, avgProcessingTime)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) prepareEmbeddingRequest(w http.ResponseWriter, req EmbeddingRequest, prepared *embedding.Set) (EmbeddingRequest, bool) {
	applyEmbeddingDefaults(&req)
	if prepared != nil {
		selected, err := selectOwnedEmbeddingMediaModel(prepared, req)
		if err != nil {
			status, code, message := classifyEmbeddingError(err)
			s.writeErrorResponse(w, status, code, message)
			return EmbeddingRequest{}, false
		}
		req = selected
	}
	var availableLayers []int
	for _, model := range prepared.Models() {
		if model.Name == req.Model {
			availableLayers = model.Layers
			break
		}
	}
	if code, message, ok := validateEmbeddingRequest(req, availableLayers); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
		return EmbeddingRequest{}, false
	}
	return req, true
}

func averageEmbeddingProcessingTime(totalProcessingTime int64, req EmbeddingRequest) float64 {
	inputCount := len(req.Texts) + len(req.Images) + len(req.Audios)
	if inputCount == 0 {
		return 0
	}
	return float64(totalProcessingTime) / float64(inputCount)
}

func applyEmbeddingDefaults(req *EmbeddingRequest) {
	req.Model = strings.ToLower(strings.TrimSpace(req.Model))
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

func validateEmbeddingRequest(req EmbeddingRequest, availableLayers []int) (string, string, bool) {
	if len(req.Texts) == 0 && len(req.Images) == 0 && len(req.Audios) == 0 {
		return "INVALID_INPUT", "at least one of texts, images or audios must be provided", false
	}
	if code, message, ok := validateEmbeddingImages(req.Images); !ok {
		return code, message, false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(invalidDimensionMessage, req.Dimension), false
	}
	if req.TargetLayer < 0 || req.TargetLayer > math.MaxInt32 || (req.TargetLayer > 0 && req.Model != "auto" && !slices.Contains(availableLayers, req.TargetLayer)) {
		return "INVALID_LAYER", fmt.Sprintf("target_layer must be 0 or one of the loaded model layers: %s (got %d)", formatLayerList(availableLayers), req.TargetLayer), false
	}
	if len(req.Audios) > maxAudiosPerRequest {
		return "INVALID_INPUT", "at most 8 audios may be provided per request", false
	}
	for index, audio := range req.Audios {
		if _, err := embedding.DecodeAudio(audio); err != nil {
			return "INVALID_AUDIO", fmt.Sprintf("audios[%d]: %v", index, err), false
		}
	}

	return "", "", true
}

// validateEmbeddingImages enforces the image-input contract: a bounded count of
// safe inline base64 image data URIs whose payloads decode.
func validateEmbeddingImages(images []string) (string, string, bool) {
	if len(images) > maxImagesPerRequest {
		return "INVALID_INPUT", fmt.Sprintf("at most %d images may be provided per request (got %d)", maxImagesPerRequest, len(images)), false
	}
	for i, image := range images {
		if !imageurl.IsSafeImageDataURL(image) {
			return "INVALID_IMAGE", fmt.Sprintf("images[%d] must be an inline base64 image data URI (data:image/<type>;base64,...)", i), false
		}
		if _, ok := imageurl.DecodeBase64(image); !ok {
			return "INVALID_IMAGE", fmt.Sprintf("images[%d] is not valid base64-encoded image data", i), false
		}
	}
	return "", "", true
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

	cfg, prepared, release, err := s.acquireEmbeddingRuntimeForRecipe(req.Recipe)
	defer release()
	if err != nil {
		s.writeEmbeddingRuntimeError(w, err)
		return
	}
	start := time.Now()
	request := EmbeddingRequest{Model: req.Model, Dimension: req.Dimension, TargetLayer: req.TargetLayer, QualityPriority: req.QualityPriority, LatencyPriority: req.LatencyPriority}
	first, err := ownedEmbeddingOutput(r.Context(), prepared, request, req.Text1)
	request.Model = first.ModelUsed
	var score float32
	if err == nil {
		second, otherErr := ownedEmbeddingOutput(r.Context(), prepared, request, req.Text2)
		err = otherErr
		if err == nil {
			score, err = embeddingCosine(first.Embedding, second.Embedding)
		}
	}
	result := SimilarityResponse{Recipe: embeddingRecipeName(req.Recipe, cfg), Similarity: score, ModelUsed: first.ModelUsed, ProcessingTimeMs: float32(time.Since(start).Microseconds()) / 1000}

	if err != nil {
		if errors.Is(err, binding.ErrInputLimit) || errors.Is(err, binding.ErrCapability) {
			s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
			return
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "SIMILARITY_CALCULATION_FAILED",
			fmt.Sprintf("failed to calculate similarity: %v", err))
		return
	}

	response := result

	logging.Infof("Calculated similarity: %.4f (model: %s, took: %.2fms)",
		result.Similarity, result.ModelUsed, result.ProcessingTimeMs)

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

	cfg, prepared, release, err := s.acquireEmbeddingRuntimeForRecipe(req.Recipe)
	defer release()
	if err != nil {
		s.writeEmbeddingRuntimeError(w, err)
		return
	}
	response, err := ownedBatchSimilarity(r.Context(), prepared, req)
	response.Recipe = embeddingRecipeName(req.Recipe, cfg)
	if err != nil {
		if errors.Is(err, binding.ErrInputLimit) || errors.Is(err, binding.ErrCapability) {
			s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
			return
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "BATCH_SIMILARITY_FAILED", err.Error())
		return
	}

	logging.Infof("Calculated batch similarity: query=%s, %d candidates, top-%d matches (model: %s, took: %.2fms)",
		logging.ContentDescriptor(req.Query), len(req.Candidates), len(response.Matches), response.ModelUsed, response.ProcessingTimeMs)

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

// isValidDimension checks if the provided dimension is valid
func isValidDimension(dim int) bool { return dim >= 0 && dim <= math.MaxInt32 }

// formatLayerList renders a layer set as a comma-separated string for error
// messages, e.g. [6 11 16 22] -> "6, 11, 16, 22".
func formatLayerList(layers []int) string {
	parts := make([]string, len(layers))
	for i, l := range layers {
		parts[i] = strconv.Itoa(l)
	}
	return strings.Join(parts, ", ")
}

func (s *ClassificationAPIServer) writeEmbeddingRuntimeError(w http.ResponseWriter, err error) {
	if errors.Is(err, services.ErrUnknownDiagnosticRecipe) {
		s.writeClassificationError(w, err)
		return
	}
	s.writeErrorResponse(w, http.StatusServiceUnavailable, "EMBEDDING_UNAVAILABLE", err.Error())
}

func embeddingRecipeName(requested string, cfg *config.RouterConfig) string {
	if requested == "" && cfg != nil && cfg.RoutingScope != "" {
		return string(cfg.RoutingScope)
	}
	return diagnosticRecipeName(requested)
}
