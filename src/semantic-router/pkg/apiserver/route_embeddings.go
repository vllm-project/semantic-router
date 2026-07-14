//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// multiModalEncodeImage is the FFI image-encode entry point, kept as a
// package-level var so tests can inject a failing encoder without a loaded model.
var multiModalEncodeImage = candle_binding.MultiModalEncodeImageFromBase64

// multiModalEmbeddingDim reports the loaded multimodal model's native embedding
// dimension (or <= 0 if none is loaded). It is a package-level var so tests can
// stub the native dimension without a loaded model.
var multiModalEmbeddingDim = candle_binding.MultiModalGetEmbeddingDim

// multiModalModelFallbackID is the reported model id when no multimodal model
// path is configured; it matches the canonical shipped checkpoint.
const multiModalModelFallbackID = "multi-modal-embed-small"

// imageTargetDimension resolves the per-request image encoding dimension.
//
// Image-only requests use req.Dimension directly (validated <= native). For
// mixed text+image requests req.Dimension targets the text side (its default,
// 768, exceeds the image ceiling); the image side honors it only when it fits
// within the native ceiling and otherwise caps at native. This keeps an
// explicitly requested, image-satisfiable dimension (e.g. 256) uniform across
// both modalities, while a text-scale dimension (e.g. the 768 default) still
// diverges instead of forcing text recall down to the image width.
//
// If no multimodal model is loaded (getter reports <= 0), fall back to
// req.Dimension so the downstream encode surfaces "model not loaded" rather
// than this helper guessing a value.
func imageTargetDimension(req EmbeddingRequest) int {
	if len(req.Texts) == 0 {
		return req.Dimension
	}
	native := multiModalEmbeddingDim()
	if native <= 0 {
		return req.Dimension
	}
	if req.Dimension < native {
		return req.Dimension
	}
	return native
}

// isMultiModalModelName reports whether an explicit model selector names the
// multimodal embedding model. Mirrors the aliases registered in
// config/registry.go so an image request may name the multimodal model
// explicitly without being rejected as a text-only selector.
func isMultiModalModelName(model string) bool {
	switch model {
	case "multimodal", "multi-modal-embed-small", "multimodal-embedding",
		"embedding-multimodal", "mom-embedding-multimodal":
		return true
	}
	return false
}

// imageEncodeError marks an image-encode failure driven by the request input:
// the payload validated as a safe base64 data URI but was not a decodable image.
// The handler maps it to 400 INVALID_IMAGE rather than 500, so a client-supplied
// bad image is reported as a client error, not an internal one.
type imageEncodeError struct {
	index int
	err   error
}

func (e *imageEncodeError) Error() string {
	return fmt.Sprintf("images[%d]: %v", e.index, e.err)
}

func (e *imageEncodeError) Unwrap() error { return e.err }

// classifyEmbeddingError maps a buildEmbeddingResults error to the HTTP status,
// error code, and client message. Input-caused image-encode failures are 400
// INVALID_IMAGE; every other failure is a genuine 500.
func classifyEmbeddingError(err error) (int, string, string) {
	var imgErr *imageEncodeError
	if errors.As(err, &imgErr) {
		return http.StatusBadRequest, "INVALID_IMAGE",
			fmt.Sprintf("images[%d] could not be decoded as an image", imgErr.index)
	}
	return http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
		fmt.Sprintf("failed to generate embedding: %v", err)
}

const (
	defaultEmbeddingDimension = 768
	defaultEmbeddingPriority  = 0.5
	// maxImagesPerRequest bounds images per request; each is a full SigLIP
	// forward pass and the body-size cap alone admits very many minimal images.
	maxImagesPerRequest = 8
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

	results, totalProcessingTime, err := buildEmbeddingResults(req, s.multiModalModelID())
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

// multiModalModelID reports the loaded multimodal model identifier for the
// response's model_used field. It derives a clean id from the configured model
// path (its base name, so a local path like "models/multi-modal-embed-small"
// or a HF repo id both reduce to "multi-modal-embed-small" without leaking the
// deployment's filesystem layout) and falls back to the canonical shipped id
// when no path is configured.
func (s *ClassificationAPIServer) multiModalModelID() string {
	if s.config != nil {
		if path := s.config.EmbeddingModels.MultiModalModelPath; path != "" {
			return filepath.Base(path)
		}
	}
	return multiModalModelFallbackID
}

func (s *ClassificationAPIServer) parseEmbeddingRequest(w http.ResponseWriter, r *http.Request) (EmbeddingRequest, bool) {
	var req EmbeddingRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return EmbeddingRequest{}, false
	}

	applyEmbeddingDefaults(&req)
	mmbertPath := ""
	if cfg := s.currentConfig(); cfg != nil {
		mmbertPath = cfg.EmbeddingModels.MmBertModelPath
	}
	availableLayers := config.MmBertAvailableLayers(mmbertPath)
	if code, message, ok := validateEmbeddingRequest(req, availableLayers); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
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
		// req.Dimension follows the primary modality:
		//   - text-only / mixed: text default (768). In mixed mode Dimension
		//     controls the text side; the image side always encodes at its
		//     native dimension and is not affected by req.Dimension.
		//   - image-only: multimodal native dimension (384), the ceiling MRL
		//     can produce.
		// Forcing mixed requests to a single uniform dimension would silently
		// downgrade text recall (768 -> 384) for a false-consistency win:
		// cross-encoder vectors live in different embedding spaces regardless
		// of dimension, so uniform width does not enable cross-modal cosine.
		// Per-item response.dimension disambiguates for the caller.
		if len(req.Texts) == 0 && len(req.Images) > 0 {
			if native := multiModalEmbeddingDim(); native > 0 {
				req.Dimension = native
			} else {
				req.Dimension = defaultEmbeddingDimension
			}
		} else {
			req.Dimension = defaultEmbeddingDimension
		}
	}
	if req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func validateEmbeddingRequest(req EmbeddingRequest, mmbertLayers []int) (string, string, bool) {
	if len(req.Texts) == 0 && len(req.Images) == 0 {
		return "INVALID_INPUT", "at least one of texts or images must be provided", false
	}
	if code, message, ok := validateEmbeddingImages(req.Images); !ok {
		return code, message, false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf("dimension must be one of: 64, 128, 256, 384, 512, 768, 1024 (got %d)", req.Dimension), false
	}
	if len(req.Images) > 0 {
		// Image-only: req.Dimension is the image target, so the multimodal
		// native dimension is a hard ceiling (MRL only truncates down; an
		// above-native request cannot be satisfied and is rejected rather
		// than silently returning native).
		// Mixed: req.Dimension applies to the text side and the image side
		// always encodes at its native dimension, so a text-legal
		// req.Dimension (e.g. 768) is not constrained by the image ceiling.
		if len(req.Texts) == 0 {
			if native := multiModalEmbeddingDim(); native > 0 && req.Dimension > native {
				return "INVALID_DIMENSION", fmt.Sprintf("dimension must be <= %d for image inputs (multimodal model native dimension; got %d)", native, req.Dimension), false
			}
		}
		// target_layer is a text-only mmbert 2DMSE control; it cannot apply to
		// image encoding, so reject instead of silently ignoring it.
		if req.TargetLayer != 0 {
			return "INVALID_PARAMETER", "target_layer is not supported for image inputs", false
		}
		// Images always use the multimodal model. When there is no text for a
		// named model to apply to, an explicit text-model selector would be
		// silently ignored; reject so the contract is explicit. Mixed
		// text+image requests still honor req.Model for the text side.
		if len(req.Texts) == 0 && req.Model != "auto" && req.Model != "" && !isMultiModalModelName(req.Model) {
			return "INVALID_PARAMETER", fmt.Sprintf("model=%q does not produce image embeddings; omit model or use the multimodal model for image inputs", req.Model), false
		}
	}
	if req.TargetLayer != 0 && req.Model != "mmbert" {
		return "INVALID_PARAMETER", "target_layer is only supported for model='mmbert'", false
	}
	if req.Model == "mmbert" && req.TargetLayer != 0 && !config.IsValidMmBertLayer(req.TargetLayer, mmbertLayers) {
		return "INVALID_LAYER", fmt.Sprintf("target_layer must be one of: %s (got %d)", formatLayerList(mmbertLayers), req.TargetLayer), false
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

func buildEmbeddingResults(req EmbeddingRequest, multiModalModelID string) ([]EmbeddingResult, int64, error) {
	results := make([]EmbeddingResult, 0, len(req.Texts)+len(req.Images))
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

	imgDim := imageTargetDimension(req)
	for i, image := range req.Images {
		// Canonicalize so the FFI's case-sensitive ";base64," scan finds the
		// payload boundary (validation already guaranteed a safe data URI).
		encodeInput := image
		if canonical, ok := imageurl.CanonicalDataURL(image); ok {
			encodeInput = canonical
		}
		output, err := multiModalEncodeImage(encodeInput, imgDim)
		if err != nil {
			// The image already passed the safe-data-URI + base64-decode gate, so
			// an encode failure here is input-caused (undecodable image bytes);
			// surface it as a 400 rather than a 500.
			return nil, 0, &imageEncodeError{index: i, err: err}
		}

		processingTime := int64(output.ProcessingTimeMs)
		results = append(results, EmbeddingResult{
			Modality:         output.Modality,
			Embedding:        output.Embedding,
			Dimension:        len(output.Embedding),
			ModelUsed:        multiModalModelID,
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
	// 384 is the multimodal model's native dimension; it must be accepted so
	// callers can request the image model's full-width vector explicitly.
	validDimensions := map[int]bool{64: true, 128: true, 256: true, 384: true, 512: true, 768: true, 1024: true}
	return validDimensions[dim]
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
