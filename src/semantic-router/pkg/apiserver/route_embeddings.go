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

type embeddingDimensionContract struct {
	ModelID   string
	Default   int
	Supported []int
}

// multiModalDimensionContract reports the immutable contract registered by
// the loaded multimodal binding. It is replaceable for tests that do not load
// a checkpoint.
var multiModalDimensionContract = func() embeddingDimensionContract {
	return embeddingDimensionContract{
		ModelID:   multiModalModelFallbackID,
		Default:   candle_binding.MultiModalGetEmbeddingDim(),
		Supported: candle_binding.MultiModalGetSupportedDimensions(),
	}
}

// multiModalModelFallbackID is used only when the configured checkpoint cannot
// be resolved through the model registry.
const multiModalModelFallbackID = "multi-modal-embed-small"

func isMultiModalModelName(model string) bool {
	switch model {
	case "multimodal", "multi-modal-embed-small", "multimodal-embedding",
		"embedding-multimodal", "mom-embedding-multimodal":
		return true
	}
	return false
}

func (c embeddingDimensionContract) supports(dimension int) bool {
	for _, supported := range c.Supported {
		if dimension == supported {
			return true
		}
	}
	return false
}

func (c embeddingDimensionContract) supportedList() string {
	return formatLayerList(c.Supported)
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

// validEmbeddingDimensions is the model-agnostic allowlist for the text
// embedding and similarity endpoints. The multimodal endpoints never consult it;
// they validate against the loaded checkpoint's declared ladder instead.
var validEmbeddingDimensions = []int{64, 128, 256, 512, 768, 1024}

// invalidDimensionMessage is derived from validEmbeddingDimensions so the
// accepted set and the rejection message cannot drift apart.
var invalidDimensionMessage = fmt.Sprintf(
	"dimension must be one of: %s (got %%d)", formatLayerList(validEmbeddingDimensions))

// embeddingValidationStatus maps a validation code to its HTTP status. Model
// unavailability is server state, not a malformed request, so it is the one code
// that is not a 400.
func embeddingValidationStatus(code string) int {
	if code == "MODEL_NOT_LOADED" {
		return http.StatusServiceUnavailable
	}
	return http.StatusBadRequest
}

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

// multiModalModelID reports the loaded multimodal checkpoint identifier for the
// response's model_used field. Known local aliases resolve through the static
// registry; custom paths fall back to their base name.
func (s *ClassificationAPIServer) multiModalModelID() string {
	if s.config != nil {
		if path := s.config.EmbeddingModels.MultiModalModelPath; path != "" {
			if model := config.GetModelByPath(path); model != nil && model.RepoID != "" {
				return filepath.Base(model.RepoID)
			}
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
	if req.Model == "" || (req.Model == "auto" && len(req.Images) > 0) {
		if len(req.Images) > 0 {
			req.Model = "multimodal"
		} else {
			req.Model = "auto"
		}
	} else if isMultiModalModelName(req.Model) {
		req.Model = "multimodal"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimensionForModel(req.Model)
	}
	if req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func defaultEmbeddingDimensionForModel(model string) int {
	if !isMultiModalModelName(model) {
		return defaultEmbeddingDimension
	}
	// A non-positive value is the binding's "no model loaded" sentinel. Leave the
	// dimension unset rather than forwarding the sentinel, so validation reports
	// unavailability instead of the encoder silently substituting a width.
	if dimension := multiModalDimensionContract().Default; dimension > 0 {
		return dimension
	}
	return 0
}

func validateEmbeddingRequest(req EmbeddingRequest, mmbertLayers []int) (string, string, bool) {
	if len(req.Texts) == 0 && len(req.Images) == 0 {
		return "INVALID_INPUT", "at least one of texts or images must be provided", false
	}
	if code, message, ok := validateEmbeddingImages(req.Images); !ok {
		return code, message, false
	}
	// The resolved model decides which dimension ladder applies, so reject an
	// image request pointed at a text model first; otherwise the caller gets the
	// text allowlist's dimension error instead of the real mismatch.
	if code, message, ok := validateImageEmbeddingParams(req); !ok {
		return code, message, false
	}
	if code, message, ok := validateEmbeddingDimension(req); !ok {
		return code, message, false
	}
	if req.TargetLayer != 0 && req.Model != "mmbert" {
		return "INVALID_PARAMETER", "target_layer is only supported for model='mmbert'", false
	}
	if req.Model == "mmbert" && req.TargetLayer != 0 && !config.IsValidMmBertLayer(req.TargetLayer, mmbertLayers) {
		return "INVALID_LAYER", fmt.Sprintf("target_layer must be one of: %s (got %d)", formatLayerList(mmbertLayers), req.TargetLayer), false
	}
	return "", "", true
}

func validateEmbeddingDimension(req EmbeddingRequest) (string, string, bool) {
	if !isMultiModalModelName(req.Model) {
		if isValidDimension(req.Dimension) {
			return "", "", true
		}
		return "INVALID_DIMENSION", fmt.Sprintf(invalidDimensionMessage, req.Dimension), false
	}
	contract := multiModalDimensionContract()
	if len(contract.Supported) == 0 {
		return "MODEL_NOT_LOADED", "the multimodal embedding model is not loaded; check /ready before retrying", false
	}
	if contract.supports(req.Dimension) {
		return "", "", true
	}
	return "INVALID_DIMENSION", fmt.Sprintf("dimension %d is not supported by model %q; supported dimensions: %s", req.Dimension, contract.ModelID, contract.supportedList()), false
}

// validateImageEmbeddingParams ensures every image-bearing request resolves all
// of its inputs to the shared multimodal embedding space.
func validateImageEmbeddingParams(req EmbeddingRequest) (string, string, bool) {
	if len(req.Images) == 0 {
		return "", "", true
	}
	if !isMultiModalModelName(req.Model) {
		return "INVALID_PARAMETER", "image inputs require model='multimodal' so text and image vectors share one embedding space", false
	}
	if req.TargetLayer != 0 {
		return "INVALID_PARAMETER", "target_layer is not supported for image inputs", false
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

	for i, image := range req.Images {
		// Canonicalize so the FFI's case-sensitive ";base64," scan finds the
		// payload boundary (validation already guaranteed a safe data URI).
		encodeInput := image
		if canonical, ok := imageurl.CanonicalDataURL(image); ok {
			encodeInput = canonical
		}
		output, err := multiModalEncodeImage(encodeInput, req.Dimension)
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

	logging.Infof("Calculated batch similarity: query=%s, %d candidates, top-%d matches (model: %s, took: %.2fms)",
		logging.ContentDescriptor(req.Query), len(req.Candidates), len(matches), result.ModelType, result.ProcessingTimeMs)

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
	for _, valid := range validEmbeddingDimensions {
		if dim == valid {
			return true
		}
	}
	return false
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
