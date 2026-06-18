//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// crossEncoderAlias is a convenience model name that always selects the
// cross-encoder backend, independent of the configured served model id.
const crossEncoderAlias = "cross-encoder"

// crossEncoderServedName is the public model id that maps to the loaded
// cross-encoder, configured at startup (see server.go initCrossEncoderFromEnv).
// When set (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"), a client can request
// that exact id and get the cross-encoder backend, matching the Cohere/vLLM
// contract where `model` names the served reranker. Empty means only the
// "cross-encoder" alias routes to the cross-encoder backend.
var crossEncoderServedName string

// requestsCrossEncoder reports whether req.Model targets the cross-encoder
// backend: either the "cross-encoder" alias or the configured served model id.
func requestsCrossEncoder(model string) bool {
	if model == crossEncoderAlias {
		return true
	}
	return crossEncoderServedName != "" && model == crossEncoderServedName
}

// servedRerankModelName is the model id echoed back in cross-encoder responses;
// it reports the actual served model when configured rather than the alias.
func servedRerankModelName() string {
	if crossEncoderServedName != "" {
		return crossEncoderServedName
	}
	return crossEncoderAlias
}

// handleRerank reranks a set of documents against a query and returns them
// ordered by relevance. The request/response schema is compatible with the
// Cohere / vLLM /v1/rerank API.
//
//   - Phase 1 (default): relevance scores come from the existing bi-encoder
//     similarity backend (candle_binding.CalculateSimilarityBatch).
//   - Phase 2: when the request names the cross-encoder (the "cross-encoder"
//     alias or the configured served model id), scores come from a true
//     cross-encoder that jointly encodes each (query, document) pair
//     (candle_binding.RerankDocuments) for higher ranking precision.
func (s *ClassificationAPIServer) handleRerank(w http.ResponseWriter, r *http.Request) {
	req, ok := s.parseRerankRequest(w, r)
	if !ok {
		return
	}

	if requestsCrossEncoder(req.Model) {
		s.handleRerankCrossEncoder(w, req)
		return
	}

	result, err := candle_binding.CalculateSimilarityBatch(
		req.Query,
		req.Documents,
		req.TopN,
		req.Model,
		req.Dimension,
	)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RERANK_FAILED",
			fmt.Sprintf("failed to rerank documents: %v", err))
		return
	}

	results, err := buildRerankResults(result, req.Documents, req.ReturnDocuments)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RERANK_INVALID_RESULT", err.Error())
		return
	}

	response := RerankResponse{
		Model:            result.ModelType,
		Results:          results,
		TotalDocuments:   len(req.Documents),
		ProcessingTimeMs: result.ProcessingTimeMs,
	}

	logging.Infof("Reranked %d documents for query (top-%d, model: %s, took: %.2fms)",
		len(req.Documents), len(results), result.ModelType, result.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleRerankCrossEncoder serves the Phase 2 path: a true cross-encoder that
// scores each (query, document) pair jointly. Requires the cross-encoder model
// to have been initialized at startup.
func (s *ClassificationAPIServer) handleRerankCrossEncoder(w http.ResponseWriter, req RerankRequest) {
	if !candle_binding.IsCrossEncoderInitialized() {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "RERANK_MODEL_UNAVAILABLE",
			"cross-encoder reranker is not initialized; configure a cross-encoder model (SR_CROSS_ENCODER_MODEL_PATH) to enable the \"cross-encoder\" model")
		return
	}

	out, err := candle_binding.RerankDocuments(req.Query, req.Documents, req.TopN)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RERANK_FAILED",
			fmt.Sprintf("failed to rerank documents: %v", err))
		return
	}

	results, err := buildRerankResultsFromCrossEncoder(out, req.Documents, req.ReturnDocuments)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "RERANK_INVALID_RESULT", err.Error())
		return
	}

	servedName := servedRerankModelName()
	response := RerankResponse{
		Model:            servedName,
		Results:          results,
		TotalDocuments:   len(req.Documents),
		ProcessingTimeMs: out.ProcessingTimeMs,
	}

	logging.Infof("Reranked %d documents for query (top-%d, model: %s, took: %.2fms)",
		len(req.Documents), len(results), servedName, out.ProcessingTimeMs)

	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) parseRerankRequest(w http.ResponseWriter, r *http.Request) (RerankRequest, bool) {
	var req RerankRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return RerankRequest{}, false
	}

	applyRerankDefaults(&req)
	if code, message, ok := validateRerankRequest(req); !ok {
		s.writeErrorResponse(w, http.StatusBadRequest, code, message)
		return RerankRequest{}, false
	}
	normalizeRerankLimit(&req)

	return req, true
}

func applyRerankDefaults(req *RerankRequest) {
	if req.Model == "" {
		req.Model = "auto"
	}
	if req.Dimension == 0 {
		req.Dimension = defaultEmbeddingDimension
	}
	if req.TopN == 0 {
		req.TopN = len(req.Documents)
	}
	if req.Model == "auto" && req.QualityPriority == 0 && req.LatencyPriority == 0 {
		req.QualityPriority = defaultEmbeddingPriority
		req.LatencyPriority = defaultEmbeddingPriority
	}
}

func validateRerankRequest(req RerankRequest) (string, string, bool) {
	if req.Query == "" {
		return "INVALID_INPUT", "query must be provided", false
	}
	if len(req.Documents) == 0 {
		return "INVALID_INPUT", "documents array cannot be empty", false
	}
	if req.TopN < 0 {
		return "INVALID_INPUT", "top_n cannot be negative", false
	}
	if !isValidDimension(req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf("dimension must be one of: 64, 128, 256, 512, 768, 1024 (got %d)", req.Dimension), false
	}
	return "", "", true
}

func normalizeRerankLimit(req *RerankRequest) {
	if req.TopN > len(req.Documents) {
		req.TopN = len(req.Documents)
	}
}

func buildRerankResults(result *candle_binding.BatchSimilarityOutput, documents []string, returnDocuments bool) ([]RerankResult, error) {
	if result == nil {
		return nil, fmt.Errorf("rerank result is nil")
	}

	results := make([]RerankResult, len(result.Matches))
	for i, match := range result.Matches {
		if match.Index < 0 || match.Index >= len(documents) {
			return nil, fmt.Errorf("match index %d is out of range for %d documents", match.Index, len(documents))
		}
		entry := RerankResult{
			Index:          match.Index,
			RelevanceScore: match.Similarity,
		}
		if returnDocuments {
			entry.Document = &RerankDocument{Text: documents[match.Index]}
		}
		results[i] = entry
	}
	return results, nil
}

func buildRerankResultsFromCrossEncoder(out *candle_binding.RerankOutput, documents []string, returnDocuments bool) ([]RerankResult, error) {
	if out == nil {
		return nil, fmt.Errorf("rerank result is nil")
	}

	results := make([]RerankResult, len(out.Matches))
	for i, match := range out.Matches {
		if match.Index < 0 || match.Index >= len(documents) {
			return nil, fmt.Errorf("match index %d is out of range for %d documents", match.Index, len(documents))
		}
		entry := RerankResult{
			Index:          match.Index,
			RelevanceScore: match.Score,
		}
		if returnDocuments {
			entry.Document = &RerankDocument{Text: documents[match.Index]}
		}
		results[i] = entry
	}
	return results, nil
}
