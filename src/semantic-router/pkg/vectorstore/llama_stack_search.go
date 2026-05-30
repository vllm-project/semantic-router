/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// Search performs a text-based similarity search in a Llama Stack vector store.
// Maps to: POST /v1/vector_stores/{vector_store_id}/search
//
// Llama Stack searches by text query, not by embedding vector. The query text
// must be passed via filter["_query_text"]; the queryEmbedding param is ignored.
//
// When searchType is "hybrid", the request includes ranking_options that tell
// Llama Stack to combine vector similarity and BM25 keyword search using
// Reciprocal Rank Fusion. This requires the Milvus vector_io provider.
func (l *LlamaStackBackend) Search(
	ctx context.Context, vectorStoreID string, queryEmbedding []float32,
	topK int, threshold float32, filter map[string]interface{},
) ([]SearchResult, error) {
	queryText, ok := filter["_query_text"].(string)
	if !ok || queryText == "" {
		return nil, fmt.Errorf(
			"llama_stack backend requires '_query_text' in the filter map for search; " +
				"Llama Stack searches by text query, not by embedding vector")
	}

	storeID, err := l.resolveStoreID(ctx, vectorStoreID)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve vector store %s for search: %w", vectorStoreID, err)
	}

	path := fmt.Sprintf("/v1/vector_stores/%s/search", storeID)
	resp, err := l.doRequest(ctx, http.MethodPost, path, l.searchRequestBody(queryText, topK, filter))
	if err != nil {
		return nil, fmt.Errorf("failed to search vector store %s in Llama Stack: %w", vectorStoreID, err)
	}

	// RRF scores (hybrid search) live on a fundamentally different scale than
	// cosine similarity (vector search):
	//   vector  → cosine similarity  → 0.0 – 1.0  (threshold ~0.7 is reasonable)
	//   hybrid  → RRF = Σ 1/(k+rank) → 0.001 – 0.05 (threshold 0.7 would drop everything)
	// When hybrid search is active the results are already ranked by the RRF
	// combiner, so we skip score-based filtering and rely on topK to limit volume.
	return parseLlamaStackSearchResults(resp, threshold, l.searchType != "hybrid")
}

func (l *LlamaStackBackend) searchRequestBody(
	queryText string, topK int, filter map[string]interface{},
) map[string]interface{} {
	body := map[string]interface{}{
		"query":           queryText,
		"max_num_results": topK,
	}
	if l.searchType == "hybrid" {
		body["ranking_options"] = map[string]interface{}{"ranker": "rrf"}
	}
	if fid, ok := filter["file_id"].(string); ok && fid != "" {
		body["filters"] = map[string]interface{}{
			"type":  "eq",
			"key":   "file_id",
			"value": fid,
		}
	}
	return body
}

type llamaStackSearchResponse struct {
	Data []llamaStackSearchHit `json:"data"`
}

type llamaStackSearchHit struct {
	Content  []map[string]interface{} `json:"content"`
	FileID   string                   `json:"file_id"`
	Filename string                   `json:"filename"`
	Score    float64                  `json:"score"`
}

func parseLlamaStackSearchResults(
	resp []byte, threshold float32, applyThreshold bool,
) ([]SearchResult, error) {
	var searchResp llamaStackSearchResponse
	if err := json.Unmarshal(resp, &searchResp); err != nil {
		return nil, fmt.Errorf("failed to parse Llama Stack search response: %w", err)
	}

	var results []SearchResult
	for _, hit := range searchResp.Data {
		if applyThreshold && float32(hit.Score) < threshold {
			continue
		}
		results = append(results, SearchResult{
			FileID:   hit.FileID,
			Filename: hit.Filename,
			Content:  llamaStackTextContent(hit.Content),
			Score:    hit.Score,
		})
	}
	return results, nil
}

func llamaStackTextContent(content []map[string]interface{}) string {
	for _, item := range content {
		if item["type"] != "text" {
			continue
		}
		text, ok := item["text"].(string)
		if ok {
			return text
		}
	}
	return ""
}
