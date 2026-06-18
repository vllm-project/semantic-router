//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

// TestRerankLiveThroughRuntimeAndAPI exercises the full cross-encoder rerank
// path the running router uses: the model is loaded by the model-runtime
// lifecycle (not the API server), and requests are served through the real
// /v1/rerank route. It asserts the documented startup behavior (503 before the
// model is loaded, 200 with ranked results after) and that the served model id
// is echoed back.
//
// Skipped unless CROSS_ENCODER_MODEL_PATH points at a downloaded BERT
// cross-encoder (e.g. cross-encoder/ms-marco-MiniLM-L-6-v2), so CI without the
// model is unaffected.
func TestRerankLiveThroughRuntimeAndAPI(t *testing.T) {
	modelPath := os.Getenv("CROSS_ENCODER_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set CROSS_ENCODER_MODEL_PATH to run the live rerank integration test")
	}
	const servedName = "cross-encoder/ms-marco-MiniLM-L-6-v2"
	t.Setenv("SR_CROSS_ENCODER_MODEL_PATH", modelPath)
	t.Setenv("SR_CROSS_ENCODER_USE_CPU", "true")
	t.Setenv("SR_CROSS_ENCODER_MODEL_NAME", servedName)

	srv := &ClassificationAPIServer{}
	mux := srv.setupRoutes()

	// 1) Before the model is loaded, a cross-encoder request must 503 rather
	//    than silently fall back to bi-encoder scores.
	configureCrossEncoderServedName()
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/rerank", liveRerankBody("cross-encoder")))
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 before load, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "RERANK_MODEL_UNAVAILABLE") {
		t.Fatalf("expected RERANK_MODEL_UNAVAILABLE before load, got %s", w.Body.String())
	}

	// 2) Load the cross-encoder via the real model-runtime lifecycle path.
	if _, err := modelruntime.PrepareRouterRuntime(context.Background(), &config.RouterConfig{}, modelruntime.PrepareRouterRuntimeOptions{
		Component:      "test",
		MaxParallelism: 1,
	}); err != nil {
		t.Fatalf("PrepareRouterRuntime failed: %v", err)
	}
	configureCrossEncoderServedName()

	// 3) After load, requesting the served model id returns 200 with ranked
	//    results, the true answer (doc 0) at rank #1, and the served id echoed.
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest(http.MethodPost, "/v1/rerank", liveRerankBody(servedName)))
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after load, got %d: %s", w.Code, w.Body.String())
	}
	var resp RerankResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	t.Logf("live rerank: model=%s total=%d results=%+v", resp.Model, resp.TotalDocuments, resp.Results)
	if resp.Model != servedName {
		t.Fatalf("expected echoed model %q, got %q", servedName, resp.Model)
	}
	if len(resp.Results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(resp.Results))
	}
	if resp.Results[0].Index != 0 {
		t.Fatalf("expected the answer (doc 0) ranked #1, got index %d", resp.Results[0].Index)
	}
	if !(resp.Results[0].RelevanceScore > resp.Results[1].RelevanceScore) {
		t.Fatalf("expected descending scores, got %v then %v", resp.Results[0].RelevanceScore, resp.Results[1].RelevanceScore)
	}
}

func liveRerankBody(model string) *bytes.Reader {
	b, _ := json.Marshal(RerankRequest{
		Model: model,
		Query: "How do I get a refund for a cancelled flight?",
		Documents: []string{
			"Submit the refund request form within 30 days and the fare returns to your original payment method.",
			"Flights may be cancelled by the airline because of weather or operational issues.",
			"Seat upgrades can be purchased at check-in subject to availability.",
		},
		ReturnDocuments: false,
	})
	return bytes.NewReader(b)
}
