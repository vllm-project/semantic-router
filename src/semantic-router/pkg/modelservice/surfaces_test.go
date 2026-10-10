package modelservice

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// surfaceRuntime answers the classify, embeddings, rerank and bundle surfaces
// like the runtime does, and counts the calls each path received. A bundle
// with more than maxTasks tasks (when set) is refused whole, as the runtime's
// --max-bundle-tasks does; /v1/models lists a card per maxInputs entry. Model
// "echo" labels each classify input with its own text.
type surfaceRuntime struct {
	direct    atomic.Int64
	bundles   atomic.Int64
	tasks     atomic.Int64
	delay     time.Duration
	maxTasks  int
	maxInputs map[string]int
	mu        sync.Mutex
	seen      []map[string]interface{}
}

func (s *surfaceRuntime) handler() http.Handler {
	mux := http.NewServeMux()
	surface := func(name string) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			s.direct.Add(1)
			var body map[string]interface{}
			_ = json.NewDecoder(r.Body).Decode(&body)
			s.mu.Lock()
			s.seen = append(s.seen, body)
			s.mu.Unlock()
			status, response := s.answer(name, body)
			writeJSON(w, status, response)
		}
	}
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, _ *http.Request) {
		limit := s.maxTasks
		if limit == 0 {
			limit = DefaultBundleTasks
		}
		cards := []interface{}{}
		for model, inputs := range s.maxInputs {
			cards = append(cards, map[string]interface{}{"id": model, "object": "model", "family": "task_heads", "surfaces": []string{"classify"}, "ready": true, "limits": map[string]int{"max_inputs": inputs}})
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"object": "list", "api_version": "2.0.0", "data": cards, "limits": map[string]int{"max_bundle_tasks": limit, "max_request_bytes": 8 << 20}})
	})
	mux.HandleFunc("/v1/classify", surface("classify"))
	mux.HandleFunc("/v1/embeddings", surface("embeddings"))
	mux.HandleFunc("/v1/rerank", surface("rerank"))
	mux.HandleFunc("/v1/bundle", func(w http.ResponseWriter, r *http.Request) {
		s.bundles.Add(1)
		time.Sleep(s.delay)
		var body struct {
			Tasks []map[string]json.RawMessage `json:"tasks"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		if s.maxTasks > 0 && len(body.Tasks) > s.maxTasks {
			writeJSON(w, http.StatusRequestEntityTooLarge, map[string]interface{}{"error": map[string]interface{}{"code": "request_too_large", "message": "too many tasks"}})
			return
		}
		results := make([]map[string]interface{}, 0, len(body.Tasks))
		for _, task := range body.Tasks {
			s.tasks.Add(1)
			var id string
			_ = json.Unmarshal(task["id"], &id)
			for _, name := range []string{"classify", "embeddings", "rerank"} {
				raw, ok := task[name]
				if !ok {
					continue
				}
				var request map[string]interface{}
				_ = json.Unmarshal(raw, &request)
				status, response := s.answer(name, request)
				result := map[string]interface{}{"id": id, "status": status}
				if status == http.StatusOK {
					result[name] = response
				} else {
					result["error"] = response["error"]
				}
				results = append(results, result)
			}
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{"results": results})
	})
	return mux
}

func (s *surfaceRuntime) answer(surface string, body map[string]interface{}) (int, map[string]interface{}) {
	model, _ := body["model"].(string)
	switch model {
	case "missing":
		return http.StatusNotFound, map[string]interface{}{"error": map[string]interface{}{"code": "model_not_found", "message": "no such model"}}
	case "busy":
		return http.StatusTooManyRequests, map[string]interface{}{"error": map[string]interface{}{"code": "overloaded", "message": "busy"}}
	case "loading":
		return http.StatusServiceUnavailable, map[string]interface{}{"error": map[string]interface{}{"code": "not_ready", "message": "loading"}}
	case "other":
		return http.StatusUnprocessableEntity, map[string]interface{}{"error": map[string]interface{}{"code": "unsupported_surface", "message": "no"}}
	}
	usage := map[string]interface{}{"input_tokens": 7, "output_tokens": 0}
	switch surface {
	case "classify":
		inputs, _ := body["input"].([]interface{})
		results := make([]map[string]interface{}, len(inputs))
		for index := range inputs {
			label := "billing"
			if item, _ := inputs[index].(map[string]interface{}); model == "echo" && item != nil {
				label, _ = item["text"].(string)
			}
			results[index] = map[string]interface{}{
				"index": index, "label": label, "probabilities": []float64{0.9, 0.1},
				"spans":   []map[string]interface{}{{"label": "PERSON", "start": 0, "end": 3, "text": "Tom", "probability": 0.99}},
				"windows": []map[string]interface{}{{"start": 0, "end": 4, "probabilities": []float64{0.9, 0.1}}},
				"input":   map[string]interface{}{"tokens": 6, "processed_tokens": 6, "truncated": false, "windows": 1},
			}
		}
		return http.StatusOK, map[string]interface{}{"model": model, "head": "default", "kind": "sequence", "labels": []string{"billing", "other"}, "results": results, "usage": usage}
	case "embeddings":
		raw := make([]byte, 8)
		binary.LittleEndian.PutUint32(raw, math.Float32bits(0.6))
		binary.LittleEndian.PutUint32(raw[4:], math.Float32bits(0.8))
		data := []map[string]interface{}{
			{"object": "embedding", "index": 1, "embedding": []float64{1, 0}},
			{"object": "embedding", "index": 0, "embedding": base64.StdEncoding.EncodeToString(raw)},
		}
		return http.StatusOK, map[string]interface{}{
			"object": "list", "model": model, "data": data, "usage": map[string]interface{}{"prompt_tokens": 4, "total_tokens": 4},
			"meta": map[string]interface{}{"representation": map[string]interface{}{"model_sha256": "f00d", "layer": 22, "dimension": 2, "normalized": true}},
		}
	default:
		return http.StatusOK, map[string]interface{}{"model": model, "results": []map[string]interface{}{
			{"index": 1, "logit": 2.5, "relevance_score": 0.92},
			{"index": 0, "logit": -1.0, "relevance_score": 0.27},
		}, "usage": usage}
	}
}

func writeJSON(w http.ResponseWriter, status int, body interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func newSurfaceClient(t *testing.T, runtime *surfaceRuntime) *Client {
	t.Helper()
	server := httptest.NewServer(runtime.handler())
	t.Cleanup(server.Close)
	client, err := NewClient(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	return client
}

func TestClassifyEncodesInputsAndDecodesResults(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	threshold := 0.4
	response, err := client.Classify(ctx, "vela-pii", ClassifyRequest{
		Inputs:   []ClassifyInput{{Text: "Tom"}, {Context: "c", Answer: "a"}},
		Overflow: "window", Window: &Window{Tokens: 512, Overlap: 64}, Threshold: &threshold,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Results) != 2 || response.Results[0].Label != "billing" {
		t.Fatalf("unexpected response %+v", response)
	}
	if span := response.Results[0].Spans[0]; span.Label != "PERSON" || span.End != 3 || response.Results[0].Windows[0].End != 4 {
		t.Fatalf("spans and windows were not decoded: %+v", response.Results[0])
	}
	body := runtime.seen[0]
	options := body["options"].(map[string]interface{})
	if options["overflow"] != "window" || options["deadline_ms"] == nil || options["window"].(map[string]interface{})["overlap"] != 64.0 {
		t.Fatalf("options were not sent: %v", options)
	}
	inputs := body["input"].([]interface{})
	if inputs[0].(map[string]interface{})["text"] != "Tom" || inputs[1].(map[string]interface{})["answer"] != "a" {
		t.Fatalf("inputs were not sent as objects: %v", inputs)
	}
}

func TestSurfaceStatusesMapToErrors(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{})
	cases := map[string]error{"busy": ErrOverloaded, "loading": ErrUnavailable, "missing": ErrRejected, "other": ErrRejected}
	for model, want := range cases {
		_, err := client.Classify(context.Background(), model, ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}})
		if !errors.Is(err, want) {
			t.Fatalf("%s: got %v, want %v", model, err, want)
		}
	}
	if _, err := client.Classify(context.Background(), "m", ClassifyRequest{}); !errors.Is(err, ErrRejected) {
		t.Fatalf("an empty classify request must be rejected locally, got %v", err)
	}
}

func TestEmbedDecodesBase64AndFloatVectorsByIndex(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{})
	response, err := client.Embed(context.Background(), "vela-embedding", EmbedRequest{Inputs: []EmbedInput{{Text: "a"}, {ImageURL: "data:image/png;base64,AA=="}}, Dimensions: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := response.Embeddings[0]; len(got) != 2 || math.Abs(float64(got[0])-0.6) > 1e-6 || math.Abs(float64(got[1])-0.8) > 1e-6 {
		t.Fatalf("base64 vector decoded as %v", got)
	}
	if got := response.Embeddings[1]; got[0] != 1 || got[1] != 0 {
		t.Fatalf("float vector decoded as %v", got)
	}
}

func TestSurfacesDoNotAskForMeta(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err := client.Classify(ctx, "m", ClassifyRequest{Inputs: []ClassifyInput{{Text: "x"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Embed(ctx, "m", EmbedRequest{Inputs: []EmbedInput{{Text: "a"}, {Text: "b"}}}); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Rerank(ctx, "m", RerankRequest{Query: "q", Documents: []string{"a", "b"}}); err != nil {
		t.Fatal(err)
	}
	if len(runtime.seen) != 3 {
		t.Fatalf("got %d surface calls, want 3", len(runtime.seen))
	}
	for _, body := range runtime.seen {
		options, _ := body["options"].(map[string]interface{})
		if _, asked := options["return_meta"]; asked || options["deadline_ms"] == nil {
			t.Fatalf("options = %v, want a deadline and no return_meta", options)
		}
	}
}

func TestRerankReturnsResultsInDocumentOrder(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{})
	response, err := client.Rerank(context.Background(), "vela-reranker", RerankRequest{Query: "q", Documents: []string{"a", "b"}})
	if err != nil {
		t.Fatal(err)
	}
	if response.Results[0].Logit != -1.0 || response.Results[1].Logit != 2.5 || response.Results[1].Score != 0.92 {
		t.Fatalf("results are not in document order: %+v", response.Results)
	}
	if _, err := client.Rerank(context.Background(), "vela-reranker", RerankRequest{Query: "q", Documents: []string{"a", "b", "c"}}); !errors.Is(err, ErrFailed) {
		t.Fatalf("a missing document result must fail, got %v", err)
	}
}

func TestExpiredDeadlinesFailBeforeTheCall(t *testing.T) {
	runtime := &surfaceRuntime{}
	client := newSurfaceClient(t, runtime)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	if _, err := client.Embed(ctx, "m", EmbedRequest{Inputs: []EmbedInput{{Text: "x"}}}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("got %v", err)
	}
	if runtime.direct.Load() != 0 {
		t.Fatal("an expired call reached the runtime")
	}
}

func TestBundleTasksDecodeLikeDirectCalls(t *testing.T) {
	client := newSurfaceClient(t, &surfaceRuntime{})
	results, err := client.Bundle(context.Background(), []api.BundleTask{{Id: "1", Rerank: &api.RerankRequest{Query: "q", Documents: []string{"a", "b"}}}})
	if err != nil || len(results) != 1 || results[0].Rerank == nil {
		t.Fatalf("got %v %v", results, err)
	}
	decoded, err := decodeRerank(*results[0].Rerank, 2)
	if err != nil || decoded.Results[1].Logit != 2.5 {
		t.Fatalf("got %+v %v", decoded, err)
	}
}
