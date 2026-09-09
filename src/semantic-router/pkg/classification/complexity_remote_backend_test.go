package classification

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// score.v1 reuses the http_classify protocol and its HuggingFace-compatible
// wire shape, so a regression head deployed behind the usual text
// classification endpoint works without a shim. The label carries no meaning
// on this contract; the score is the product.
func TestScoringHTTPBackend_ReadsTheScore(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/classify" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var request map[string]string
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if request["inputs"] != "solve this step by step" {
			t.Errorf("inputs = %q", request["inputs"])
		}
		_, _ = w.Write([]byte(`[{"label":"LABEL_0","score":0.83}]`))
	}))
	defer server.Close()

	backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-scorer-svc",
	}, time.Second)
	if err != nil {
		t.Fatalf("newScoringHTTPBackend: %v", err)
	}

	score, err := backend.Score(context.Background(), "solve this step by step")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 0.83 {
		t.Fatalf("score = %v, want 0.83", score)
	}
}

// A caller that gives up must cancel the outbound request rather than leave
// it running to the backend's own deadline. ScoringBackend takes a context
// for exactly this reason.
func TestScoringHTTPBackend_PreservesCallerCancellation(t *testing.T) {
	requestStarted := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(requestStarted)
		time.Sleep(200 * time.Millisecond)
		_, _ = w.Write([]byte(`[{"label":"LABEL_0","score":0.5}]`))
	}))
	defer server.Close()

	backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-scorer-svc",
	}, time.Second)
	if err != nil {
		t.Fatalf("newScoringHTTPBackend: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errCh := make(chan error, 1)
	go func() {
		_, scoreErr := backend.Score(ctx, "cancel this")
		errCh <- scoreErr
	}()

	select {
	case <-requestStarted:
	case <-time.After(time.Second):
		t.Fatal("HTTP request did not reach the test server")
	}
	cancel()

	select {
	case err := <-errCh:
		if err == nil || !errors.Is(err, context.Canceled) {
			t.Fatalf("Score error = %v, want context.Canceled", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Score did not stop after caller cancellation")
	}
}

// More than one entry leaves no defined answer to "which is the score", an
// empty array carries no score at all, and a bare number says nothing about
// which number it is. All fail rather than guess; the accepted shapes are
// covered in complexity_remote_backend_shape_test.go.
func TestScoringHTTPBackend_RejectsAmbiguousResponses(t *testing.T) {
	cases := map[string]string{
		"empty array":    `[]`,
		"two entries":    `[{"label":"a","score":0.1},{"label":"b","score":0.9}]`,
		"bare scalar":    `0.5`,
		"malformed json": `{`,
	}

	for name, body := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(body))
		}))

		backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
			ModelEndpoint: endpointForTestServer(t, server),
			ModelName:     "difficulty-scorer-svc",
		}, time.Second)
		if err != nil {
			server.Close()
			t.Fatalf("%s: newScoringHTTPBackend: %v", name, err)
		}
		if _, err := backend.Score(context.Background(), "text"); err == nil {
			t.Errorf("%s: expected an error", name)
		}
		server.Close()
	}
}

// score.v1 needs no label mapping, so it must not inherit the two-label
// minimum the label-distribution constructor enforces.
func TestScoringHTTPBackend_NeedsNoLabelMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"label":"LABEL_0","score":1.5}]`))
	}))
	defer server.Close()

	backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-scorer-svc",
	}, time.Second)
	if err != nil {
		t.Fatalf("a score backend must construct without a label mapping: %v", err)
	}

	// A score is not a probability: values outside [0,1] are legitimate for a
	// regression head, and the boundaries decide what they mean.
	score, err := backend.Score(context.Background(), "text")
	if err != nil {
		t.Fatalf("Score: %v", err)
	}
	if score != 1.5 {
		t.Fatalf("score = %v, want 1.5 preserved unclamped", score)
	}
}

func TestScoringHTTPBackend_RequiresAnEndpoint(t *testing.T) {
	if _, err := newScoringHTTPBackend(nil, time.Second); err == nil {
		t.Error("expected a nil external model to be rejected")
	}
	if _, err := newScoringHTTPBackend(&config.ExternalModelConfig{ModelName: "x"}, time.Second); err == nil {
		t.Error("expected a missing endpoint address to be rejected")
	}
}

// A response with no score field, or an explicit null, must not read as a
// legitimate 0. Zero is a meaningful score - for a [0,1] scorer it is the
// easiest possible request - so silently substituting it turns a malformed
// response into a confident verdict.
func TestScoringHTTPBackend_RejectsAMissingScore(t *testing.T) {
	cases := map[string]string{
		"field absent":  `[{"label":"LABEL_0"}]`,
		"explicit null": `[{"label":"LABEL_0","score":null}]`,
	}

	for name, body := range cases {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(body))
		}))

		backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
			ModelEndpoint: endpointForTestServer(t, server),
			ModelName:     "difficulty-scorer-svc",
		}, time.Second)
		if err != nil {
			server.Close()
			t.Fatalf("%s: newScoringHTTPBackend: %v", name, err)
		}
		if _, err := backend.Score(context.Background(), "text"); err == nil {
			t.Errorf("%s: expected a missing score to be rejected, not read as 0", name)
		}
		server.Close()
	}
}

// A genuine zero must still be accepted - it is the easiest end of a [0,1]
// scorer's range, not an error.
func TestScoringHTTPBackend_AcceptsAGenuineZero(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"label":"LABEL_0","score":0}]`))
	}))
	defer server.Close()

	backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
		ModelEndpoint: endpointForTestServer(t, server),
		ModelName:     "difficulty-scorer-svc",
	}, time.Second)
	if err != nil {
		t.Fatalf("newScoringHTTPBackend: %v", err)
	}

	score, err := backend.Score(context.Background(), "text")
	if err != nil {
		t.Fatalf("a reported zero is a valid score: %v", err)
	}
	if score != 0 {
		t.Fatalf("score = %v, want 0", score)
	}
}

// A non-finite score cannot be compared against any boundary.
func TestScoringHTTPBackend_RejectsNonFiniteScores(t *testing.T) {
	for name, body := range map[string]string{
		"NaN string": `[{"label":"LABEL_0","score":"NaN"}]`,
		"inf string": `[{"label":"LABEL_0","score":"Infinity"}]`,
	} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(body))
		}))
		backend, err := newScoringHTTPBackend(&config.ExternalModelConfig{
			ModelEndpoint: endpointForTestServer(t, server),
			ModelName:     "difficulty-scorer-svc",
		}, time.Second)
		if err != nil {
			server.Close()
			t.Fatalf("%s: newScoringHTTPBackend: %v", name, err)
		}
		if _, err := backend.Score(context.Background(), "text"); err == nil {
			t.Errorf("%s: expected a non-finite score to be rejected", name)
		}
		server.Close()
	}
}
