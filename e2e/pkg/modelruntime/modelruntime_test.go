package modelruntime

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestParseMetricsReadsLabelsEscapesAndSpecialValues(t *testing.T) {
	text := strings.Join([]string{
		"# HELP vsr_model_runtime_ready Whether a deployment answers.",
		"# TYPE vsr_model_runtime_ready gauge",
		`vsr_model_runtime_ready{deployment="vela-domain"} 1`,
		`vsr_model_runtime_ready{deployment="offline"} 0`,
		`vsr_model_runtime_restarts_total{deployment="vela-domain",reason="exit"} 2`,
		`vsr_model_runtime_restarts_total{deployment="vela-domain",reason="health"} 1`,
		`quoted{path="a\"b\\c",empty=""} +Inf 1700000000`,
		"plain_total 3",
	}, "\n")

	metrics, err := ParseMetrics(text)
	if err != nil {
		t.Fatal(err)
	}
	if !metrics.DeploymentReady("vela-domain") || metrics.DeploymentReady("offline") || metrics.DeploymentReady("absent") {
		t.Fatalf("readiness misread: %+v", metrics.Samples)
	}
	if got := metrics.DeploymentRestarts("vela-domain"); got != 3 {
		t.Fatalf("restarts = %v, want 3", got)
	}
	value, ok := metrics.Value("quoted", map[string]string{"path": `a"b\c`, "empty": ""})
	if !ok || value < 1e308 {
		t.Fatalf("escaped label sample = %v, %v", value, ok)
	}
	if value, ok := metrics.Value("plain_total", nil); !ok || value != 3 {
		t.Fatalf("plain sample = %v, %v", value, ok)
	}
}

func TestParseMetricsRejectsMalformedSamples(t *testing.T) {
	for _, line := range []string{`name{label="x" 1`, `name{label=x} 1`, "lonely", `name{} nan-ish`} {
		if _, err := ParseMetrics(line); err == nil {
			t.Fatalf("%q parsed without error", line)
		}
	}
}

func TestEventuallyReturnsOnFirstSuccessAndExplainsTimeouts(t *testing.T) {
	calls := 0
	err := Eventually(context.Background(), time.Second, func(context.Context) error {
		calls++
		if calls < 3 {
			return errors.New("not yet")
		}
		return nil
	})
	if err != nil || calls != 3 {
		t.Fatalf("Eventually = %v after %d calls", err, calls)
	}

	err = Eventually(context.Background(), 300*time.Millisecond, func(context.Context) error {
		return errors.New("still starting")
	})
	if err == nil || !strings.Contains(err.Error(), "still starting") {
		t.Fatalf("timeout error %v does not carry the last check error", err)
	}
}

func TestClientDecodesSurfacesAndRequiresEveryAnswer(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body map[string]interface{}
		if request.Body != nil {
			_ = json.NewDecoder(request.Body).Decode(&body)
		}
		switch request.URL.Path {
		case "/health":
			writer.WriteHeader(http.StatusServiceUnavailable)
			_, _ = writer.Write([]byte(`{"status":"warming","models":{"vela-domain":{"status":"warming"}}}`))
		case "/v1/models":
			_, _ = writer.Write([]byte(`{"object":"list","data":[{"id":"vela-domain","object":"model","family":"task_heads",
				"surfaces":["classify"],"ready":true,"heads":[{"name":"default","kind":"sequence","labels":["code","math"]}]}]}`))
		case "/v1/classify":
			if body["model"] != "vela-domain" {
				t.Errorf("classify body %v", body)
			}
			_, _ = writer.Write([]byte(`{"model":"vela-domain","head":"default","kind":"sequence","labels":["code","math"],
				"results":[{"index":0,"label":"math","probabilities":[0.25,0.75]}],"usage":{"input_tokens":4,"output_tokens":0}}`))
		case "/v1/decisions":
			_, _ = writer.Write([]byte(`{"answers":{"kind":{"choice":"code","probabilities":{"code":0.9}}}}`))
		default:
			writer.WriteHeader(http.StatusNotFound)
			_, _ = writer.Write([]byte(`{"error":{"code":"model_not_found","message":"no"}}`))
		}
	}))
	defer server.Close()
	client := NewHTTPClient(server.URL, 5*time.Second)
	ctx := context.Background()

	if _, status, err := client.Health(ctx); err != nil || status != http.StatusServiceUnavailable {
		t.Fatalf("health = %d, %v", status, err)
	}
	card, err := client.Model(ctx, "vela-domain")
	if err != nil || !card.HasSurface("classify") || card.Heads[0].Labels[1] != "math" {
		t.Fatalf("card = %+v, %v", card, err)
	}
	if _, err := client.Model(ctx, "missing"); err == nil {
		t.Fatal("a missing model resolved")
	}
	classified, err := client.Classify(ctx, ClassifyRequest{Model: "vela-domain", Input: []string{"2+2"}})
	if err != nil {
		t.Fatal(err)
	}
	if probability, ok := classified.Results[0].Probability(classified.Labels, "math"); !ok || probability != 0.75 {
		t.Fatalf("math probability = %v, %v", probability, ok)
	}
	questions := map[string]Question{"kind": {Type: "choice"}, "hard": {Type: "noul"}}
	if _, err := client.Decide(ctx, DecisionsRequest{State: "x", Questions: questions}); err == nil {
		t.Fatal("a missing answer was accepted")
	}
	var statusErr *StatusError
	if _, err := client.Rerank(ctx, RerankRequest{Query: "q", Documents: []string{"d"}}); !errors.As(err, &statusErr) || statusErr.Status != http.StatusNotFound {
		t.Fatalf("rerank error = %v", err)
	}
}

func TestParseSocketClientOutputSplitsStatusAndBody(t *testing.T) {
	status, payload, err := parseSocketClientOutput([]byte("503\n{\"status\":\"loading\"}"))
	if err != nil || status != http.StatusServiceUnavailable || string(payload) != `{"status":"loading"}` {
		t.Fatalf("parsed %d %q %v", status, payload, err)
	}
	for _, output := range []string{"", "no newline", "abc\n{}"} {
		if _, _, err := parseSocketClientOutput([]byte(output)); err == nil {
			t.Fatalf("%q parsed without error", output)
		}
	}
}
