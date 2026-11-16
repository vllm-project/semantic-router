package embedding

import (
	"context"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

type embeddingMeterTransport struct{ body string }

func (r embeddingMeterTransport) RoundTrip(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(r.body))}, nil
}

func TestRemoteEmbeddingRequestMeter(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		hasUsage   bool
		tokens     int
		wantError  bool
	}{
		{"usage", `{"data":[{"index":0,"embedding":[1,0]}],"usage":{"prompt_tokens":12}}`, true, 12, false},
		{"no usage", `{"data":[{"index":0,"embedding":[1,0]}]}`, false, 0, false},
		{"invalid vector", `{"data":[],"usage":{"prompt_tokens":12}}`, true, 12, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p, err := NewOpenAICompatibleProvider(OpenAICompatibleConfig{BaseURL: "http://embedding/v1", Model: "embed", ExpectedDimension: 2, HTTPClient: &http.Client{Transport: embeddingMeterTransport{tc.body}}})
			if err != nil {
				t.Fatal(err)
			}
			ctx, meter := modelservice.WithRequestMeter(context.Background())
			ctx = modelservice.WithMeterDeployment(ctx, "remote-embedding")
			_, err = p.Embed(ctx, "hello")
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v", err)
			}
			calls := meter.Calls()
			if len(calls) != 1 || calls[0].InputTokens != tc.tokens || calls[0].HasUsage != tc.hasUsage || calls[0].Deployment != "remote-embedding" {
				t.Fatalf("calls = %+v", calls)
			}
			_, _ = p.Embed(context.Background(), "warmup")
			if len(meter.Calls()) != 1 {
				t.Fatal("warmup included")
			}
		})
	}
}
