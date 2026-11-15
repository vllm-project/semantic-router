package modelservice

import (
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

type meterDoer struct{ body string }

func (d meterDoer) Do(*http.Request) (*http.Response, error) {
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"application/json"}}, Body: io.NopCloser(strings.NewReader(d.body))}, nil
}

func meterClient(t *testing.T, body string) *Client {
	t.Helper()
	generated, err := api.NewClientWithResponses("http://runtime", api.WithHTTPClient(meterDoer{body}))
	if err != nil {
		t.Fatal(err)
	}
	return &Client{api: generated}
}

func TestRequestMeterConcurrentDecisions(t *testing.T) {
	client := meterClient(t, `{"model":"decision","answers":{"q":{"type":"choice","error":"invalid_model_output"}},"usage":{"input_tokens":12,"output_tokens":3}}`)
	ctx, meter := WithRequestMeter(context.Background())
	ctx = WithMeterDeployment(ctx, "primary")
	var wg sync.WaitGroup
	for range 32 {
		wg.Go(func() {
			response, err := client.Decide(ctx, Request{Questions: []Question{{ID: "q", Type: "choice"}}})
			if err != nil || response.Answers["q"].Error == "" {
				t.Errorf("return changed: %+v %v", response, err)
			}
		})
	}
	wg.Wait()
	calls := meter.Calls()
	if len(calls) != 32 {
		t.Fatalf("calls = %d", len(calls))
	}
	for _, call := range calls {
		if call.Deployment != "primary" || call.InputTokens != 12 || call.OutputTokens != 3 || !call.HasUsage {
			t.Fatalf("call = %+v", call)
		}
	}
	// A background call through the same client has no request attribution.
	_, _ = client.Decide(context.Background(), Request{})
	if len(meter.Calls()) != 32 {
		t.Fatal("background call was included")
	}
}

func TestRequestMeterMissingUsageAndEmbeddings(t *testing.T) {
	ctx, meter := WithRequestMeter(context.Background())
	client := meterClient(t, `{"model":"decision","answers":{}}`)
	_, _ = client.Decide(ctx, Request{})
	if calls := meter.Calls(); len(calls) != 1 || calls[0].HasUsage {
		t.Fatalf("calls = %+v", calls)
	}
	reportRuntimeCall(ctx, api.BundleTask{Embeddings: &api.EmbeddingsRequest{}}, api.BundleResult{Embeddings: &api.EmbeddingsResponse{Model: "embed", Usage: api.EmbeddingsUsage{PromptTokens: 7}}})
	calls := meter.Calls()
	if len(calls) != 2 || calls[1].Kind != "embeddings" || calls[1].InputTokens != 7 {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestRequestMeterFusedTaskCountedOnce(t *testing.T) {
	ctx, meter := WithRequestMeter(context.Background())
	task := &bundleTask{task: api.BundleTask{Classify: &api.ClassifyRequest{}}, calls: []*bundleCall{
		{ctx: ctx, done: make(chan struct{})}, {ctx: ctx, done: make(chan struct{})},
	}, sizes: []int{1, 1}}
	task.answer(api.BundleResult{Status: 200, Classify: &api.ClassifyResponse{Usage: api.Usage{InputTokens: 10}}}, nil)
	if calls := meter.Calls(); len(calls) != 1 || calls[0].InputTokens != 10 {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestRequestMeterRemoteUsage(t *testing.T) {
	ctx, meter := WithRequestMeter(context.Background())
	reportRemoteCall(ctx, "remote", "fallback", []byte(`{"model":"system","usage":{"input_tokens":20,"output_tokens":4}}`))
	reportRemoteCall(ctx, "remote", "fallback", nil)
	calls := meter.Calls()
	if len(calls) != 2 || calls[0].Kind != "systemone" || calls[0].Model != "system" || !calls[0].HasUsage || calls[1].HasUsage {
		t.Fatalf("calls = %+v", calls)
	}
}

func TestRequestMeterRemoteUsageRequiresBothCounters(t *testing.T) {
	for _, tc := range []struct {
		name   string
		usage  string
		input  int
		output int
		known  bool
	}{
		{"empty", `{}`, 0, 0, false},
		{"output only", `{"output_tokens":5}`, 0, 5, false},
		{"input only", `{"input_tokens":7}`, 7, 0, false},
		{"null input", `{"input_tokens":null,"output_tokens":5}`, 0, 5, false},
		{"null output", `{"input_tokens":7,"output_tokens":null}`, 7, 0, false},
		{"explicit zeros", `{"input_tokens":0,"output_tokens":0}`, 0, 0, true},
		{"zero output", `{"input_tokens":7,"output_tokens":0}`, 7, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, meter := WithRequestMeter(context.Background())
			reportRemoteCall(ctx, "remote", "system", []byte(`{"usage":`+tc.usage+`}`))
			calls := meter.Calls()
			if len(calls) != 1 || calls[0].HasUsage != tc.known || calls[0].InputTokens != tc.input || calls[0].OutputTokens != tc.output {
				t.Fatalf("calls = %+v, want input=%d output=%d HasUsage=%v", calls, tc.input, tc.output, tc.known)
			}
		})
	}
}
