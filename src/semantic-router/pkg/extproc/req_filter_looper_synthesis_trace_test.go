package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestFusionSynthesisTraceThroughExtProc(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model string `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		writeFusionReplayCompletion(w, request.Model, []map[string]interface{}{{"index": 0, "message": map[string]interface{}{"role": "assistant", "content": "A useful answer"}, "finish_reason": "stop"}}, map[string]int64{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3})
	}))
	defer server.Close()
	for _, stream := range []bool{false, true} {
		mode := "buffered"
		if stream {
			mode = "streaming"
		}
		t.Run(mode, func(t *testing.T) {
			for _, trace := range []bool{false, true} {
				name := "traces_disabled_control"
				if trace {
					name = "default_trace_enabled"
				}
				t.Run(name, func(t *testing.T) {
					router, ctx, request := quorumFallbackRouter(server.URL)
					ctx.ExpectStreamingResponse = stream
					request.Stream = stream
					decision := &config.Decision{Name: "release-fusion", Algorithm: &config.AlgorithmConfig{Type: config.DecisionAlgorithmFusion, Fusion: &config.FusionAlgorithmConfig{Model: "judge", AnalysisModels: []string{"panel-a", "panel-b"}, MinSuccessfulResponses: 1, AnalysisMode: config.FusionAnalysisModeNone}}}
					if !trace {
						off := false
						decision.Algorithm.Fusion.IncludeAnalysis = &off
						decision.Algorithm.Fusion.IncludeIntermediateResponses = &off
					}
					out, err := router.handleLooperExecution(context.Background(), request, decision, ctx)
					if err != nil {
						t.Fatal(err)
					}
					immediate := out.GetImmediateResponse()
					t.Logf("status=%d body=%s", immediate.GetStatus().GetCode(), immediate.GetBody())
					if trace && !strings.Contains(string(immediate.GetBody()), `"fusion"`) {
						t.Errorf("public Fusion trace missing: %s", immediate.GetBody())
					}
					if immediate.GetStatus().GetCode() != 200 {
						t.Errorf("served synthesis should return200, got%d", immediate.GetStatus().GetCode())
					}
				})
			}
		})
	}
}
