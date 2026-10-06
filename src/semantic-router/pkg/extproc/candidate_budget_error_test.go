package extproc

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func TestCandidatePoolBudgetRejectionRequiresEveryCandidate(t *testing.T) {
	r := strictCandidateRouter()
	broken := r.Config.ModelConfig["text"]
	broken.APIFormat = "unavailable-wire"
	r.Config.ModelConfig["broken-codec"] = broken
	request := testNeutralRequest("public", "hello")
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(1025)
	demand := selection.DemandForRequest(request)
	for _, test := range []struct {
		name   string
		refs   []config.ModelRef
		budget bool
	}{
		{"all budget", []config.ModelRef{{Model: "text"}, {Model: "vision"}}, true},
		{"unknown model", []config.ModelRef{{Model: "text"}, {Model: "unknown"}}, false},
		{"empty inventory", nil, false},
		{"unavailable codec", []config.ModelRef{{Model: "text"}, {Model: "broken-codec"}}, false},
		{"no capabilities", []config.ModelRef{{Model: "unknown"}}, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := r.eligibleDemandModelRefs(r.Config.CandidateRequirements, test.refs, demand)
			var budget *selection.RequestBudgetError
			if !errors.Is(err, selection.ErrNoEligibleCandidates) || errors.As(err, &budget) != test.budget {
				t.Fatalf("err=%v budget=%v", err, budget)
			}
		})
	}
	request = strictCandidateRequest()
	request.Sampling.MaxOutputTokens = llmprotocol.Int64(1025)
	_, err := r.eligibleDemandModelRefs(r.Config.CandidateRequirements, []config.ModelRef{{Model: "vision"}, {Model: "text"}}, selection.DemandForRequest(request))
	var budget *selection.RequestBudgetError
	if errors.As(err, &budget) {
		t.Fatal("mixed capability and budget failure misclassified")
	}
}

func TestCandidateBudgetRejectionClientAndReplayAgree(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, format := range extProcMatrixFormats {
			t.Run(string(format)+map[bool]string{false: "/buffered", true: "/stream"}[stream], func(t *testing.T) {
				recorder := routerreplay.NewRecorder(store.NewMemoryStore(10, 0))
				r := &OpenAIRouter{ReplayRecorder: recorder}
				replayConfig := config.DefaultRouterReplayPluginConfig()
				replayConfig.Enabled = true
				replayConfig.CaptureResponseBody = true
				ctx := &RequestContext{RequestID: "budget-request", SourceFormat: format, SemanticRequest: testNeutralRequest("entrypoint-model", "hello"), RouterReplayPluginConfig: &replayConfig, ExpectStreamingResponse: stream}
				rejection := &selection.RequestBudgetError{Code: "context_length_exceeded", Message: "estimated input plus requested output exceeds the 32768-token context window"}
				response := r.respondSelectionRejected(ctx, "entrypoint-model", rejection)
				response = r.encodeImmediateResponseForClient(response, ctx)
				immediate := response.GetImmediateResponse()
				if immediate == nil || int(immediate.GetStatus().GetCode()) != 400 {
					t.Fatalf("want HTTP400: %+v", immediate)
				}
				translated, err := protocolcodec.NewBuiltinEngine().TranslateTransportError(format, format, immediate.Body, nil)
				if err != nil || translated.TransportError.Error.Category != llmprotocol.ErrorInvalidRequest || translated.TransportError.Error.Message != rejection.Message {
					t.Fatalf("client lost budget reason: %s (%v)", immediate.Body, err)
				}
				record, ok := recorder.GetRecord(ctx.RouterReplayID)
				if !ok || record.LifecycleState != routerreplay.LifecycleFailed || record.ResponseStatus != 400 || record.TerminalReason != "request_budget_exceeded" || record.ResponseBody != string(immediate.Body) {
					t.Fatalf("replay differs: %+v", record)
				}
			})
		}
	}
}
