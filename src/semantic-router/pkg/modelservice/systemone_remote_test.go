package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestEncodeSystemOneRequestSendsCriteria(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	threshold := 0.4
	body, err := EncodeSystemOneRequest(ctx, Request{
		Model: "vllm-sr/d3-ultra", State: "Our app has returned 500 errors since the 09:00 deploy.", MaxTokens: 4096,
		Questions: []Question{
			{ID: "route", Type: "choice", Instructions: "Which team should handle this ticket?", Truncate: true, Choices: []Choice{
				{Key: "engineering", Description: "Outages and bugs"}, {Key: "billing", Description: "Payments and refunds"}, {Key: "other"},
			}},
			{ID: "refund", Type: "noul", Instructions: "Is it a refund?", Choices: []Choice{{Key: "false", Description: "no"}, {Key: "true", Description: "yes"}}},
			{ID: "urgency", Type: "score", Instructions: "How urgent?", Levels: []string{"low", "high"}},
			{ID: "topics", Type: "set", Instructions: "Which topics?", Threshold: &threshold, Labels: []Choice{{Key: "money"}, {Key: "bugs", Description: "defects"}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	raw := string(body)
	for _, want := range []string{
		`"criteria":{"engineering":"Outages and bugs","billing":"Payments and refunds","other":null}`,
		`"criteria":{"false":"no","true":"yes"}`,
		`"criteria":["low","high"]`,
		`"criteria":{"money":null,"bugs":"defects"}`,
	} {
		if !strings.Contains(raw, want) {
			t.Fatalf("missing %s in %s", want, raw)
		}
	}
	for _, unwanted := range []string{`"choices"`, `"levels"`, `"labels"`, `"overflow"`, `"max_tokens"`} {
		if strings.Contains(raw, unwanted) {
			t.Fatalf("System One request carries %s: %s", unwanted, raw)
		}
	}
	var decoded struct {
		Model   string `json:"model"`
		Options struct {
			DeadlineMs float64 `json:"deadline_ms"`
		} `json:"options"`
	}
	if decodeErr := json.Unmarshal(body, &decoded); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	if decoded.Model != "vllm-sr/d3-ultra" || decoded.Options.DeadlineMs <= 0 || decoded.Options.DeadlineMs > 2000 || decoded.Options.DeadlineMs != float64(int(decoded.Options.DeadlineMs)) {
		t.Fatalf("model or deadline: %+v", decoded)
	}

	unbounded, err := EncodeSystemOneRequest(context.Background(), Request{State: "x", Questions: []Question{{ID: "q", Type: "noul", Instructions: "x?"}}})
	if err != nil || strings.Contains(string(unbounded), `"options"`) {
		t.Fatalf("a call without a deadline sends no options: %s %v", unbounded, err)
	}
	expired, cancelExpired := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancelExpired()
	if _, err := EncodeSystemOneRequest(expired, Request{State: "x"}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("an expired call is not sent: %v", err)
	}
}

func TestDecodeSystemOneResponseReadsAnswersAsTheRuntimeDoes(t *testing.T) {
	questions := []Question{
		{ID: "route", Type: "choice", Choices: []Choice{{Key: "billing"}, {Key: "engineering"}}},
		{ID: "refund", Type: "noul"},
		{ID: "urgency", Type: "score", Levels: []string{"not urgent", "somewhat urgent", "very urgent"}},
	}
	body := `{"answers":{"route":{"type":"choice","choice":"engineering","probabilities":{"billing":1.5e-06,"engineering":0.9999985},"confidence":0.999997},` +
		`"refund":{"type":"noul","noul":0.99999},` +
		`"urgency":{"type":"score","score":1.16,"probabilities":{"0":0.0042,"1":0.8316,"2":0.1642},"confidence":0.747,"legend":{"0":"not urgent","1":"somewhat urgent","2":"very urgent"}}},` +
		`"model":"vllm-sr/d3-ultra","usage":{"input_tokens":192,"output_tokens":0}}`
	response, err := DecodeSystemOneResponse(200, []byte(body), questions)
	if err != nil {
		t.Fatal(err)
	}
	route, refund, urgency := response.Answers["route"], response.Answers["refund"], response.Answers["urgency"]
	if route.Error != "" || route.Choice != "engineering" || route.Probabilities["billing"] != 1.5e-06 || route.Confidence != 0.999997 {
		t.Fatalf("choice answer: %+v", route)
	}
	if refund.Error != "" || refund.Noul != 0.99999 {
		t.Fatalf("noul answer: %+v", refund)
	}
	if urgency.Error != "" || urgency.Score != 1.16 || urgency.Probabilities["1"] != 0.8316 {
		t.Fatalf("score answer: %+v", urgency)
	}
	if response.Model != "vllm-sr/d3-ultra" || response.InputTokens != 192 {
		t.Fatalf("response identity: %+v", response)
	}

	unfit := `{"answers":{"route":{"type":"choice","choice":"engineering","probabilities":{"engineering":1}}},"model":"m","usage":{"input_tokens":1}}`
	if response, err := DecodeSystemOneResponse(200, []byte(unfit), questions[:1]); err != nil || response.Answers["route"].Error != "invalid_model_output" {
		t.Fatalf("an answer that does not fit its question is unknown: %+v %v", response.Answers["route"], err)
	}
	for status, want := range map[int]error{401: ErrRejected, 422: ErrRejected, 429: ErrOverloaded, 503: ErrUnavailable, 502: ErrFailed, 504: ErrFailed} {
		if _, err := DecodeSystemOneResponse(status, []byte(`{"error":{"code":"x"}}`), questions); !errors.Is(err, want) {
			t.Fatalf("status %d: %v, want %v", status, err, want)
		}
	}
	if _, err := DecodeSystemOneResponse(200, []byte(`<html>`), questions); !errors.Is(err, ErrFailed) {
		t.Fatalf("a body that is not an answer fails: %v", err)
	}
}

func TestAnAskerAwaitingARemoteModelDoesNotHoldBackItsLocalCall(t *testing.T) {
	runtime := runtimetest.New(jointVela2("vela"))
	lease := attachedLease(t, map[*runtimetest.Runtime][]string{runtime: {"vela"}})
	answered := make(chan struct{})
	held := false
	done := make(chan []error, 1)
	go func() {
		_, errs := runStage(DefaultBundleWindow, asker{deployments: []string{"vela", "remote"}, ask: func(ctx context.Context) error {
			var local error
			Fan(ctx, 2, func(i int) {
				if i == 0 {
					_, local = lease.Decide(ctx, "vela", Request{State: "x", Questions: []Question{choiceQuestion("local")}})
					close(answered)
					return
				}
				release := Await(ctx)
				defer release()
				select {
				case <-answered:
				case <-time.After(3 * time.Second):
					held = true
				}
			})
			return local
		}})
		done <- errs
	}()
	select {
	case errs := <-done:
		if err := errors.Join(errs...); err != nil || held {
			t.Fatalf("errors %v, local call held back by the remote one: %v", err, held)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the stage deadlocked")
	}
}
