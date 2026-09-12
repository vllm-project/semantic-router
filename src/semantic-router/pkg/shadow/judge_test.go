package shadow

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func testJudgeServer(t *testing.T, reply string, delay time.Duration) (*httptest.Server, *sync.Map) {
	t.Helper()
	var received sync.Map // key "body" -> raw request JSON
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]interface{}
		_ = json.NewDecoder(r.Body).Decode(&payload)
		received.Store("payload", payload)
		if delay > 0 {
			time.Sleep(delay)
		}
		w.Header().Set("Content-Type", "application/json")
		body := `{"id":"j","object":"chat.completion","created":1,"model":"judge","choices":[{"index":0,` +
			`"message":{"role":"assistant","content":` + reply + `},"finish_reason":"stop"}],"usage":{}}`
		if _, err := w.Write([]byte(body)); err != nil {
			t.Errorf("write judge body: %v", err)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &received
}

func judgeFixtures() (config.ShadowJudgeConfig, []config.ShadowArmConfig, []ArmResult) {
	cfg := config.ShadowJudgeConfig{Enabled: true, Model: "judge-m", Endpoint: "http://unused", RubricVersion: "v1"}
	arms := []config.ShadowArmConfig{
		{Name: "arm-a", Model: "model-a", Endpoint: "http://a"},
		{Name: "arm-b", Model: "model-b", Endpoint: "http://b"},
	}
	results := []ArmResult{
		{Arm: "arm-a", Model: "model-a", Outcome: OutcomeCompleted, Content: "answer one"},
		{Arm: "arm-b", Model: "model-b", Outcome: OutcomeCompleted, Content: "answer two"},
		{Arm: "arm-c", Model: "model-c", Outcome: OutcomeFailed, Err: "boom"},
	}
	return cfg, arms, results
}

func TestJudgeBlindIDsStable(t *testing.T) {
	cfg, arms, _ := judgeFixtures()
	judge := NewJudge(cfg, arms)
	if judge.BlindID("arm-a") != "arm-1" || judge.BlindID("arm-b") != "arm-2" {
		t.Fatalf("blind ids not derived in config order: a=%s b=%s", judge.BlindID("arm-a"), judge.BlindID("arm-b"))
	}
	// Stable across instances (dataset correlation), never exposing the name.
	judge2 := NewJudge(cfg, arms)
	for _, name := range []string{"arm-a", "arm-b"} {
		if judge.BlindID(name) != judge2.BlindID(name) {
			t.Fatalf("blind id for %s not stable across judge instances", name)
		}
		if strings.Contains(judge.BlindID(name), name) {
			t.Fatalf("blind id %q leaks arm name %q", judge.BlindID(name), name)
		}
	}
}

func TestJudgeInsufficientArms(t *testing.T) {
	cfg, arms, _ := judgeFixtures()
	judge := NewJudge(cfg, arms)
	decision := judge.Decide(context.Background(), "q", []ArmResult{
		{Arm: "arm-a", Model: "m", Outcome: OutcomeCompleted},
		{Arm: "arm-b", Model: "m", Outcome: OutcomeFailed, Err: "x"},
	})
	if decision.Outcome != JudgeInsufficientArms {
		t.Fatalf("single completed arm must be insufficient, got %s", decision.Outcome)
	}
}

// TestJudgeWinnerBlind assumes the hard isolation contract: the judge payload
// carried only opaque arm ids (never model/provider identity or arm names),
// the configured presentation order was honored, and the winner maps back.
func TestJudgeWinnerBlind(t *testing.T) {
	cfg, arms, results := judgeFixtures()
	srv, received := testJudgeServer(t, `"{\"winner\":\"arm-2\"}"`, 0)
	cfg.Endpoint = srv.URL
	judge := NewJudge(cfg, arms)

	for _, order := range [][]int{{1, 0}, {0, 1}} {
		decision := judge.Decide(context.Background(), "what is best?", results, order)
		if decision.Outcome != JudgeWinner || decision.WinnerArmID != "arm-2" {
			t.Fatalf("order %v: want winner arm-2, got %s winner=%s", order, decision.Outcome, decision.WinnerArmID)
		}
		if decision.JudgeModel != "judge-m" || decision.JudgeRubricVersion != "v1" {
			t.Fatalf("judge version not surfaced: %s@%s", decision.JudgeModel, decision.JudgeRubricVersion)
		}
		payload, _ := received.Load("payload")
		raw, _ := json.Marshal(payload)
		text := string(raw)
		for _, forbidden := range []string{"model-a", "model-b", "arm-a", "arm-b"} {
			if strings.Contains(text, forbidden) {
				t.Fatalf("judge payload leaks %q: %s", forbidden, text)
			}
		}
		for _, id := range []string{"arm-1", "arm-2"} {
			if !strings.Contains(text, id) {
				t.Fatalf("judge payload missing opaque id %q: %s", id, text)
			}
		}
	}
}

func TestJudgeTieAbstainMalformedUnknown(t *testing.T) {
	cases := []struct {
		name   string
		reply  string
		want   JudgeOutcome
		wantID string
	}{
		{name: "tie", reply: `"{\"tie\":[\"arm-1\",\"arm-2\"]}"`, want: JudgeTie},
		{name: "abstain", reply: `"{\"winner\":null}"`, want: JudgeAbstain},
		{name: "malformed", reply: `"not json"`, want: JudgeMalformed},
		{name: "unknown id", reply: `"{\"winner\":\"arm-9\"}"`, want: JudgeMalformed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg, arms, results := judgeFixtures()
			srv, _ := testJudgeServer(t, tc.reply, 0)
			cfg.Endpoint = srv.URL
			judge := NewJudge(cfg, arms)
			decision := judge.Decide(context.Background(), "q", results, []int{0, 1})
			if decision.Outcome != tc.want {
				t.Fatalf("want %s, got %s (%s)", tc.want, decision.Outcome, decision.Reason)
			}
		})
	}
}

func TestJudgeTimeout(t *testing.T) {
	cfg, arms, results := judgeFixtures()
	srv, _ := testJudgeServer(t, `"{\"winner\":\"arm-1\"}"`, 2*time.Second)
	cfg.Endpoint = srv.URL
	cfg.TimeoutSeconds = 1
	judge := NewJudge(cfg, arms)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	decision := judge.Decide(ctx, "q", results, []int{0, 1})
	if decision.Outcome != JudgeTimeout {
		t.Fatalf("want timeout, got %s", decision.Outcome)
	}
}
