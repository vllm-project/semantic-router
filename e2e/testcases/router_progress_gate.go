package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const gateFixtureCalibration = "fixture/session-gate@1.0.0"

func init() {
	pkgtestcases.Register("progress-gate-evidence-to-switch", pkgtestcases.TestCase{
		Description: "Real requests and authenticated outcomes drive bounded evidence, enforced switches, cooldown, and Replay",
		Tags:        []string{"router-replay", "functional", "session", "learning"},
		Fn:          testProgressGateVertical,
	})
}

func testProgressGateVertical(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	public, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer public.Close()
	management, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer management.Close()
	return runProgressGateVertical(ctx, public.BaseURL(), management.BaseURL(), "router-replay-e2e-operator-token", "enforce")
}

type gateReplayVerdict struct {
	Decision          string `json:"decision"`
	Reason            string `json:"suppression_reason"`
	CalibrationID     string `json:"calibration_id"`
	EvidenceVersion   string `json:"evidence_version"`
	Enforced          bool   `json:"enforced"`
	Applied           bool   `json:"applied"`
	FinalModel        string `json:"final_model"`
	WindowCount       int    `json:"window_count"`
	AttributableCount int    `json:"attributable_count"`
	RegressionStreak  int    `json:"regression_streak"`
	Switches          int    `json:"switches_in_window"`
	LastSwitchKnown   bool   `json:"last_switch_known"`
}

type gateReplayRecord struct {
	ID             string `json:"id"`
	SelectedModel  string `json:"selected_model"`
	ResponseStatus int    `json:"response_status"`
	SessionPolicy  struct {
		Gate   *gateReplayVerdict `json:"switch_gate"`
		Rescue *gateReplayVerdict `json:"rescue_switch_gate"`
	} `json:"session_policy"`
}

func runProgressGateVertical(ctx context.Context, public, management, token, mode string) error {
	client := &http.Client{Timeout: 45 * time.Second}
	sid := fmt.Sprintf("gate-e2e-%d", time.Now().UnixNano())
	const a, b = "openai/gpt-oss-20b", "openai/shadow-candidate"
	post := func(path string, payload any, headers map[string]string) ([]byte, error) {
		body, _, err := gateHTTPRequest(ctx, client, http.MethodPost, path, token, payload, headers)
		return body, err
	}
	turn := func(keyword string, turnIndex int) (gateReplayRecord, error) {
		body, headers, err := gateHTTPRequest(ctx, client, http.MethodPost, public+"/v1/chat/completions", "", map[string]any{
			"model": "auto", "max_tokens": 16, "messages": []map[string]string{{"role": "user", "content": keyword}},
		}, map[string]string{"x-session-id": sid, "x-conversation-id": sid, "x-authz-user-id": "gate-e2e-user"})
		if err != nil {
			return gateReplayRecord{}, err
		}
		var reply struct {
			Model string `json:"model"`
			Usage struct {
				CompletionTokens int `json:"completion_tokens"`
			} `json:"usage"`
		}
		if err = json.Unmarshal(body, &reply); err != nil {
			return gateReplayRecord{}, err
		}
		selected, replayID := headers.Get("x-vsr-selected-model"), headers.Get("x-vsr-replay-id")
		if selected == "" || replayID == "" || reply.Model == "" || reply.Usage.CompletionTokens <= 0 {
			return gateReplayRecord{}, fmt.Errorf("missing routing identity or backend output: headers=%v body=%s", headers, body)
		}
		deadline := time.NewTimer(15 * time.Second)
		defer deadline.Stop()
		tick := time.NewTicker(100 * time.Millisecond)
		defer tick.Stop()
		for {
			data, _, err := gateHTTPRequest(ctx, client, http.MethodGet, management+"/v1/router_replay?showDetails=true&limit=30&session_id="+url.QueryEscape(sid), token, nil, nil)
			if err != nil {
				return gateReplayRecord{}, err
			}
			var list struct {
				Data []gateReplayRecord `json:"data"`
			}
			if err = json.Unmarshal(data, &list); err != nil {
				return gateReplayRecord{}, err
			}
			for _, record := range list.Data {
				if record.ID != replayID {
					continue
				}
				// Providers may return their primary alias instead of the requested logical name.
				if record.SelectedModel != selected || record.ResponseStatus != 200 {
					return record, fmt.Errorf("turn %d: routed model %q and replay disagree: %+v", turnIndex, selected, record)
				}
				fmt.Printf("session=%s turn=%d model=%s tokens=%d gate=%+v\n", sid, turnIndex, record.SelectedModel, reply.Usage.CompletionTokens, record.SessionPolicy.Gate)
				return record, nil
			}
			select {
			case <-ctx.Done():
				return gateReplayRecord{}, ctx.Err()
			case <-deadline.C:
				return gateReplayRecord{}, fmt.Errorf("turn %d missing replay", turnIndex)
			case <-tick.C:
			}
		}
	}
	feedback := func(record gateReplayRecord) error {
		payload := map[string]any{"replay_id": record.ID, "target": "model", "target_ref": record.SelectedModel, "verdict": "underpowered", "score": 0.1}
		for attempt := 0; attempt < 2; attempt++ {
			body, err := post(management+"/v1/router/outcomes", payload, map[string]string{"Idempotency-Key": record.ID})
			if err != nil {
				return err
			}
			var ack struct {
				Success   bool `json:"success"`
				Duplicate bool `json:"duplicate"`
			}
			if err = json.Unmarshal(body, &ack); err != nil {
				return err
			}
			if !ack.Success || attempt == 1 && !ack.Duplicate {
				return fmt.Errorf("feedback not accepted/idempotent: %s", body)
			}
		}
		return nil
	}
	check := func(record gateReplayRecord, model, decision, reason string, count int) error {
		g := record.SessionPolicy.Gate
		if g == nil || g.CalibrationID != gateFixtureCalibration || g.EvidenceVersion == "" ||
			g.Decision != decision || g.Reason != reason || g.Enforced != (mode == "enforce") ||
			g.WindowCount != count || g.AttributableCount != count || record.SelectedModel != model || g.FinalModel != model {
			return fmt.Errorf("want model=%s gate=%s/%s window=%d mode=%s; got record=%+v gate=%+v", model, decision, reason, count, mode, record, g)
		}
		if mode == "enforce" && decision == "suppress" && !g.Applied {
			return fmt.Errorf("suppression not applied: %+v", g)
		}
		return nil
	}

	checkObserve := func(record gateReplayRecord, model string, index int) error {
		if record.SelectedModel != model {
			return fmt.Errorf("turn %d: observe want model=%s got %s", index, model, record.SelectedModel)
		}
		g := record.SessionPolicy.Gate
		if g == nil {
			return nil
		}
		if g.Enforced || g.Applied {
			return fmt.Errorf("turn %d: observe applied a verdict: %+v", index, g)
		}
		if g.FinalModel != model {
			return fmt.Errorf("turn %d: observe final model mismatch: %+v", index, g)
		}
		return nil
	}

	seed, err := turn("session-gate-seed", 1)
	if err != nil {
		return err
	}
	if seed.SelectedModel != a || seed.SessionPolicy.Gate != nil {
		return fmt.Errorf("initial selection is not a switch: %+v", seed)
	}
	if err = feedback(seed); err != nil {
		return err
	}
	second, err := turn("session-gate-next", 2)
	if err != nil {
		return err
	}
	want := a
	if mode == "observe" {
		want = b
	}
	if err = check(second, want, "suppress", "insufficient_evidence", 1); err != nil {
		return err
	}
	if mode == "observe" {
		// Observe runs the same seven-turn script as enforce: verdicts are
		// recorded but never applied, so accepted proposals still execute and
		// the visible model sequence diverges from the enforced run.
		if second.SessionPolicy.Gate.Applied {
			return fmt.Errorf("observe modified the proposal: %+v", second.SessionPolicy.Gate)
		}
		if err = feedback(second); err != nil {
			return err
		}
		for _, step := range []struct {
			keyword string
			index   int
			model   string
		}{
			{"session-gate-forward", 3, b},
			{"session-gate-next", 4, b},
			{"session-gate-back", 5, a},
		} {
			rec, turnErr := turn(step.keyword, step.index)
			if turnErr != nil {
				return turnErr
			}
			if err = checkObserve(rec, step.model, step.index); err != nil {
				return err
			}
			if err = feedback(rec); err != nil {
				return err
			}
		}
		if err = gateWait(ctx, 11*time.Second); err != nil {
			return err
		}
		sixth, err := turn("session-gate-return", 6)
		if err != nil {
			return err
		}
		if err = checkObserve(sixth, a, 6); err != nil {
			return err
		}
		if err = feedback(sixth); err != nil {
			return err
		}
		if err = gateWait(ctx, 11*time.Second); err != nil {
			return err
		}
		seventh, err := turn("session-gate-next", 7)
		if err != nil {
			return err
		}
		if err = checkObserve(seventh, b, 7); err != nil {
			return err
		}
		if seventh.SessionPolicy.Gate.Switches != 2 {
			return fmt.Errorf("observe window switch count: %+v", seventh.SessionPolicy.Gate)
		}
		return nil
	}
	if err = feedback(second); err != nil {
		return err
	}
	third, err := turn("session-gate-forward", 3)
	if err != nil {
		return err
	}
	if err = check(third, a, "suppress", "insufficient_evidence", 2); err != nil {
		return err
	}
	if err = feedback(third); err != nil {
		return err
	}
	fourth, err := turn("session-gate-next", 4)
	if err != nil {
		return err
	}
	if err = check(fourth, b, "switch", "", 3); err != nil {
		return err
	}
	if err = feedback(fourth); err != nil {
		return err
	}
	fifth, err := turn("session-gate-back", 5)
	if err != nil {
		return err
	}
	if err = check(fifth, b, "suppress", "cooldown", 4); err != nil {
		return err
	}
	if !fifth.SessionPolicy.Gate.LastSwitchKnown || fifth.SessionPolicy.Gate.Switches != 1 {
		return fmt.Errorf("switch history not visible: %+v", fifth.SessionPolicy.Gate)
	}
	if err = feedback(fifth); err != nil {
		return err
	}
	if err = gateWait(ctx, 11*time.Second); err != nil {
		return err
	}
	sixth, err := turn("session-gate-return", 6)
	if err != nil {
		return err
	}
	if err = check(sixth, a, "switch", "", 5); err != nil {
		return err
	}
	if err = feedback(sixth); err != nil {
		return err
	}
	if err = gateWait(ctx, 11*time.Second); err != nil {
		return err
	}
	seventh, err := turn("session-gate-next", 7)
	if err != nil {
		return err
	}
	if err = check(seventh, a, "suppress", "oscillation_guard", 6); err != nil {
		return err
	}
	if seventh.SessionPolicy.Gate.Switches != 2 {
		return fmt.Errorf("window switch count: %+v", seventh.SessionPolicy.Gate)
	}
	return nil
}

func gateWait(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func gateHTTPRequest(ctx context.Context, client *http.Client, method, endpoint, token string, payload any, headers map[string]string) ([]byte, http.Header, error) {
	var body []byte
	if payload != nil {
		var err error
		body, err = json.Marshal(payload)
		if err != nil {
			return nil, nil, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, resp.Header, fmt.Errorf("%s %s: HTTP %d: %s", method, endpoint, resp.StatusCode, data)
	}
	return data, resp.Header, nil
}
