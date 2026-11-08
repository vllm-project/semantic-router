package testcases

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const protectionReleaseHeader = "x-vsr-protection-release"

func init() {
	pkgtestcases.Register("protection-explicit-release", pkgtestcases.TestCase{
		Description: "A release header re-decides a held conversation in the same session, bypasses the progress gate, and enforces its interval",
		Tags:        []string{"router-replay", "functional", "session", "learning"},
		Fn:          testProtectionExplicitRelease,
	})
}

func testProtectionExplicitRelease(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
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
	return runProtectionExplicitRelease(ctx, public.BaseURL(), management.BaseURL(), "router-replay-e2e-operator-token")
}

type releaseReplayRecord struct {
	ID             string `json:"id"`
	SessionID      string `json:"session_id"`
	SelectedModel  string `json:"selected_model"`
	ResponseStatus int    `json:"response_status"`
	Learning       struct {
		Protection struct {
			Action  string `json:"action"`
			Release *struct {
				Honored       bool   `json:"honored"`
				Reason        string `json:"reason"`
				PreviousModel string `json:"previous_model"`
				SelectedModel string `json:"selected_model"`
			} `json:"release"`
		} `json:"protection"`
	} `json:"learning"`
}

func runProtectionExplicitRelease(ctx context.Context, public, management, token string) error {
	client := &http.Client{Timeout: 45 * time.Second}
	sid := fmt.Sprintf("release-e2e-%d", time.Now().UnixNano())
	const a, b = "openai/gpt-oss-20b", "openai/shadow-candidate"

	turn := func(keyword string, release bool) (releaseReplayRecord, http.Header, error) {
		headers := map[string]string{"x-session-id": sid, "x-conversation-id": sid, "x-authz-user-id": "release-e2e-user"}
		if release {
			headers[protectionReleaseHeader] = "1"
		}
		body, respHeaders, err := gateHTTPRequest(ctx, client, http.MethodPost, public+"/v1/chat/completions", "", map[string]any{
			"model": "vllm-sr/auto", "max_tokens": 16, "messages": []map[string]string{{"role": "user", "content": keyword}},
		}, headers)
		if err != nil {
			return releaseReplayRecord{}, nil, err
		}
		reply, err := gateBackendReply(body, respHeaders)
		if err != nil {
			return releaseReplayRecord{}, nil, err
		}
		record, err := waitSessionReplay(ctx, client, management, token, sid, reply.replayID, func(r releaseReplayRecord) string { return r.ID })
		if err != nil {
			return releaseReplayRecord{}, nil, err
		}
		if record.SelectedModel != reply.selected || record.ResponseStatus != http.StatusOK {
			return record, nil, fmt.Errorf("routed model %q and replay disagree: %+v", reply.selected, record)
		}
		return record, respHeaders, nil
	}

	seed, _, err := turn("session-gate-seed", false)
	if err != nil {
		return err
	}
	if seed.SelectedModel != a {
		return fmt.Errorf("seed selected %q, want %q", seed.SelectedModel, a)
	}
	held, _, err := turn("session-gate-next", false)
	if err != nil {
		return err
	}
	if held.SelectedModel != a {
		return fmt.Errorf("progress gate did not hold %q without release: %+v", a, held)
	}
	released, _, err := turn("session-gate-next", true)
	if err != nil {
		return err
	}
	rel := released.Learning.Protection.Release
	if released.SelectedModel != b || released.Learning.Protection.Action != "release_switch" ||
		rel == nil || !rel.Honored || rel.PreviousModel != a || rel.SelectedModel != b {
		return fmt.Errorf("release not applied: %+v release=%+v", released, rel)
	}
	// gate_back admits both models and prefers a, so b stays only through the hold.
	newHold, _, err := turn("session-gate-back", false)
	if err != nil {
		return err
	}
	if newHold.SelectedModel != b || newHold.Learning.Protection.Action == "release_switch" || newHold.Learning.Protection.Release != nil {
		return fmt.Errorf("plain turn after release did not hold %q: %+v", b, newHold)
	}
	refused, headers, err := turn("session-gate-back", true)
	if err != nil {
		return err
	}
	if refused.SelectedModel != b {
		return fmt.Errorf("refused release changed the held model: %+v", refused)
	}
	if warnings := headers.Get("x-vsr-protocol-warnings"); !strings.Contains(warnings, "release_interval") {
		return fmt.Errorf("x-vsr-protocol-warnings %q lacks release_interval", warnings)
	}
	if rel := refused.Learning.Protection.Release; rel == nil || rel.Honored || rel.Reason != "release_interval" {
		return fmt.Errorf("refused release replay = %+v", rel)
	}
	for _, record := range []releaseReplayRecord{seed, held, released, newHold, refused} {
		if record.SessionID != sid {
			return fmt.Errorf("release split the run: record %s has session %q, want %q", record.ID, record.SessionID, sid)
		}
	}
	return nil
}
