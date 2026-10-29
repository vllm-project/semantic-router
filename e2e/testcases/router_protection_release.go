package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
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
	ID            string `json:"id"`
	SessionID     string `json:"session_id"`
	SelectedModel string `json:"selected_model"`
	Learning      struct {
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
		_, respHeaders, err := gateHTTPRequest(ctx, client, http.MethodPost, public+"/v1/chat/completions", "", map[string]any{
			"model": "vllm-sr/auto", "max_tokens": 16, "messages": []map[string]string{{"role": "user", "content": keyword}},
		}, headers)
		if err != nil {
			return releaseReplayRecord{}, nil, err
		}
		replayID := respHeaders.Get("x-vsr-replay-id")
		if replayID == "" {
			return releaseReplayRecord{}, nil, fmt.Errorf("missing replay id: %v", respHeaders)
		}
		record, err := waitReleaseReplay(ctx, client, management, token, sid, replayID)
		return record, respHeaders, err
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
	refused, headers, err := turn("session-gate-seed", true)
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
	for _, record := range []releaseReplayRecord{seed, held, released, refused} {
		if record.SessionID != sid {
			return fmt.Errorf("release split the run: record %s has session %q, want %q", record.ID, record.SessionID, sid)
		}
	}
	return nil
}

func waitReleaseReplay(ctx context.Context, client *http.Client, management, token, sid, replayID string) (releaseReplayRecord, error) {
	deadline := time.NewTimer(15 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	for {
		data, _, err := gateHTTPRequest(ctx, client, http.MethodGet, management+"/api/v1/observability/replays?showDetails=true&limit=30&session_id="+url.QueryEscape(sid), token, nil, nil)
		if err != nil {
			return releaseReplayRecord{}, err
		}
		var list struct {
			Data []releaseReplayRecord `json:"data"`
		}
		if err = json.Unmarshal(data, &list); err != nil {
			return releaseReplayRecord{}, err
		}
		for _, record := range list.Data {
			if record.ID == replayID {
				return record, nil
			}
		}
		select {
		case <-ctx.Done():
			return releaseReplayRecord{}, ctx.Err()
		case <-deadline.C:
			return releaseReplayRecord{}, fmt.Errorf("replay %s missing", replayID)
		case <-tick.C:
		}
	}
}
