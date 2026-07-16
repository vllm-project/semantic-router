package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// This is the durable acceptance gate for the forward_authorization_header
// feature and the internal-leg trust boundary (issues #2286 / #2375), replacing
// the manual LiteLLM run. It exercises the gateway-observable scenarios against
// a router deployed with the forward-auth profile config:
//
//   - Direct routing to a forward-auth backend: a caller Authorization is
//     required; its absence is rejected with 401 at the selected backend, and
//     its presence is accepted (200).
//   - Per-backend (mixed static/forward) enforcement: a request to the
//     static-key backend succeeds without any caller Authorization, proving the
//     requirement is enforced per selected backend, not decision-wide.
//   - Spoofed reserved headers: a client that forges x-vsr-looper-request and
//     x-vsr-inbound-authorization cannot bypass the boundary — the forged
//     carrier is ignored (the forward backend still 401s) and the request is not
//     treated as an internal looper request.
//
// The security-path assertions are on router-emitted behavior (401) produced
// BEFORE any upstream call. The success paths (200) additionally assert the
// Authorization that actually reached the upstream: the profile's backends run
// the mock-vllm image, which echoes the received Authorization back on the
// x-echo-authorization response header. This proves the caller's token is
// forwarded verbatim to a forward_authorization_header backend (issue #2286) and
// that the static-key backend receives the injected key instead of the caller's.
//
// Looper re-dispatch enforcement (the caller-Authorization requirement surviving
// the internal leg) is covered by the Go unit tests in pkg/extproc — see
// TestBuildRouteHeaderStateForwardOnLooperLeg* and
// TestHandleLooperExecutionMixedDecisionNotRejectedUpFront. It is intentionally
// NOT asserted here: a looper leg's 401 surfaces to the client edge as a 500
// (the looper client treats any non-200 as a failure), and the profile's looper
// endpoint cannot be resolved until cluster-provision time, so an in-cluster
// looper assertion here would be unreliable.
//
// The forward-auth profile config (e2e/profiles/forward-auth/values.yaml) must
// define model "forward-model" (backend with forward_authorization_header: true)
// and model "static-model" (backend with a static api_key).
const (
	forwardAuthForwardModel = "forward-model"
	forwardAuthStaticModel  = "static-model"
	forwardAuthCallerToken  = "Bearer caller-virtual-key-e2e"
	// forwardAuthCallerSecret / forwardAuthStaticSecret are the distinguishing
	// substrings the upstream must echo back on x-echo-authorization. The caller
	// secret must appear when the caller's token is forwarded; the static secret
	// (the api_key configured on static-be in values.yaml) must appear when the
	// router injects the static key instead.
	forwardAuthCallerSecret = "caller-virtual-key-e2e"
	forwardAuthStaticSecret = "some-static-key"
	// echoAuthHeader is the response header mock-vllm sets to the Authorization it
	// received (see tools/mock-vllm/app.py).
	echoAuthHeader = "x-echo-authorization"
)

func init() {
	pkgtestcases.Register("forward-authorization", pkgtestcases.TestCase{
		Description: "Forward per-request Authorization to opt-in backends; enforce it per-leg; reject spoofed internal headers",
		Tags:        []string{"forward-auth", "security", "authz"},
		Fn:          testForwardAuthorization,
	})
}

type forwardAuthCase struct {
	name           string
	model          string
	authorization  string            // caller Authorization; empty = omit
	extraHeaders   map[string]string // e.g. spoofed reserved headers
	expectRejected bool              // true = expect 401
	// wantAuthSubstr, when set on a success (200) case, is asserted to be present
	// in the Authorization the upstream echoed back on x-echo-authorization.
	wantAuthSubstr string
}

func testForwardAuthorization(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing forward_authorization_header + internal-leg trust boundary")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []forwardAuthCase{
		{
			name:           "direct forward backend with caller Authorization is accepted",
			model:          forwardAuthForwardModel,
			authorization:  forwardAuthCallerToken,
			expectRejected: false,
			// The upstream must receive the caller's verbatim token, not a static key.
			wantAuthSubstr: forwardAuthCallerSecret,
		},
		{
			name:           "direct forward backend without Authorization is rejected 401",
			model:          forwardAuthForwardModel,
			authorization:  "",
			expectRejected: true,
		},
		{
			name:           "mixed candidates: static-key backend needs no caller Authorization",
			model:          forwardAuthStaticModel,
			authorization:  "",
			expectRejected: false,
			// The upstream must receive the router-injected static key.
			wantAuthSubstr: forwardAuthStaticSecret,
		},
		{
			name:          "spoofed reserved headers cannot forge the caller identity",
			model:         forwardAuthForwardModel,
			authorization: "", // caller sent NO real Authorization
			extraHeaders: map[string]string{
				// Forge the internal path + a stolen caller identity. The trust
				// boundary must strip these, so the forward backend still 401s.
				"x-vsr-looper-request":        "true",
				"x-vsr-inbound-authorization": "Bearer stolen-victim-key",
			},
			expectRejected: true,
		},
	}

	total := 0
	passed := 0
	for _, tc := range cases {
		total++
		ok, detail := runForwardAuthCase(ctx, tc, localPort)
		if ok {
			passed++
			if opts.Verbose {
				fmt.Printf("[Test] ✓ %s\n", tc.name)
			}
		} else {
			fmt.Printf("[Test] ✗ %s: %s\n", tc.name, detail)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":  total,
			"passed_tests": passed,
			"failed_tests": total - passed,
		})
	}

	if passed != total {
		return fmt.Errorf("forward-authorization: %d/%d cases passed", passed, total)
	}
	return nil
}

// runForwardAuthCase returns (pass, detail). It asserts only router-observable
// behavior: a rejected case must be a 401; a non-rejected case must be anything
// but 401 (the upstream sim answers the success paths).
func runForwardAuthCase(ctx context.Context, tc forwardAuthCase, localPort string) (bool, string) {
	body := map[string]interface{}{
		"model":    tc.model,
		"messages": []map[string]string{{"role": "user", "content": "forward-auth e2e probe"}},
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return false, fmt.Sprintf("marshal request: %v", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return false, fmt.Sprintf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if tc.authorization != "" {
		req.Header.Set("Authorization", tc.authorization)
	}
	for k, v := range tc.extraHeaders {
		req.Header.Set(k, v)
	}

	httpClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return false, fmt.Sprintf("send request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	respBody, _ := io.ReadAll(resp.Body)

	if tc.expectRejected {
		if resp.StatusCode == http.StatusUnauthorized {
			return true, ""
		}
		return false, fmt.Sprintf("expected 401, got %d: %s", resp.StatusCode, truncate(respBody))
	}
	// Success paths must reach the upstream sim and return 200; a non-200 (401,
	// 5xx from a dead backend, etc.) is a failure, so a broken deployment cannot
	// pass vacuously.
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Sprintf("expected 200, got %d: %s", resp.StatusCode, truncate(respBody))
	}
	// Assert the Authorization the upstream actually received (echoed by mock-vllm).
	// This is what makes the 200 path meaningful: it proves the caller's token was
	// forwarded verbatim, or the static key injected, rather than dropped.
	if tc.wantAuthSubstr != "" {
		gotAuth := resp.Header.Get(echoAuthHeader)
		if !strings.Contains(gotAuth, tc.wantAuthSubstr) {
			return false, fmt.Sprintf("upstream received Authorization %q, want it to contain %q", gotAuth, tc.wantAuthSubstr)
		}
	}
	return true, ""
}

func truncate(b []byte) string {
	const max = 200
	if len(b) > max {
		return string(b[:max]) + "…"
	}
	return string(b)
}
