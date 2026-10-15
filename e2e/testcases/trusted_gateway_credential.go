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

// The trusted gateway credential contract (issue #2286), observed at the
// gateway edge of the trusted-gateway profile. The profile's Lua filter stands
// in for the deployment authenticator: it strips client-supplied
// authentication outputs and, for alice's client token, attaches
// x-authz-user-id and the gateway credential gw-virtual-key-alice. Both
// backends are provider-mocker instances that accept exactly one Authorization
// value and refuse any request still carrying the gateway credential header,
// so a 200 proves what the upstream received.
const (
	trustedGatewayModel        = "gateway-model"
	trustedGatewayStaticModel  = "static-model"
	trustedGatewayAliceToken   = "Bearer e2e-client-token-alice" // #nosec G101 -- E2E fixture token
	trustedGatewayBobToken     = "Bearer e2e-client-token-bob"   // #nosec G101 -- E2E fixture token
	trustedGatewayRejectReason = "trusted gateway credential"    // #nosec G101 -- error text, not a credential
)

func init() {
	pkgtestcases.Register("trusted-gateway-credential", pkgtestcases.TestCase{
		Description: "Forward the deployment-injected gateway credential only to trusted_gateway backends; reject requests without one before dispatch",
		Tags:        []string{"trusted-gateway", "security", "authz"},
		Fn:          testTrustedGatewayCredential,
	})
}

type trustedGatewayCase struct {
	name          string
	model         string
	authorization string
	extraHeaders  map[string]string
	// wantRouterReject expects the Router's own 401, produced before any
	// upstream call, naming the missing credential.
	wantRouterReject bool
}

func testTrustedGatewayCredential(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing the trusted gateway credential contract")
	}
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []trustedGatewayCase{
		{
			// Trusted forwarding: the gateway accepts only alice's credential as
			// its Bearer token, so the client token was replaced and the
			// credential header did not travel upstream.
			name:          "authenticated request reaches the trusted gateway with its credential",
			model:         trustedGatewayModel,
			authorization: trustedGatewayAliceToken,
		},
		{
			// Untrusted rejection: no authentication result, so the Router
			// fails closed before dispatch.
			name:             "unauthenticated request to the trusted gateway is rejected by the Router",
			model:            trustedGatewayModel,
			wantRouterReject: true,
		},
		{
			// Sanitization: client-supplied copies of the authenticator's
			// outputs are stripped at the gateway and cannot pose as one.
			name:  "client-supplied credential and identity headers are stripped",
			model: trustedGatewayModel,
			extraHeaders: map[string]string{ // #nosec G101 -- E2E fixture values
				"x-authz-user-id":          "alice",
				"x-vsr-gateway-credential": "gw-virtual-key-alice",
			},
			wantRouterReject: true,
		},
		{
			// An authenticated principal without an issued credential is not
			// given a shared one.
			name:             "authenticated request without a gateway credential is rejected",
			model:            trustedGatewayModel,
			authorization:    trustedGatewayBobToken,
			wantRouterReject: true,
		},
		{
			// Default-disabled: an ordinary backend keeps its static key, and
			// the credential that alice carries never reaches it.
			name:          "static-key backend receives its own key, not the gateway credential",
			model:         trustedGatewayStaticModel,
			authorization: trustedGatewayAliceToken,
		},
	}

	total, passed := 0, 0
	for _, tc := range cases {
		total++
		if detail := runTrustedGatewayCase(ctx, tc, localPort); detail == "" {
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
		return fmt.Errorf("trusted-gateway-credential: %d/%d cases passed", passed, total)
	}
	return nil
}

// runTrustedGatewayCase returns an empty string when the case passes.
func runTrustedGatewayCase(ctx context.Context, tc trustedGatewayCase, localPort string) string {
	payload, err := json.Marshal(map[string]interface{}{
		"model":    tc.model,
		"messages": []map[string]string{{"role": "user", "content": "trusted gateway probe"}},
	})
	if err != nil {
		return fmt.Sprintf("marshal request: %v", err)
	}
	url := fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return fmt.Sprintf("build request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if tc.authorization != "" {
		req.Header.Set("Authorization", tc.authorization)
	}
	for name, value := range tc.extraHeaders {
		req.Header.Set(name, value)
	}
	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return fmt.Sprintf("send request: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()
	body, _ := io.ReadAll(resp.Body)

	if tc.wantRouterReject {
		// The Router's reply, not the mocker's: it names the contract and is
		// stamped as a router error response.
		if resp.StatusCode != http.StatusUnauthorized {
			return fmt.Sprintf("expected 401, got %d: %s", resp.StatusCode, truncateBody(body))
		}
		if !strings.Contains(string(body), trustedGatewayRejectReason) {
			return fmt.Sprintf("401 did not come from the Router's credential check: %s", truncateBody(body))
		}
		if path := resp.Header.Get("x-vsr-response-path"); path != "error" {
			return fmt.Sprintf("x-vsr-response-path = %q, want the Router's error path", path)
		}
		return ""
	}
	// The mocker answers 200 only to its expected Authorization and only
	// without the gateway credential header; any other status means the
	// wrong credential, or the header itself, reached the upstream.
	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("expected 200, got %d: %s", resp.StatusCode, truncateBody(body))
	}
	return ""
}

func truncateBody(body []byte) string {
	const limit = 200
	if len(body) > limit {
		return string(body[:limit]) + "…"
	}
	return string(body)
}
