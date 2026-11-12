package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The delegated credential contract (issue #2286), observed at the gateway
// edge of the delegated-credential profile. The profile's Lua filter stands
// in for the deployment authenticator: it strips client-supplied
// authentication outputs and, for alice's client token, attaches
// x-authz-user-id and the delegated credential dc-virtual-key-alice. Both
// backends, and the delegated decision-model deployment, are provider-mocker
// instances that accept exactly one Authorization value and refuse any
// request still carrying the delegated credential header, so a 200 proves
// what the upstream received.
const (
	delegatedGatewayModel        = "gateway-model"
	delegatedStaticModel         = "static-model"
	delegatedRoutedModel         = "vllm-sr/auto"
	delegatedAliceToken          = "Bearer e2e-client-token-alice" // #nosec G101 -- E2E fixture token
	delegatedBobToken            = "Bearer e2e-client-token-bob"   // #nosec G101 -- E2E fixture token
	delegatedRejectReason        = "delegated credential"          // #nosec G101 -- error text, not a credential
	delegatedDecisionMarker      = "__SYSTEMONE_YES__"
	delegatedDecisionRoute       = "decision_model_route"
	delegatedDecisionDeployment  = "delegated-decisions"
	delegatedRouterRefusalEvent  = "delegated_router_call_refused"
	delegatedRouterRefusalWindow = 60 * time.Second
)

func init() {
	pkgtestcases.Register("delegated-credential", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "Present the delegated credential only to delegated upstreams, a provider backend and a decision-model deployment, by calls made for a request; reject requests without one before the call; refuse Router-own calls",
		Tags:        []string{"delegated-credential", "security", "authz"},
		Fn:          testDelegatedCredential,
	})
	pkgtestcases.Register("delegated-credential-disabled", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "With the delegated credential contract off, a delegated credential header never reaches an upstream",
		Tags:        []string{"delegated-credential", "security", "authz"},
		Fn:          testDelegatedCredentialDisabled,
	})
}

// testDelegatedCredentialDisabled runs in a profile that leaves
// global.services.authz.delegated_credential off. The request carries what an
// authenticator would attach, a principal and a delegated credential; the
// Router removes the reserved header, and the profile's provider-mocker
// refuses any request that still carries it, so a 200 proves no upstream
// received it.
func testDelegatedCredentialDisabled(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	return runChatCompletionsRequest(ctx, client, opts, map[string]string{ // #nosec G101 -- E2E fixture values
		"x-authz-user-id":            "alice",
		"x-vsr-delegated-credential": "dc-virtual-key-alice",
	})
}

type delegatedCase struct {
	name          string
	model         string
	content       string
	authorization string
	extraHeaders  map[string]string
	// wantRouterReject expects the Router's own 401, produced before any
	// upstream call, naming the missing credential.
	wantRouterReject bool
	// wantDecision is the decision a 200 must report.
	wantDecision string
}

func testDelegatedCredential(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing the delegated credential contract")
	}
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []delegatedCase{
		{
			// Delegated forwarding: the gateway accepts only alice's credential
			// as its Bearer token, so the client token was replaced and the
			// credential header did not travel upstream.
			name:          "authenticated request reaches the delegated backend with its credential",
			model:         delegatedGatewayModel,
			authorization: delegatedAliceToken,
		},
		{
			// Untrusted rejection: no authentication result, so the Router
			// fails closed before dispatch.
			name:             "unauthenticated request to the delegated backend is rejected by the Router",
			model:            delegatedGatewayModel,
			wantRouterReject: true,
		},
		{
			// Sanitization: client-supplied copies of the authenticator's
			// outputs are stripped at the gateway and cannot pose as one.
			name:  "client-supplied credential and identity headers are stripped",
			model: delegatedGatewayModel,
			extraHeaders: map[string]string{ // #nosec G101 -- E2E fixture values
				"x-authz-user-id":            "alice",
				"x-vsr-delegated-credential": "dc-virtual-key-alice",
			},
			wantRouterReject: true,
		},
		{
			// An authenticated principal without an issued credential is not
			// given a shared one.
			name:             "authenticated request without a delegated credential is rejected",
			model:            delegatedGatewayModel,
			authorization:    delegatedBobToken,
			wantRouterReject: true,
		},
		{
			// An ordinary backend keeps its static key, and the credential
			// that alice carries never reaches it.
			name:          "static-key backend receives its own key, not the delegated credential",
			model:         delegatedStaticModel,
			authorization: delegatedAliceToken,
		},
		{
			// The decision model answers only alice's credential: the routed
			// decision proves the System One call carried it. Without it the
			// rule would be unknown and the request would go to static-model.
			name:          "the decision-model call made for the request carries its credential",
			model:         delegatedRoutedModel,
			content:       "delegated decision probe " + delegatedDecisionMarker,
			authorization: delegatedAliceToken,
			wantDecision:  delegatedDecisionRoute,
		},
		{
			// bob's request would otherwise reach static-model, which needs no
			// delegated credential: the 401 comes from the decision-model call
			// the Router refused before making it.
			name:             "a decision-model call without the credential is answered 401 before the call",
			model:            delegatedRoutedModel,
			content:          "delegated decision probe " + delegatedDecisionMarker,
			authorization:    delegatedBobToken,
			wantRouterReject: true,
		},
	}

	total, passed := 0, 0
	for _, tc := range cases {
		total++
		if detail := runDelegatedCase(ctx, tc, localPort); detail == "" {
			passed++
			if opts.Verbose {
				fmt.Printf("[Test] ✓ %s\n", tc.name)
			}
		} else {
			fmt.Printf("[Test] ✗ %s: %s\n", tc.name, detail)
		}
	}
	// A Router-own call never carries a user's credential: the Router's own
	// read of the delegated deployment's model listing is refused, not sent.
	total++
	if detail := delegatedRouterOwnCallRefused(ctx, client, opts); detail == "" {
		passed++
		if opts.Verbose {
			fmt.Println("[Test] ✓ a Router-own call to the delegated deployment is refused")
		}
	} else {
		fmt.Printf("[Test] ✗ a Router-own call to the delegated deployment is refused: %s\n", detail)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":  total,
			"passed_tests": passed,
			"failed_tests": total - passed,
		})
	}
	if passed != total {
		return fmt.Errorf("delegated-credential: %d/%d cases passed", passed, total)
	}
	return nil
}

// runDelegatedCase returns an empty string when the case passes.
func runDelegatedCase(ctx context.Context, tc delegatedCase, localPort string) string {
	content := tc.content
	if content == "" {
		content = "delegated credential probe"
	}
	payload, err := json.Marshal(map[string]interface{}{
		"model":    tc.model,
		"messages": []map[string]string{{"role": "user", "content": content}},
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
		if !strings.Contains(string(body), delegatedRejectReason) {
			return fmt.Sprintf("401 did not come from the Router's credential check: %s", truncateBody(body))
		}
		if path := resp.Header.Get("x-vsr-response-path"); path != "error" {
			return fmt.Sprintf("x-vsr-response-path = %q, want the Router's error path", path)
		}
		return ""
	}
	// The mocker answers 200 only to its expected Authorization and only
	// without the delegated credential header; any other status means the
	// wrong credential, or the header itself, reached the upstream.
	if resp.StatusCode != http.StatusOK {
		return fmt.Sprintf("expected 200, got %d: %s", resp.StatusCode, truncateBody(body))
	}
	if tc.wantDecision != "" {
		if decision := resp.Header.Get("x-vsr-selected-decision"); decision != tc.wantDecision {
			return fmt.Sprintf("x-vsr-selected-decision = %q, want %q", decision, tc.wantDecision)
		}
	}
	return ""
}

// delegatedRouterOwnCallRefused reads the Router's log for its refusal of
// the model listing it reads from the delegated decision-model deployment on
// its own behalf, which carries no user's credential.
func delegatedRouterOwnCallRefused(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) string {
	deadline := time.Now().Add(delegatedRouterRefusalWindow)
	for {
		found, err := delegatedRefusalLogged(ctx, client, opts)
		if err != nil {
			return err.Error()
		}
		if found {
			return ""
		}
		if time.Now().After(deadline) {
			return fmt.Sprintf("no %s event for %s in the Router's log", delegatedRouterRefusalEvent, delegatedDecisionDeployment)
		}
		select {
		case <-ctx.Done():
			return ctx.Err().Error()
		case <-time.After(2 * time.Second):
		}
	}
}

func delegatedRefusalLogged(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (bool, error) {
	pod, err := modelruntime.RouterPod(ctx, client, opts.RestConfig)
	if err != nil {
		return false, err
	}
	logs, err := client.CoreV1().Pods(pod.Namespace).GetLogs(pod.Pod, &corev1.PodLogOptions{Container: pod.Container}).DoRaw(ctx)
	if err != nil {
		return false, fmt.Errorf("read the Router's log: %w", err)
	}
	scanner := bufio.NewScanner(bytes.NewReader(logs))
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		var event struct {
			Msg        string `json:"msg"`
			Call       string `json:"call"`
			Deployment string `json:"deployment"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil {
			continue
		}
		if event.Msg == delegatedRouterRefusalEvent && event.Deployment == delegatedDecisionDeployment && event.Call == "systemone_listing" {
			return true, nil
		}
	}
	return false, scanner.Err()
}

func truncateBody(body []byte) string {
	const limit = 200
	if len(body) > limit {
		return string(body[:limit]) + "…"
	}
	return string(body)
}
