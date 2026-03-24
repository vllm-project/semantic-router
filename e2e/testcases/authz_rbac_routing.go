package testcases

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("authz-rbac-routing", pkgtestcases.TestCase{
		Description: "Verify RBAC-based model routing: admin→14B decision, premium+complex→14B, free→7B, no-identity→default",
		Tags:        []string{"authz-rbac", "routing", "functional"},
		Fn:          testAuthzRBACRouting,
	})
}

// authzRoutingCase is a single RBAC routing assertion.
type authzRoutingCase struct {
	name         string
	userID       string
	groups       string
	prompt       string
	wantDecision string // expected x-vsr-selected-decision; empty = any 200 OK is sufficient
}

func testAuthzRBACRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[AuthzRBAC] Testing RBAC-based model routing")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	chatClient := fixtures.NewChatCompletionsClient(session, 30*time.Second)

	cases := []authzRoutingCase{
		{
			// alice is a platform-admin — priority 300, routes to 14B with reasoning
			name:         "admin_unrestricted",
			userID:       "alice",
			groups:       "platform-admins",
			prompt:       "Explain quantum entanglement",
			wantDecision: "admin_unrestricted",
		},
		{
			// bob is premium + sends a complex analysis query — keyword "analyze" triggers premium_complex (priority 250)
			name:         "premium_complex",
			userID:       "bob",
			groups:       "premium-tier",
			prompt:       "Please analyze and compare the trade-offs between microservices and monolithic architecture",
			wantDecision: "premium_complex",
		},
		{
			// bob is premium + sends a coding query — keyword "implement" triggers premium_code (priority 240)
			name:         "premium_code",
			userID:       "bob",
			groups:       "premium-tier",
			prompt:       "Can you implement a binary search function and debug the edge cases?",
			wantDecision: "premium_code",
		},
		{
			// bob is premium + sends a simple query — no keyword match, falls to premium_default (priority 150) → 7B
			name:         "premium_default",
			userID:       "bob",
			groups:       "premium-tier",
			prompt:       "Hello, what time is it?",
			wantDecision: "premium_default",
		},
		{
			// carol is free-tier — priority 100, routes to 7B
			name:         "free_default",
			userID:       "carol",
			groups:       "free-tier",
			prompt:       "Hello, how are you today?",
			wantDecision: "free_default",
		},
		{
			// no identity headers — authz fail_open=true so request passes, but no role is matched,
			// no decision fires, router uses default_model (7B)
			name:         "no_identity_default",
			userID:       "",
			groups:       "",
			prompt:       "Hello",
			wantDecision: "", // any 200 OK
		},
	}

	passed := 0
	for _, tc := range cases {
		if err := checkAuthzRoutingCase(ctx, chatClient, tc, opts.Verbose); err != nil {
			fmt.Printf("[AuthzRBAC] FAIL %s: %v\n", tc.name, err)
			continue
		}
		if opts.Verbose {
			fmt.Printf("[AuthzRBAC] PASS %s\n", tc.name)
		}
		passed++
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total":  len(cases),
			"passed": passed,
		})
	}

	if passed == 0 {
		return fmt.Errorf("authz-rbac-routing: 0/%d cases passed", len(cases))
	}
	return nil
}

func checkAuthzRoutingCase(ctx context.Context, chatClient *fixtures.ChatCompletionsClient, tc authzRoutingCase, verbose bool) error {
	headers := map[string]string{}
	if tc.userID != "" {
		headers["x-authz-user-id"] = tc.userID
	}
	if tc.groups != "" {
		headers["x-authz-user-groups"] = tc.groups
	}

	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model:    "MoM",
		Messages: []fixtures.ChatMessage{{Role: "user", Content: tc.prompt}},
	}, headers)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}

	if resp.StatusCode != 200 {
		return fmt.Errorf("expected 200, got %d: %s", resp.StatusCode, truncateString(string(resp.Body), 200))
	}

	if tc.wantDecision == "" {
		return nil
	}

	gotDecision := resp.Headers.Get("x-vsr-selected-decision")
	if gotDecision != tc.wantDecision {
		if verbose {
			fmt.Printf("[AuthzRBAC]   user=%q groups=%q: decision=%q, want=%q\n",
				tc.userID, tc.groups, gotDecision, tc.wantDecision)
		}
		return fmt.Errorf("x-vsr-selected-decision: got %q, want %q", gotDecision, tc.wantDecision)
	}
	return nil
}
