package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("decision-scoped-multi-factor", pkgtestcases.TestCase{
		Description: "Verify each matched decision uses its own multi-factor policy",
		Tags:        []string{"routing", "selection", "multi-factor", "regression"},
		Fn:          testDecisionScopedMultiFactor,
	})
}

func testDecisionScopedMultiFactor(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []struct {
		query        string
		wantDecision string
		wantModel    string
	}{
		{
			query:        "Apply the decision scoped quality policy",
			wantDecision: "quality_policy_decision",
			wantModel:    "premium-model",
		},
		{
			query:        "Apply the decision scoped cost policy",
			wantDecision: "cost_policy_decision",
			wantModel:    "economy-model",
		},
	}

	for _, tc := range cases {
		decision, model, err := requestDecisionScopedSelection(ctx, localPort, tc.query)
		if err != nil {
			return err
		}
		if decision != tc.wantDecision || model != tc.wantModel {
			return fmt.Errorf(
				"query %q selected decision=%q model=%q, want decision=%q model=%q",
				tc.query,
				decision,
				model,
				tc.wantDecision,
				tc.wantModel,
			)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"verified_policies": len(cases)})
	}
	return nil
}

func requestDecisionScopedSelection(ctx context.Context, localPort, query string) (string, string, error) {
	payload, err := json.Marshal(map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": query},
		},
	})
	if err != nil {
		return "", "", fmt.Errorf("marshal chat request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort),
		bytes.NewReader(payload),
	)
	if err != nil {
		return "", "", fmt.Errorf("create chat request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := (&http.Client{Timeout: 30 * time.Second}).Do(req)
	if err != nil {
		return "", "", fmt.Errorf("send chat request: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", fmt.Errorf("read chat response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("chat request returned HTTP %d: %s", resp.StatusCode, string(body))
	}

	return resp.Header.Get("x-vsr-selected-decision"), resp.Header.Get("x-vsr-selected-model"), nil
}
