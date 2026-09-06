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
	pkgtestcases.Register("metadata-routing", pkgtestcases.TestCase{
		Description: "Route from bounded caller metadata",
		Tags:        []string{"signal-decision", "metadata", "routing"},
		Fn:          testMetadataRouting,
	})
}

func testMetadataRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	payload := map[string]interface{}{
		"model": "vllm-sr/metadata-policy",
		"messages": []map[string]string{
			{"role": "user", "content": "route this request"},
		},
		"metadata": map[string]string{"cohort": "canary"},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort),
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(response.Body)
		return fmt.Errorf("metadata routing returned HTTP %d: %s", response.StatusCode, responseBody)
	}
	if decision := response.Header.Get("x-vsr-selected-decision"); decision != "metadata_recipe_decision" {
		return fmt.Errorf("selected decision = %q, want metadata_recipe_decision", decision)
	}
	if recipe := response.Header.Get("x-vsr-selected-recipe"); recipe != "metadata-policy" {
		return fmt.Errorf("selected recipe = %q, want metadata-policy", recipe)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"selected_decision": "metadata_recipe_decision",
			"selected_recipe":   "metadata-policy",
			"metadata_key":      "cohort",
		})
	}
	return nil
}
