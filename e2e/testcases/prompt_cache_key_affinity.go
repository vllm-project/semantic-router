package testcases

import (
	"context"
	"fmt"
	"net/http"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// The response-api profile routes these keywords to decisions over two
// identical candidates; see e2e/profiles/response-api/values.yaml.
const (
	cacheAffinitySeedKeyword  = "cache_affinity_seed_probe"
	cacheAffinityProbeKeyword = "cache_affinity_followup_probe"
	cacheAffinityTieModel     = "mock/cache-affinity-a"
	cacheAffinitySeedModel    = "mock/cache-affinity-b"
)

func init() {
	pkgtestcases.Register("prompt-cache-key-affinity", pkgtestcases.TestCase{
		Description: "A repeated prompt_cache_key keeps a hybrid tie on the model that served the key; a new or absent key does not",
		Tags:        []string{"routing", "selection", "response-api", "agents"},
		Fn:          testPromptCacheKeyAffinity,
	})
}

// Every request text is unique, so no two requests share a derived session and
// the key is the only link between them.
func testPromptCacheKeyAffinity(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	for _, api := range protocolCodecE2EClients {
		if api.path == "/v1/messages" {
			continue
		}
		run := uuid.NewString()
		key := "cache-affinity-" + run
		steps := []struct {
			name      string
			keyword   string
			key       string
			wantModel string
		}{
			{name: "seed", keyword: cacheAffinitySeedKeyword, key: key, wantModel: cacheAffinitySeedModel},
			{name: "same key", keyword: cacheAffinityProbeKeyword, key: key, wantModel: cacheAffinitySeedModel},
			{name: "new key", keyword: cacheAffinityProbeKeyword, key: key + "-new", wantModel: cacheAffinityTieModel},
			{name: "no key", keyword: cacheAffinityProbeKeyword, wantModel: cacheAffinityTieModel},
		}
		for _, step := range steps {
			body := api.request("MoM", fmt.Sprintf("%s %s %s", step.keyword, step.name, run), false)
			if step.key != "" {
				body["prompt_cache_key"] = step.key
			}
			result, err := sendProtocolMatrixRaw(ctx, session, api.path, body, false, nil)
			if err != nil {
				return fmt.Errorf("%s %s: %w", api.name, step.name, err)
			}
			if result.StatusCode != http.StatusOK {
				return fmt.Errorf("%s %s: HTTP %d: %s", api.name, step.name, result.StatusCode, truncateString(string(result.Body), 500))
			}
			model := result.Headers.Get("x-vsr-selected-model")
			if opts.Verbose {
				fmt.Printf("[Test] %s %s: %s via %s\n", api.name, step.name, model, result.Headers.Get(vsrSelectedDecisionHeader))
			}
			if model != step.wantModel {
				return fmt.Errorf("%s %s: routed to %q via %q, want %q",
					api.name, step.name, model, result.Headers.Get(vsrSelectedDecisionHeader), step.wantModel)
			}
		}
	}
	return nil
}
