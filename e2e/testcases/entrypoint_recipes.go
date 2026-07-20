package testcases

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("entrypoint-recipe-routing", pkgtestcases.TestCase{
		Description: "Verify entrypoint virtual model names select their routing recipe and are listed by /v1/models (issue #2331)",
		Tags:        []string{"kubernetes", "routing", "entrypoints"},
		Fn:          testEntrypointRecipeRouting,
	})
}

// The kubernetes profile wires two entrypoints (see
// e2e/profiles/ai-gateway/values.yaml): vllm-sr/e2e-recipe-probe selects the
// entrypoint-probe recipe, whose only decision is gated on the sentinel
// keyword below, and vllm-sr/e2e-default-alias selects the default recipe.
const (
	entrypointProbeModel    = "vllm-sr/e2e-recipe-probe"
	entrypointDefaultAlias  = "vllm-sr/e2e-default-alias"
	entrypointProbeKeyword  = "__ENTRYPOINT_RECIPE_PROBE__"
	entrypointProbeDecision = "entrypoint_probe_decision"
	entrypointProbeAdapter  = "general-expert"
)

// testEntrypointRecipeRouting asserts the request-path half of the entrypoint
// contract deterministically: the probe entrypoint must route through its
// recipe's decision, the default-recipe alias must not leak into another
// recipe's decisions even though the probe signal is registered globally, and
// both virtual names must be listed by /v1/models.
func testEntrypointRecipeRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing entrypoint recipe routing")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	probeDecision, err := assertEntrypointProbeSelectsRecipe(ctx, localPort, opts.Verbose)
	if err != nil {
		return err
	}
	aliasDecision, err := assertDefaultAliasStaysIsolated(ctx, localPort, opts.Verbose)
	if err != nil {
		return err
	}
	modelIDs, err := assertEntrypointsListedInModels(ctx, client, opts)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"probe_decision":  probeDecision,
			"alias_decision":  aliasDecision,
			"model_ids_count": len(modelIDs),
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] Entrypoint recipe routing verified ✓")
	}
	return nil
}

// assertEntrypointProbeSelectsRecipe sends the sentinel query through the
// probe entrypoint and requires the recipe's decision and adapter.
func assertEntrypointProbeSelectsRecipe(ctx context.Context, localPort string, verbose bool) (string, error) {
	query := "Entrypoint probe " + entrypointProbeKeyword + " please route this request."

	response, err := sendLocalChatCompletion(ctx, localPort, entrypointProbeModel, query, 30*time.Second)
	if err != nil {
		return "", fmt.Errorf("probe entrypoint request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(verbose, response, "entrypoint-recipe-routing", "Query: "+query)
		return "", fmt.Errorf("probe entrypoint request: %s", formatUnexpectedChatCompletionStatus(response))
	}

	decision := response.Headers.Get("x-vsr-selected-decision")
	if decision != entrypointProbeDecision {
		return "", fmt.Errorf("expected x-vsr-selected-decision=%s for model %s, got %q",
			entrypointProbeDecision, entrypointProbeModel, decision)
	}
	if model := response.Headers.Get("x-vsr-selected-model"); model != entrypointProbeAdapter {
		return "", fmt.Errorf("expected x-vsr-selected-model=%s for model %s, got %q",
			entrypointProbeAdapter, entrypointProbeModel, model)
	}
	if verbose {
		fmt.Printf("[Test]   %s -> decision %s, model %s ✓\n", entrypointProbeModel, decision, entrypointProbeAdapter)
	}
	return decision, nil
}

// assertDefaultAliasStaysIsolated sends a sentinel-bearing query through the
// default-recipe alias. The probe keyword signal lives in the global registry,
// but only the selected recipe's decisions may consume it, so the probe
// decision must never be selected here.
func assertDefaultAliasStaysIsolated(ctx context.Context, localPort string, verbose bool) (string, error) {
	query := "Compliance check with token " + entrypointProbeKeyword + " on the shared profile."

	response, err := sendLocalChatCompletion(ctx, localPort, entrypointDefaultAlias, query, 30*time.Second)
	if err != nil {
		return "", fmt.Errorf("default alias request failed: %w", err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(verbose, response, "entrypoint-recipe-routing", "Query: "+query)
		return "", fmt.Errorf("default alias request: %s", formatUnexpectedChatCompletionStatus(response))
	}

	decision := response.Headers.Get("x-vsr-selected-decision")
	if decision == entrypointProbeDecision {
		return "", fmt.Errorf("default-recipe alias %s must not select another recipe's decision %s",
			entrypointDefaultAlias, entrypointProbeDecision)
	}
	if verbose {
		fmt.Printf("[Test]   %s -> decision %q (isolated from probe recipe) ✓\n", entrypointDefaultAlias, decision)
	}
	return decision, nil
}

// assertEntrypointsListedInModels requires both entrypoint virtual names in
// the router API server's /v1/models listing.
func assertEntrypointsListedInModels(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) ([]string, error) {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	modelIDs, err := fetchModelIDs(ctx, session.HTTPClient(30*time.Second), session.URL("/v1/models"))
	if err != nil {
		return nil, err
	}
	for _, name := range []string{entrypointProbeModel, entrypointDefaultAlias} {
		if !containsString(modelIDs, name) {
			return nil, fmt.Errorf("expected /v1/models to include entrypoint %s, got %v", name, modelIDs)
		}
	}
	if opts.Verbose {
		fmt.Printf("[Test]   /v1/models lists %s and %s ✓\n", entrypointProbeModel, entrypointDefaultAlias)
	}
	return modelIDs, nil
}
