package testcases

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

type shadowJudgeCandidate struct {
	Arm           string `json:"arm"`
	Text          string `json:"text"`
	NamesOwnModel bool   `json:"names_own_model"`
}

type shadowJudgeTaskSet struct {
	ManifestDigest string `json:"manifest_digest"`
	Counts         struct {
		Pairs    int            `json:"pairs"`
		Tasks    int            `json:"tasks"`
		Excluded map[string]int `json:"excluded"`
	} `json:"counts"`
	Tasks []struct {
		Pair   string               `json:"pair"`
		Input  string               `json:"input"`
		First  shadowJudgeCandidate `json:"first"`
		Second shadowJudgeCandidate `json:"second"`
	} `json:"tasks"`
}

func init() {
	pkgtestcases.Register("shadow-dataset-judge-tasks", pkgtestcases.TestCase{
		Description: "The judge task route turns a captured primary and shadow comparison into blinded pairwise tasks in both orders, and refuses a caller without replay.detail",
		Tags:        []string{"router-replay", "shadow-dispatch", "functional"},
		Fn:          testShadowDatasetJudgeTasks,
	})
}

func testShadowDatasetJudgeTasks(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()

	sessionID, err := run.primary(ctx, "vllm-sr/shadow-judge", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	if _, err = run.replay(ctx, sessionID, 1); err != nil {
		return err
	}

	target := "/api/v1/observability/replays/dataset/judge-tasks?session_id=" +
		url.QueryEscape(sessionID) + shadowDatasetSplitPlan + "&blinding_key=e2e-judge-key"
	raw, err := doRouterReplayManagementGETAs(ctx, run.api, target, routerReplayDetailToken)
	if err != nil {
		return fmt.Errorf("GET judge tasks: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return fmt.Errorf("GET judge tasks status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var set shadowJudgeTaskSet
	if err = raw.DecodeJSON(&set); err != nil {
		return fmt.Errorf("decode judge tasks: %w", err)
	}
	if err = requireBlindedPair(&set); err != nil {
		return err
	}

	// The tasks carry captured text, so a reader who may list replays but not
	// read their bodies is refused rather than handed redacted tasks.
	viewer, err := doRouterReplayManagementGETAs(ctx, run.api, target, routerReplayManagementToken)
	if err != nil {
		return fmt.Errorf("GET judge tasks as viewer: %w", err)
	}
	if viewer.StatusCode != http.StatusForbidden {
		return fmt.Errorf("viewer read judge tasks with status %d, want 403: %s", viewer.StatusCode, string(viewer.Body))
	}
	return nil
}

// requireBlindedPair checks the one comparison came back as a single pair in
// both orders, with the stored text on each side and no model name outside it.
func requireBlindedPair(set *shadowJudgeTaskSet) error {
	if set.ManifestDigest == "" || set.Counts.Pairs != 1 || set.Counts.Tasks != 2 || len(set.Tasks) != 2 {
		return fmt.Errorf("judge tasks %+v, want one pair in two tasks", set.Counts)
	}
	first, second := set.Tasks[0], set.Tasks[1]
	if first.Pair != second.Pair || first.First != second.Second || first.Second != second.First {
		return fmt.Errorf("judge tasks are not one pair with the sides swapped: %+v", set.Tasks)
	}

	shadowText := "Hello from openai/shadow-candidate."
	texts := map[string]bool{first.First.Text: true, first.Second.Text: true}
	if !texts[shadowPrimaryContent] || !texts[shadowText] {
		return fmt.Errorf("judge task texts %v, want the primary and the shadow answer", texts)
	}
	if !strings.Contains(first.Input, "Please say hello.") {
		return fmt.Errorf("judge task input %q does not carry the request", first.Input)
	}
	// Both mock answers introduce their own model, which is the leak no label
	// can hide, so both sides have to be flagged.
	if !first.First.NamesOwnModel || !first.Second.NamesOwnModel {
		return fmt.Errorf("judge task sides %+v and %+v, want both flagged as naming their model", first.First, first.Second)
	}
	for _, arm := range []string{first.First.Arm, first.Second.Arm} {
		if strings.Contains(arm, "openai") || strings.Contains(arm, "primary") || strings.Contains(arm, "shadow") {
			return fmt.Errorf("arm label %q is not opaque", arm)
		}
	}
	return nil
}
