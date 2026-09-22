package testcases

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const shadowDatasetSplitPlan = "&seed=e2e-shadow-dataset&split=train:8&split=eval:2"

type shadowDatasetArm struct {
	Model        string `json:"model"`
	Backend      string `json:"backend"`
	OutputDigest string `json:"output_digest"`
}

type shadowDatasetExample struct {
	ID          string             `json:"id"`
	InputDigest string             `json:"input_digest"`
	Split       string             `json:"split"`
	Primary     shadowDatasetArm   `json:"primary"`
	Shadows     []shadowDatasetArm `json:"shadows"`
	Lineage     struct {
		ReplayID string `json:"replay_id"`
	} `json:"lineage"`
}

type shadowDatasetManifest struct {
	Version string `json:"version"`
	Digest  string `json:"digest"`
	Counts  struct {
		Records  int            `json:"records"`
		Examples int            `json:"examples"`
		Excluded map[string]int `json:"excluded"`
	} `json:"counts"`
	Examples []shadowDatasetExample `json:"examples"`
}

func init() {
	pkgtestcases.Register("shadow-dataset-export-manifest", pkgtestcases.TestCase{
		Description: "The dataset export turns a captured primary and shadow comparison into a reproducible manifest that carries digests and no prompt or response text",
		Tags:        []string{"router-replay", "shadow-dispatch", "functional"},
		Fn:          testShadowDatasetExportManifest,
	})
}

func testShadowDatasetExportManifest(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	run, err := openShadowDispatchRun(ctx, client, opts)
	if err != nil {
		return err
	}
	defer run.Close()

	sessionID, err := run.primary(ctx, "vllm-sr/shadow-ok", shadowPrimaryCeiling)
	if err != nil {
		return err
	}
	record, err := run.replay(ctx, sessionID, 1)
	if err != nil {
		return err
	}

	target := "/api/v1/observability/replays/dataset?session_id=" +
		url.QueryEscape(sessionID) + shadowDatasetSplitPlan
	manifest, body, err := fetchShadowDatasetManifest(ctx, run.api, target, routerReplayDetailToken)
	if err != nil {
		return err
	}
	if err = requireComparedExample(manifest, record); err != nil {
		return err
	}

	// The manifest is published beside the numbers it supports, so it must
	// carry identity and digests alone. A reader without replay.detail
	// therefore has nothing to redact and reads the same manifest.
	for _, secret := range []string{shadowPrimaryContent, "Please say hello."} {
		if strings.Contains(string(body), secret) {
			return fmt.Errorf("dataset manifest carries captured text %q", secret)
		}
	}
	viewerManifest, _, err := fetchShadowDatasetManifest(ctx, run.api, target, routerReplayManagementToken)
	if err != nil {
		return err
	}
	if viewerManifest.Digest != manifest.Digest {
		return fmt.Errorf("viewer read manifest digest %s, operator read %s",
			viewerManifest.Digest, manifest.Digest)
	}

	// The digest is the dataset's identity. The same records under the same
	// policy have to rebuild it, otherwise nothing downstream can cite it.
	repeat, _, err := fetchShadowDatasetManifest(ctx, run.api, target, routerReplayDetailToken)
	if err != nil {
		return err
	}
	if repeat.Digest != manifest.Digest {
		return fmt.Errorf("second export digest %s, first export %s", repeat.Digest, manifest.Digest)
	}
	return nil
}

func requireComparedExample(manifest *shadowDatasetManifest, record *shadowReplayRecord) error {
	if manifest.Version == "" || manifest.Digest == "" {
		return fmt.Errorf("dataset manifest has no version or digest: %+v", manifest)
	}
	if len(manifest.Counts.Excluded) != 0 {
		return fmt.Errorf("dataset manifest excluded the compared observation: %v", manifest.Counts.Excluded)
	}
	if manifest.Counts.Examples != 1 || len(manifest.Examples) != 1 {
		return fmt.Errorf("dataset manifest has %d examples, want one", manifest.Counts.Examples)
	}

	example := manifest.Examples[0]
	if example.Lineage.ReplayID != record.ID {
		return fmt.Errorf("example lineage replay %q, want %q", example.Lineage.ReplayID, record.ID)
	}
	if example.Primary.Model != shadowPrimaryModel || example.Primary.OutputDigest == "" {
		return fmt.Errorf("example primary arm %+v, want %s with a digest", example.Primary, shadowPrimaryModel)
	}
	if len(example.Shadows) != 1 {
		return fmt.Errorf("example has %d shadow arms, want one", len(example.Shadows))
	}
	shadow := example.Shadows[0]
	if shadow.Model != "openai/shadow-candidate" || shadow.OutputDigest == "" {
		return fmt.Errorf("example shadow arm %+v, want openai/shadow-candidate with a digest", shadow)
	}
	if example.InputDigest == "" || example.Split == "" {
		return fmt.Errorf("example %s has no input digest or split", example.ID)
	}
	return nil
}

func fetchShadowDatasetManifest(
	ctx context.Context,
	apiSession *fixtures.ServiceSession,
	requestTarget string,
	token string,
) (*shadowDatasetManifest, []byte, error) {
	raw, err := doRouterReplayManagementGETAs(ctx, apiSession, requestTarget, token)
	if err != nil {
		return nil, nil, fmt.Errorf("GET dataset manifest: %w", err)
	}
	if raw.StatusCode != http.StatusOK {
		return nil, nil, fmt.Errorf("GET dataset manifest status %d: %s", raw.StatusCode, string(raw.Body))
	}
	var manifest shadowDatasetManifest
	if err = raw.DecodeJSON(&manifest); err != nil {
		return nil, nil, fmt.Errorf("decode dataset manifest: %w", err)
	}
	return &manifest, raw.Body, nil
}
