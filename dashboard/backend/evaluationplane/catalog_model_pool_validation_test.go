package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func catalogTestArm(id string, modalities []string) ModelArm {
	return ModelArm{
		ID: id, Model: "org/" + id,
		ProviderModelIDDigest: "sha256:" + strings.Repeat("a", 64),
		Modalities:            modalities,
	}
}

func TestLiveModelPoolCandidateSeparatesPolicyFromArmComposition(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	configPath := filepath.Join(root, "config.yaml")
	poolBaselineYAML := strings.Replace(
		modelArmTestYAML,
		"    - name: metadata-only\n      provider_model_id: metadata-only-upstream",
		"    - name: pool/third\n      provider_model_id: provider-third\n      backend_refs:\n        - name: pool-third-primary\n          endpoint: pool-third.example.test:8004\n      pricing:\n        currency: USD\n        prompt_per_1m: 0.4\n        completion_per_1m: 0.8\n    - name: metadata-only\n      provider_model_id: metadata-only-upstream",
		1,
	)
	poolBaselineYAML = strings.Replace(
		poolBaselineYAML,
		"    - name: metadata-only\n      modality: text",
		"    - name: pool/third\n      modality: text\n    - name: metadata-only\n      modality: text",
		1,
	)
	poolBaselineYAML = strings.Replace(
		poolBaselineYAML,
		"        - model: local/omni",
		"        - model: local/omni\n        - model: pool/third",
		1,
	)
	if err := os.WriteFile(configPath, []byte(poolBaselineYAML), 0o600); err != nil {
		t.Fatalf("write baseline config: %v", err)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "live pool baseline",
		SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "model_pool",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	baseline, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("create baseline: %v", err)
	}
	baselineManifest, _, err := service.readDurableManifest(baseline.ID)
	if err != nil {
		t.Fatalf("read baseline manifest: %v", err)
	}
	baseline = completeTestRun(t, service, baseline)

	changed := strings.Replace(poolBaselineYAML, "\n        - model: local/omni", "", 1)
	if changed == poolBaselineYAML {
		t.Fatal("pool membership fixture did not change")
	}
	if writeCandidateErr := os.WriteFile(configPath, []byte(changed), 0o600); writeCandidateErr != nil {
		t.Fatalf("write candidate config: %v", writeCandidateErr)
	}
	request.ClientRequestID = newTestClientRequestID()
	request.Name = "live pool candidate"
	request.BaselineRunID = baseline.ID
	candidate, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("allowed model-pool treatment rejected: %v", err)
	}
	candidateManifest, _, err := service.readDurableManifest(candidate.ID)
	if err != nil {
		t.Fatalf("read candidate manifest: %v", err)
	}
	if baselineManifest.PolicySnapshotDigest != candidateManifest.PolicySnapshotDigest ||
		baselineManifest.Target.Mixture == nil || candidateManifest.Target.Mixture == nil ||
		baselineManifest.Target.Mixture.PoolDigest == candidateManifest.Target.Mixture.PoolDigest ||
		baselineManifest.Target.Mixture.BindingDigest == candidateManifest.Target.Mixture.BindingDigest ||
		baselineManifest.Target.BackendTopologyDigest == candidateManifest.Target.BackendTopologyDigest {
		t.Fatalf("model-pool factors were not isolated: baseline=%#v candidate=%#v", baselineManifest.Target, candidateManifest.Target)
	}

	registry, err := service.registrySnapshot()
	if err != nil {
		t.Fatalf("read current registry: %v", err)
	}
	target, ok := registry.target(request.TargetID)
	if !ok {
		t.Fatal("candidate target disappeared")
	}
	target.EnvoyURL = "http://different-runtime.invalid"
	if err := service.validateComparableTargetSnapshot("model_pool", target, baseline.ID); !errors.Is(err, ErrInvalid) || !strings.Contains(err.Error(), "runtime origins") {
		t.Fatalf("model_pool runtime-origin drift error=%v, want fail-closed ErrInvalid", err)
	}
}

func TestPendingLiveRunFailsClosedWhenItsMixtureDriftsBeforeStart(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write baseline config: %v", err)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "frozen live mixture",
		SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "recipe",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	changed := strings.Replace(modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1)
	if writeChangedConfigErr := os.WriteFile(configPath, []byte(changed), 0o600); writeChangedConfigErr != nil {
		t.Fatalf("write changed config: %v", writeChangedConfigErr)
	}
	if _, startErr := service.StartRunAs(context.Background(), SystemActor(), run.ID); !errors.Is(startErr, ErrConflict) {
		t.Fatalf("StartRun after recipe drift error=%v, want ErrConflict", startErr)
	}
	stored, err := service.GetRunAs(SystemActor(), run.ID)
	if err != nil || stored.Status != StatusPending {
		t.Fatalf("drifted run did not remain pending: run=%+v err=%v", stored, err)
	}
}
