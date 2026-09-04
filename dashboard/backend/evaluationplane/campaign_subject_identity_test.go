package evaluationplane

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestCandidateSubjectDigestExcludesExecutionContext(t *testing.T) {
	manifest, report := candidateSubjectTestEvidence(t, ModeLive, "00000000-0000-4000-8000-000000000101")
	want := requireCandidateSubjectDigest(t, manifest, report)

	manifest.RunID = "00000000-0000-4000-8000-000000000102"
	manifest.Name = "different gate evidence"
	manifest.Description = "different execution context"
	manifest.SuiteIDs = []string{"live-capacity"}
	manifest.SuiteRevisions = map[string]string{"live-capacity": "executor-v2"}
	manifest.SuiteExecutors = map[string]string{"live-capacity": liveRuntimeExecutorID}
	manifest.TrackIDs = []TrackID{"capacity"}
	manifest.SampleLimit = 97
	manifest.Concurrency = 8
	manifest.Seed = 9001
	manifest.CreatedAt = manifest.CreatedAt.Add(time.Hour)
	report.Run.ID = manifest.RunID
	report.Run.ClientRequestID = manifest.RunID
	report.Run.Name = manifest.Name
	report.Run.Description = manifest.Description
	report.Run.SuiteIDs = append([]string(nil), manifest.SuiteIDs...)
	report.Run.TrackIDs = append([]TrackID(nil), manifest.TrackIDs...)
	report.Run.SampleLimit = manifest.SampleLimit
	report.Run.Concurrency = manifest.Concurrency
	report.Run.Seed = manifest.Seed
	report.Run.CreatedAt = manifest.CreatedAt
	report.Provenance.Seed = manifest.Seed
	report.Provenance.BenchmarkRevisions = map[string]string{"live-capacity": "executor-v2"}
	report.Provenance.WorkloadSnapshotDigest = digestString("different-workload")
	report.Provenance.EnvironmentSnapshotDigest = digestString("different-live-execution-context")
	sealCandidateSubjectTestManifest(t, &manifest)

	got := requireCandidateSubjectDigest(t, manifest, report)
	if got != want {
		t.Fatalf("execution context changed candidate subject digest: got %q, want %q", got, want)
	}
}

func TestCandidateSubjectDigestMatchesAcrossLiveAndReplay(t *testing.T) {
	liveManifest, liveReport := candidateSubjectTestEvidence(t, ModeLive, "00000000-0000-4000-8000-000000000111")
	replayManifest, replayReport := candidateSubjectTestEvidence(t, ModeReplay, "00000000-0000-4000-8000-000000000112")
	replayManifest.SuiteIDs = []string{"live-mom-core"}
	replayManifest.SuiteRevisions = map[string]string{"live-mom-core": "native-reference-v1"}
	replayManifest.SuiteExecutors = map[string]string{"live-mom-core": momReplayExecutorID}
	replayReport.Run.SuiteIDs = append([]string(nil), replayManifest.SuiteIDs...)
	replayReport.Provenance.BenchmarkRevisions = map[string]string{"live-mom-core": "native-reference-v1"}
	replayReport.Provenance.EnvironmentSnapshotDigest = digestString("mixture-replay-execution-context")
	sealCandidateSubjectTestManifest(t, &replayManifest)

	liveDigest := requireCandidateSubjectDigest(t, liveManifest, liveReport)
	replayDigest := requireCandidateSubjectDigest(t, replayManifest, replayReport)
	if liveDigest != replayDigest {
		t.Fatalf("same candidate differs across live/replay: live %q, replay %q", liveDigest, replayDigest)
	}
}

func TestCandidateSubjectDigestChangesWithEverySubjectFactor(t *testing.T) {
	baseManifest, baseReport := candidateSubjectTestEvidence(t, ModeLive, "00000000-0000-4000-8000-000000000121")
	want := requireCandidateSubjectDigest(t, baseManifest, baseReport)

	tests := []struct {
		name   string
		mutate func(*RunManifest, *Report)
	}{
		{name: "change profile", mutate: func(manifest *RunManifest, report *Report) {
			manifest.ChangeProfile, report.Run.ChangeProfile = "selector", "selector"
		}},
		{name: "target and mixture identity", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Target.Mixture.RecipeName = "candidate-recipe-v2"
			manifest.Target.Mixture.ID = "mom-" + strings.TrimPrefix(digestString("candidate-recipe-v2"), "sha256:")
			manifest.Target.ID = manifest.Target.Mixture.ID
			report.Run.TargetID, report.Provenance.TargetID = manifest.Target.ID, manifest.Target.ID
		}},
		{name: "code revision", mutate: func(manifest *RunManifest, report *Report) {
			manifest.CodeRevision = strings.Repeat("b", 40)
			report.Provenance.CodeRevision = manifest.CodeRevision
		}},
		{name: "configuration", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.ConfigDigest = digestString("config-v2")
		}},
		{name: "recipe and policy", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Target.Mixture.RecipeDigest = digestString("recipe-v2")
			manifest.PolicySnapshotDigest = manifest.Target.Mixture.RecipeDigest
			report.Provenance.PolicySnapshotDigest = manifest.PolicySnapshotDigest
		}},
		{name: "selector", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.Target.Mixture.SelectorPolicyDigest = digestString("selector-policy-v2")
			manifest.Target.Mixture.SelectorDigest = selectorSnapshotDigest(
				manifest.Target.Mixture.SelectorPolicyDigest,
				manifest.Target.Mixture.SupportModels,
			)
		}},
		{name: "adaptation", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.Target.Mixture.AdaptationDigest = digestString("adaptation-v2")
		}},
		{name: "binding", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Target.Mixture.BindingDigest = digestString("binding-v2")
			report.Provenance.BindingSnapshotDigest = manifest.Target.Mixture.BindingDigest
		}},
		{name: "model pool arm", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Target.Mixture.ModelArms[0].ProviderModelIDDigest = digestString("provider-fast-v2")
			manifest.Target.Mixture.PoolDigest = modelPoolSnapshotDigest(manifest.Target.Mixture.ModelArms)
			report.Provenance.PoolSnapshotDigest = manifest.Target.Mixture.PoolDigest
		}},
		{name: "support model", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.Target.Mixture.SupportModels[0].ConfigDigest = digestString("support-config-v2")
			manifest.Target.Mixture.SelectorDigest = selectorSnapshotDigest(
				manifest.Target.Mixture.SelectorPolicyDigest,
				manifest.Target.Mixture.SupportModels,
			)
		}},
		{name: "backend topology", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.Target.BackendTopologyDigest = digestString("backend-topology-v2")
		}},
	}

	for index, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest, report := candidateSubjectTestEvidence(
				t,
				ModeLive,
				"00000000-0000-4000-8000-"+fmtCandidateSubjectTestSuffix(index+130),
			)
			test.mutate(&manifest, &report)
			mustFreezeTestRoutingRecipePlan(manifest.Target.Mixture)
			report.Run.Mixture = catalogMixtureFromManifest(manifest.Target.Mixture)
			sealCandidateSubjectTestManifest(t, &manifest)
			got := requireCandidateSubjectDigest(t, manifest, report)
			if got == want {
				t.Fatalf("subject factor drift left digest unchanged: %q", got)
			}
		})
	}
}

func TestCandidateSubjectDigestRejectsUnboundOrMismatchedEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*RunManifest, *Report)
	}{
		{name: "fixture replay has no Mixture subject", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Mode, report.Run.Mode = ModeReplay, ModeReplay
			manifest.Target.Kind, manifest.Target.ID = "builtin-fixture", "fixture"
			manifest.Target.Mixture, report.Run.Mixture = nil, nil
			manifest.Target.BackendTopologyDigest = ""
			report.Run.TargetID, report.Provenance.TargetID = "fixture", "fixture"
		}},
		{name: "normalized source has no Mixture subject", mutate: func(manifest *RunManifest, report *Report) {
			manifest.Mode, report.Run.Mode = ModeReplay, ModeReplay
			manifest.Target.Kind, manifest.Target.ID = "normalized-benchmark-source", "benchmark-source"
			manifest.Target.Mixture, report.Run.Mixture = nil, nil
			manifest.Target.BackendTopologyDigest = ""
			report.Run.TargetID, report.Provenance.TargetID = "benchmark-source", "benchmark-source"
		}},
		{name: "manifest/report Mixture mismatch", mutate: func(_ *RunManifest, report *Report) {
			report.Run.Mixture.BindingDigest = digestString("unbound-report-binding")
		}},
		{name: "policy provenance mismatch", mutate: func(_ *RunManifest, report *Report) {
			report.Provenance.PolicySnapshotDigest = digestString("unbound-policy")
		}},
		{name: "missing manifest digest", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.ManifestDigest = ""
		}},
		{name: "missing configuration digest", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.ConfigDigest = ""
		}},
		{name: "missing environment lineage digest", mutate: func(_ *RunManifest, report *Report) {
			report.Provenance.EnvironmentSnapshotDigest = ""
		}},
		{name: "missing backend topology digest", mutate: func(manifest *RunManifest, _ *Report) {
			manifest.Target.BackendTopologyDigest = ""
		}},
		{name: "report is not server attested", mutate: func(_ *RunManifest, report *Report) {
			report.AttestationRevision = ""
		}},
	}

	for index, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			manifest, report := candidateSubjectTestEvidence(
				t,
				ModeLive,
				"00000000-0000-4000-8000-"+fmtCandidateSubjectTestSuffix(index+150),
			)
			test.mutate(&manifest, &report)
			if manifest.ManifestDigest != "" {
				sealCandidateSubjectTestManifest(t, &manifest)
			}
			if _, err := candidateSubjectDigest(manifest, report); err == nil {
				t.Fatal("invalid candidate evidence was accepted")
			}
		})
	}
}

func candidateSubjectTestEvidence(t *testing.T, mode Mode, runID string) (RunManifest, Report) {
	t.Helper()
	mixture := candidateSubjectTestMixture()
	createdAt := time.Date(2026, time.August, 31, 8, 0, 0, 0, time.UTC)
	manifest := RunManifest{
		SchemaVersion: SchemaVersion,
		RunID:         runID,
		Name:          "candidate subject evidence",
		Mode:          mode,
		Target: ManifestTarget{
			SchemaVersion:         SchemaVersion,
			ID:                    mixture.ID,
			Kind:                  "mixture-of-models",
			Mixture:               mixture,
			BackendTopologyDigest: digestString("backend-topology-v1"),
		},
		ChangeProfile:        "recipe",
		GateContractVersion:  GateContractVersion,
		SuiteIDs:             []string{"live-mom-core"},
		SuiteRevisions:       map[string]string{"live-mom-core": "native-reference-v1"},
		SuiteExecutors:       map[string]string{"live-mom-core": liveRuntimeExecutorID},
		TrackIDs:             []TrackID{"routing", "model_pool", "joint"},
		SampleLimit:          64,
		Concurrency:          1,
		Seed:                 17,
		CreatedAt:            createdAt,
		CodeRevision:         strings.Repeat("a", 40),
		ConfigDigest:         digestString("config-v1"),
		PolicySnapshotDigest: mixture.RecipeDigest,
		RedactionPolicy:      "evaluation-default-v1",
	}
	sealCandidateSubjectTestManifest(t, &manifest)
	report := Report{
		SchemaVersion:       SchemaVersion,
		AttestationRevision: ServerAttestationRevision,
		Run: Run{
			SchemaVersion:   SchemaVersion,
			ID:              runID,
			ClientRequestID: runID,
			Name:            manifest.Name,
			Status:          StatusCompleted,
			Mode:            mode,
			TargetID:        manifest.Target.ID,
			Mixture:         catalogMixtureFromManifest(mixture),
			ChangeProfile:   manifest.ChangeProfile,
			SuiteIDs:        append([]string(nil), manifest.SuiteIDs...),
			TrackIDs:        append([]TrackID(nil), manifest.TrackIDs...),
			SampleLimit:     manifest.SampleLimit,
			Concurrency:     manifest.Concurrency,
			Seed:            manifest.Seed,
			CreatedAt:       createdAt,
		},
		Provenance: Provenance{
			SchemaVersion:             SchemaVersion,
			CodeRevision:              manifest.CodeRevision,
			BenchmarkRevisions:        map[string]string{"live-mom-core": "native-reference-v1"},
			PolicySnapshotDigest:      manifest.PolicySnapshotDigest,
			BindingSnapshotDigest:     mixture.BindingDigest,
			PoolSnapshotDigest:        mixture.PoolDigest,
			WorkloadSnapshotDigest:    digestString("workload-v1"),
			EnvironmentSnapshotDigest: digestString("runtime-environment-v1"),
			TargetID:                  manifest.Target.ID,
			Seed:                      manifest.Seed,
		},
	}
	return manifest, report
}

func candidateSubjectTestMixture() *ManifestMixture {
	recipeName := "candidate-recipe-v1"
	arms := []ModelArm{
		{
			ID: "arm-fast", Model: "model-fast", ProviderModelIDDigest: digestString("provider-fast"),
			InputCostPerMillionTokensUSD: 1, OutputCostPerMillionTokensUSD: 2,
			Capabilities: []string{"chat"}, Modalities: []string{"text"},
		},
		{
			ID: "arm-strong", Model: "model-strong", ProviderModelIDDigest: digestString("provider-strong"),
			InputCostPerMillionTokensUSD: 3, OutputCostPerMillionTokensUSD: 4,
			Capabilities: []string{"chat", "vision"}, Modalities: []string{"text", "image"},
		},
	}
	support := []SupportModel{{
		Model: "selector-model", ProviderModelIDDigest: digestString("selector-provider"),
		ConfigDigest: digestString("selector-config"), BackendTopologyDigest: digestString("selector-topology"),
	}}
	selectorPolicyDigest := digestString("selector-policy-v1")
	mixture := &ManifestMixture{
		SchemaVersion:        SchemaVersion,
		ID:                   "mom-" + strings.TrimPrefix(digestString(recipeName), "sha256:"),
		EntrypointModel:      "virtual-entrypoint",
		Aliases:              []string{"virtual-entrypoint", "virtual-entrypoint-alias"},
		RecipeName:           recipeName,
		RecipeDescription:    "presentation metadata",
		RecipeDigest:         digestString("recipe-v1"),
		PoolDigest:           modelPoolSnapshotDigest(arms),
		SelectorPolicyDigest: selectorPolicyDigest,
		SelectorDigest:       selectorSnapshotDigest(selectorPolicyDigest, support),
		AdaptationDigest:     digestString("adaptation-v1"),
		BindingDigest:        digestString("binding-v1"),
		ModelArms:            arms,
		SupportModels:        support,
		FallbackArmID:        "arm-fast",
		Decisions: []MixtureDecisionBinding{{
			Name: "route", Algorithm: "static", ArmIDs: []string{"arm-fast", "arm-strong"},
		}},
	}
	mustFreezeTestRoutingRecipePlan(mixture)
	return mixture
}

func sealCandidateSubjectTestManifest(t *testing.T, manifest *RunManifest) {
	t.Helper()
	manifest.ManifestDigest = ""
	digest, err := manifestSemanticDigest(*manifest)
	if err != nil {
		t.Fatalf("manifestSemanticDigest: %v", err)
	}
	manifest.ManifestDigest = digest
}

func requireCandidateSubjectDigest(t *testing.T, manifest RunManifest, report Report) string {
	t.Helper()
	digest, err := candidateSubjectDigest(manifest, report)
	if err != nil {
		t.Fatalf("candidateSubjectDigest: %v", err)
	}
	if !digestPattern.MatchString(digest) {
		t.Fatalf("candidate subject digest = %q, want sha256 digest", digest)
	}
	return digest
}

func fmtCandidateSubjectTestSuffix(value int) string {
	return fmt.Sprintf("%012d", value)
}
