package evaluationplane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRealSealedRunKeepsSemanticAndArtifactManifestDigestsDistinct(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create run: %v", err)
	}
	run = stageSealingTestRun(t, service, run)
	spec := ProcessSpec{
		ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName),
		StorePath:    root,
	}
	if writeReportErr := writeProcessReport(spec); writeReportErr != nil {
		t.Fatalf("write production-shaped worker bundle: %v", writeReportErr)
	}
	if sealReportErr := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	}); sealReportErr != nil {
		t.Fatalf("seal report through production validator: %v", sealReportErr)
	}
	if closeErr := service.Close(); closeErr != nil {
		t.Fatalf("close sealing service before reload: %v", closeErr)
	}
	reloaded, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("reload service from sealed store: %v", err)
	}
	t.Cleanup(func() { _ = reloaded.Close() })
	service = reloaded

	manifest, manifestBytes, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatalf("reload durable manifest: %v", err)
	}
	anchor, err := service.store.readReportAnchor(run.ID)
	if err != nil {
		t.Fatalf("reload report anchor: %v", err)
	}
	artifactDigest := digestBytes(manifestBytes)
	if anchor.ManifestSemanticDigest != manifest.ManifestDigest ||
		anchor.ManifestArtifactDigest != artifactDigest {
		t.Fatalf("manifest identities are not exact: manifest=%+v anchor=%+v", manifest, anchor)
	}
	if anchor.ManifestSemanticDigest == anchor.ManifestArtifactDigest {
		t.Fatal("semantic manifest identity was conflated with serialized artifact bytes")
	}
	if _, verifyReportErr := service.ReportJSONAs(SystemActor(), run.ID); verifyReportErr != nil {
		t.Fatalf("real sealed report failed reload verification: %v", verifyReportErr)
	}

	manifestPath := filepath.Join(service.store.runsRoot, run.ID, manifestFileName)
	if reformatManifestErr := os.WriteFile(manifestPath, append(manifestBytes, '\n'), 0o600); reformatManifestErr != nil {
		t.Fatalf("reformat durable manifest: %v", reformatManifestErr)
	}
	reformatted, _, err := service.readDurableManifest(run.ID)
	if err != nil || reformatted.ManifestDigest != manifest.ManifestDigest {
		t.Fatalf("semantic identity changed after whitespace-only artifact change: manifest=%+v err=%v", reformatted, err)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("whitespace-tampered manifest artifact error=%v, want ErrInvalid", err)
	}
}

func TestCampaignReferenceRejectsManifestArtifactSubstitutionAfterSeal(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("create run: %v", err)
	}
	run = stageSealingTestRun(t, service, run)
	if writeReportErr := writeProcessReport(ProcessSpec{
		ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root,
	}); writeReportErr != nil {
		t.Fatalf("write production-shaped worker bundle: %v", writeReportErr)
	}
	if sealReportErr := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	}); sealReportErr != nil {
		t.Fatalf("seal report through production validator: %v", sealReportErr)
	}
	anchor, err := service.store.readReportAnchor(run.ID)
	if err != nil {
		t.Fatal(err)
	}
	campaign := Campaign{
		ID: newTestClientRequestID(), CreatedAt: anchor.CreatedAt.Add(time.Second),
		Decision: CampaignDecision{Evidence: []CampaignEvidenceAnchor{{
			RunID:                  run.ID,
			ManifestSemanticDigest: anchor.ManifestSemanticDigest,
			ManifestArtifactDigest: anchor.ManifestArtifactDigest,
			ReportDigest:           anchor.ReportDigest,
			PrivateReceiptDigest:   anchor.PrivateReceiptDigest,
		}}},
	}
	if validateReferenceErr := service.store.validateCampaignRunReferencesUnlocked(campaign); validateReferenceErr != nil {
		t.Fatalf("real campaign reference rejected: %v", validateReferenceErr)
	}
	manifestPath := filepath.Join(service.store.runsRoot, run.ID, manifestFileName)
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(manifestBytes, ' '), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := service.store.validateCampaignRunReferencesUnlocked(campaign); err == nil {
		t.Fatal("campaign accepted substituted manifest bytes with the same semantic identity")
	}
}

func TestControlledPairSourceRevalidatesBothManifestIdentities(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	if err := os.WriteFile(filepath.Join(root, "config.yaml"), []byte(modelArmTestYAML), 0o600); err != nil {
		t.Fatalf("write live Mixture config: %v", err)
	}
	request := CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "controlled pair source",
		SuiteIDs: []string{"live-mom-core"}, TrackIDs: []TrackID{"routing"},
		Mode: ModeLive, TargetID: mixtureTargetID("default"), ChangeProfile: "recipe",
		SampleLimit: 4, Concurrency: 1, Seed: 17,
	}
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("create live run: %v", err)
	}
	manifest, _, err := service.readDurableManifest(run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if writeRecordsErr := os.WriteFile(filepath.Join(service.store.runsRoot, run.ID, "records.jsonl"), []byte{}, 0o600); writeRecordsErr != nil {
		t.Fatal(writeRecordsErr)
	}
	attestation := validExecutionAttestation(t, run.ID)
	attestation.ManifestDigest = manifest.ManifestDigest
	attestation.TargetID = manifest.Target.ID
	attestation.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	attestation.BackendTopologyDigest = manifest.Target.BackendTopologyDigest
	routingRecipeReport := controlledPairRoutingRecipeReport(t, manifest, &attestation)
	refreshExecutionAttestationDigests(t, &attestation)
	if writeAttestationErr := service.store.writeExecutionAttestation(attestation); writeAttestationErr != nil {
		t.Fatalf("write real execution attestation: %v", writeAttestationErr)
	}
	report := reportForRun(run, nil)
	report.Provenance.BenchmarkRevisions = copyCampaignRevisionMap(manifest.SuiteRevisions)
	report.Provenance.RedactionPolicy = manifest.RedactionPolicy
	report.RoutingRecipeReport = routingRecipeReport
	writeAnchoredControlledPairReport(t, service, run.ID, report)
	anchor, err := service.store.readReportAnchor(run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if removeAnchorErr := os.Remove(filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName)); removeAnchorErr != nil {
		t.Fatal(removeAnchorErr)
	}
	anchor.ExecutionAttestationDigest = attestation.Digest
	if writeAnchorErr := service.store.writeReportAnchor(run.ID, anchor); writeAnchorErr != nil {
		t.Fatal(writeAnchorErr)
	}
	if _, loadSourceErr := service.loadControlledPairSource(run.ID); loadSourceErr != nil {
		t.Fatalf("real controlled-pair source rejected: %v", loadSourceErr)
	}

	manifestPath := filepath.Join(service.store.runsRoot, run.ID, manifestFileName)
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(manifestBytes, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.loadControlledPairSource(run.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("controlled pair accepted substituted manifest artifact: %v", err)
	}
}

func TestCompareRejectsManifestArtifactTamperAfterBothRunsWereSealed(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	baseline := sealReplayRunThroughProductionValidator(t, service, root, validCreateRequest())
	if err := service.Close(); err != nil {
		t.Fatal(err)
	}
	service = reopenManifestAnchorTestService(t, root, testSourceRevision)
	if recovered, err := service.GetRunAs(SystemActor(), baseline.ID); err != nil || recovered.Status != StatusCompleted {
		t.Fatalf("recover production-sealed baseline: run=%+v err=%v", recovered, err)
	}

	service.codeRevision = strings.Repeat("b", 40)
	candidateRequest := validCreateRequest()
	candidateRequest.Name = "candidate"
	candidateRequest.BaselineRunID = baseline.ID
	candidate := sealReplayRunThroughProductionValidator(t, service, root, candidateRequest)
	if err := service.Close(); err != nil {
		t.Fatal(err)
	}
	service = reopenManifestAnchorTestService(t, root, strings.Repeat("b", 40))
	t.Cleanup(func() { _ = service.Close() })
	if recovered, err := service.GetRunAs(SystemActor(), candidate.ID); err != nil || recovered.Status != StatusCompleted {
		t.Fatalf("recover production-sealed candidate: run=%+v err=%v", recovered, err)
	}
	if _, err := service.CompareAs(SystemActor(), baseline.ID, candidate.ID); err != nil {
		t.Fatalf("compare two production-sealed runs: %v", err)
	}

	manifestPath := filepath.Join(service.store.runsRoot, candidate.ID, manifestFileName)
	manifestBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(manifestPath, append(manifestBytes, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := service.CompareAs(SystemActor(), baseline.ID, candidate.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("Compare accepted a manifest artifact substituted after seal: %v", err)
	}
}

func sealReplayRunThroughProductionValidator(
	t *testing.T,
	service *Service,
	root string,
	request CreateRunRequest,
) Run {
	t.Helper()
	run, err := service.CreateRunAs(context.Background(), SystemActor(), request)
	if err != nil {
		t.Fatalf("create production-sealed run: %v", err)
	}
	run = stageSealingTestRun(t, service, run)
	if err := writeProcessReport(ProcessSpec{
		ManifestPath: filepath.Join(root, "runs", run.ID, manifestFileName), StorePath: root,
	}); err != nil {
		t.Fatalf("write production-shaped worker bundle: %v", err)
	}
	if err := service.store.withEvidencePublication(func() error {
		return service.validateAndAnchorReportDuringPublication(run.ID)
	}); err != nil {
		t.Fatalf("seal report through production validator: %v", err)
	}
	return run
}

func reopenManifestAnchorTestService(t *testing.T, root, revision string) *Service {
	t.Helper()
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: filepath.Join(root, "config.yaml"),
		RouterAPIURL: "http://router.invalid", EnvoyURL: "http://envoy.invalid",
		CodeRevision: revision, MaxConcurrent: 1, Process: &controlledProcess{},
	})
	if err != nil {
		t.Fatalf("reopen production-sealed store: %v", err)
	}
	return service
}
