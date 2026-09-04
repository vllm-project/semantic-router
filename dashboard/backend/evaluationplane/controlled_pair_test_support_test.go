package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type controlledPairStoreTestProcess struct {
	controlledProcess
	freezes int
}

const (
	controlledPairCampaignSuiteID       = "controlled-pair-campaign-test"
	controlledPairCampaignSuiteRevision = "controlled-pair-campaign-test-v1"
)

func controlledPairCampaignSuite() CatalogSuite {
	return CatalogSuite{
		ID:          controlledPairCampaignSuiteID,
		Name:        "Controlled pair campaign test",
		Description: "Synthetic campaign cohort used only by controlled-pair service tests.",
		Executors: map[Mode]string{
			ModeReplay: momReplayExecutorID,
			ModeLive:   liveRuntimeExecutorID,
		},
		TrackIDs:      []TrackID{"routing", "model_pool", "joint"},
		Modes:         []Mode{ModeReplay, ModeLive},
		EvidenceLevel: "E0",
		CaseCount:     64,
		CampaignProtocol: &CampaignProtocol{
			SchemaVersion: campaignCohortSchemaVersion,
			MinimumCases:  campaignPairedMinimumCases,
		},
		Revision: controlledPairCampaignSuiteRevision,
		Tags:     []string{"test-only", "campaign"},
		Methods: []CatalogMethod{
			configuredCatalogMethod("controlled-pair-test.routing.v1", "routing", nil, CatalogMethodEvidenceSourceLiveRuntime),
			configuredCatalogMethod("controlled-pair-test.model-pool.v1", "model_pool", nil, CatalogMethodEvidenceSourceLiveRuntime),
			configuredCatalogMethod("controlled-pair-test.joint.v1", "joint", nil, CatalogMethodEvidenceSourceLiveRuntime),
		},
	}
}

func controlledPairRegistryConstructor(
	routerAPIURL, envoyURL string,
	registryOptions ...RegistryOptions,
) (*Registry, error) {
	registry, err := NewRegistry(routerAPIURL, envoyURL, registryOptions...)
	if err != nil {
		return nil, err
	}
	if err := registry.registerSuite(controlledPairCampaignSuite()); err != nil {
		return nil, err
	}
	return registry, nil
}

func newControlledPairTestService(options Options) (*Service, error) {
	return newService(options, controlledPairRegistryConstructor)
}

func (p *controlledPairStoreTestProcess) Run(
	ctx context.Context,
	spec ProcessSpec,
	_ func(WorkerEvent) error,
) (ProcessResult, error) {
	p.calls.Add(1)
	if p.started != nil {
		p.started <- spec
	}
	<-ctx.Done()
	return ProcessResult{}, ctx.Err()
}

func (p *controlledPairStoreTestProcess) freezeControlledPairCredentials(
	context.Context,
	RunManifest,
) (workerBrokerCredentials, error) {
	p.freezes++
	return workerBrokerCredentials{}, nil
}

func pendingControlledPairAggregate(
	t *testing.T,
	service *Service,
	actor Actor,
) (controlledPairManifest, RunManifest, RunManifest) {
	t.Helper()
	deployments, loadErr := LoadEvaluationDeploymentRegistry(service.registrySource.deploymentsDir, "")
	if loadErr != nil || len(deployments) < 2 {
		t.Fatalf("load controlled pair source deployments: count=%d err=%v", len(deployments), loadErr)
	}
	baselineSource := createStoreControlledPairSource(t, service, actor, deployments[0].TargetID)
	candidateSource := createStoreControlledPairSource(t, service, actor, deployments[1].TargetID)
	createdAt := time.Now().UTC().Truncate(time.Microsecond)
	baseline, baselineManifest, err := cloneControlledPairRun(
		baselineSource, newTestClientRequestID(), "", controlledPairRoleBaseline, createdAt,
	)
	if err != nil {
		t.Fatalf("clone controlled pair baseline: %v", err)
	}
	candidate, candidateManifest, err := cloneControlledPairRun(
		candidateSource, newTestClientRequestID(), baseline.ID, controlledPairRoleCandidate, createdAt.Add(time.Microsecond),
	)
	if err != nil {
		t.Fatalf("clone controlled pair candidate: %v", err)
	}
	request := CreateControlledPairRequest{
		ClientRequestID: newTestClientRequestID(), BaselineSourceRunID: baselineSource.run.ID,
		CandidateSourceRunID: candidateSource.run.ID, BaselineRunID: baseline.ID, CandidateRunID: candidate.ID,
	}
	pair, err := newControlledPairManifest(
		actor, request, baselineSource, candidateSource, baseline, candidate, baselineManifest, candidateManifest,
	)
	if err != nil {
		t.Fatalf("build controlled pair manifest: %v", err)
	}
	return pair, baselineManifest, candidateManifest
}

func newControlledPairStoreTestService(t *testing.T) (*Service, string) {
	t.Helper()
	service, _, _ := newControlledPairExecutionTestService(t, &controlledPairStoreTestProcess{}, 8)
	t.Cleanup(func() { _ = service.Close() })
	return service, service.store.Root()
}

func createStoreControlledPairSource(
	t *testing.T,
	service *Service,
	actor Actor,
	targetID string,
) controlledPairSource {
	t.Helper()
	run, createErr := service.CreateRunAs(context.Background(), actor, CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "store controlled pair source",
		SuiteIDs: []string{controlledPairCampaignSuiteID}, TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		Mode: ModeLive, TargetID: targetID, ChangeProfile: "recipe",
		SampleLimit: 64, Concurrency: 1, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("create controlled pair source: %v", createErr)
	}
	manifest, _, manifestErr := service.readDurableManifest(run.ID)
	if manifestErr != nil {
		t.Fatalf("read controlled pair source manifest: %v", manifestErr)
	}
	if err := os.WriteFile(filepath.Join(service.store.runsRoot, run.ID, "records.jsonl"), []byte{}, 0o600); err != nil {
		t.Fatalf("write controlled pair source records: %v", err)
	}
	attestation := validExecutionAttestation(t, run.ID)
	attestation.ManifestDigest = manifest.ManifestDigest
	attestation.TargetID = manifest.Target.ID
	attestation.Mode = manifest.Mode
	attestation.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	attestation.BackendTopologyDigest = manifest.Target.BackendTopologyDigest
	routingRecipeReport := controlledPairRoutingRecipeReport(t, manifest, &attestation)
	refreshExecutionAttestationDigests(t, &attestation)
	if err := service.store.writeExecutionAttestation(attestation); err != nil {
		t.Fatalf("write controlled pair source attestation: %v", err)
	}
	report := reportForRun(run, nil)
	report.Provenance.CodeRevision = manifest.CodeRevision
	report.Provenance.BenchmarkRevisions = copyCampaignRevisionMap(manifest.SuiteRevisions)
	report.Provenance.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	report.Provenance.PoolSnapshotDigest = run.Mixture.PoolDigest
	report.Provenance.BindingSnapshotDigest = run.Mixture.BindingDigest
	report.Provenance.WorkloadSnapshotDigest = digestString("store-controlled-pair-workload")
	report.Provenance.EnvironmentSnapshotDigest = digestString("store-controlled-pair-environment")
	report.Provenance.RedactionPolicy = manifest.RedactionPolicy
	report.RoutingRecipeReport = routingRecipeReport
	writeAnchoredControlledPairReport(t, service, run.ID, report)
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatalf("read controlled pair source anchor: %v", anchorErr)
	}
	if err := os.Remove(filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName)); err != nil {
		t.Fatalf("replace controlled pair source anchor: %v", err)
	}
	anchor.ExecutionAttestationDigest = attestation.Digest
	if err := service.store.writeReportAnchor(run.ID, anchor); err != nil {
		t.Fatalf("write controlled pair source anchor: %v", err)
	}
	completed, completedErr := service.store.GetRun(run.ID)
	if completedErr != nil {
		t.Fatalf("read controlled pair source: %v", completedErr)
	}
	manifestBytes, manifestBytesErr := readEvidenceBytes(filepath.Join(service.store.runsRoot, run.ID, manifestFileName), maxStructuredArtifactBytes)
	if manifestBytesErr != nil {
		t.Fatalf("read controlled pair source manifest bytes: %v", manifestBytesErr)
	}
	anchorBytes, anchorBytesErr := readEvidenceBytes(filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName), maxReportAnchorBytes)
	if anchorBytesErr != nil {
		t.Fatalf("read controlled pair source anchor bytes: %v", anchorBytesErr)
	}
	reportValue, reportErr := service.decodedReport(run.ID)
	if reportErr != nil {
		t.Fatalf("decode controlled pair source report: %v", reportErr)
	}
	manifestArtifactDigest, _ := digestAndSize(manifestBytes)
	anchorDigest, _ := digestAndSize(anchorBytes)
	return controlledPairSource{
		run: completed, manifest: manifest, report: reportValue,
		manifestArtifactDigest: manifestArtifactDigest, anchorDigest: anchorDigest,
		attestationDigest: attestation.Digest,
	}
}

func sealExistingControlledPairMember(t *testing.T, service *Service, run Run) controlledPairSource {
	t.Helper()
	manifest, manifestBytes, manifestErr := service.readDurableManifest(run.ID)
	if manifestErr != nil {
		t.Fatalf("read member manifest: %v", manifestErr)
	}
	if err := os.WriteFile(filepath.Join(service.store.runsRoot, run.ID, "records.jsonl"), []byte{}, 0o600); err != nil {
		t.Fatalf("write member records: %v", err)
	}
	attestation := validExecutionAttestation(t, run.ID)
	attestation.ManifestDigest, attestation.TargetID, attestation.Mode = manifest.ManifestDigest, manifest.Target.ID, manifest.Mode
	attestation.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	attestation.BackendTopologyDigest = manifest.Target.BackendTopologyDigest
	routingRecipeReport := controlledPairRoutingRecipeReport(t, manifest, &attestation)
	refreshExecutionAttestationDigests(t, &attestation)
	if err := service.store.writeExecutionAttestation(attestation); err != nil {
		t.Fatalf("write member attestation: %v", err)
	}
	report := reportForRun(run, nil)
	report.Provenance.CodeRevision = manifest.CodeRevision
	report.Provenance.BenchmarkRevisions = copyCampaignRevisionMap(manifest.SuiteRevisions)
	report.Provenance.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	report.Provenance.PoolSnapshotDigest = run.Mixture.PoolDigest
	report.Provenance.BindingSnapshotDigest = run.Mixture.BindingDigest
	report.Provenance.WorkloadSnapshotDigest = digestString("controlled-pair-member-workload")
	report.Provenance.EnvironmentSnapshotDigest = digestString("controlled-pair-member-environment")
	report.Provenance.RedactionPolicy = manifest.RedactionPolicy
	report.RoutingRecipeReport = routingRecipeReport
	writeAnchoredControlledPairReport(t, service, run.ID, report)
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatal(anchorErr)
	}
	if err := os.Remove(filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName)); err != nil {
		t.Fatal(err)
	}
	anchor.ExecutionAttestationDigest = attestation.Digest
	if err := service.store.writeReportAnchor(run.ID, anchor); err != nil {
		t.Fatal(err)
	}
	anchorBytes, anchorBytesErr := readEvidenceBytes(
		filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName), maxReportAnchorBytes,
	)
	if anchorBytesErr != nil {
		t.Fatal(anchorBytesErr)
	}
	manifestArtifactDigest, _ := digestAndSize(manifestBytes)
	anchorDigest, _ := digestAndSize(anchorBytes)
	return controlledPairSource{
		run: run, manifest: manifest, report: report,
		manifestArtifactDigest: manifestArtifactDigest, anchorDigest: anchorDigest,
		attestationDigest: attestation.Digest,
	}
}

type recordingControlledPairPersistence struct {
	delegate                    controlledPairPersistence
	fail                        string
	failManifestDirectorySync   bool
	failManifestDirectorySyncAt int
	manifestWrites              int
	operations                  []string
	removeAll                   func(string) error
}

type postLinkFailingLifecycleAuditWriter struct {
	delegate lifecycleAuditWriter
	fail     bool
}

func (w *postLinkFailingLifecycleAuditWriter) WriteExclusive(path string, value any) error {
	if err := w.delegate.WriteExclusive(path, value); err != nil {
		return err
	}
	if w.fail {
		w.fail = false
		return errors.New("injected audit directory sync failure after link")
	}
	return nil
}

func (w *postLinkFailingLifecycleAuditWriter) SyncDirectory(path, description string) error {
	return w.delegate.SyncDirectory(path, description)
}

func (p *recordingControlledPairPersistence) record(operation string) error {
	p.operations = append(p.operations, operation)
	if p.fail == operation {
		p.fail = ""
		return errors.New("injected controlled pair persistence failure")
	}
	return nil
}

func (p *recordingControlledPairPersistence) EnsurePrivateDirectory(path string) (bool, error) {
	if err := p.record("mkdir"); err != nil {
		return false, err
	}
	return p.delegate.EnsurePrivateDirectory(path)
}

func (p *recordingControlledPairPersistence) RemoveAll(path string) error {
	if p.removeAll != nil {
		return p.removeAll(path)
	}
	return p.delegate.RemoveAll(path)
}

func (p *recordingControlledPairPersistence) SyncDirectory(path, description string) error {
	if err := p.record("sync_parent"); err != nil {
		return err
	}
	return p.delegate.SyncDirectory(path, description)
}

func (p *recordingControlledPairPersistence) WriteManifest(path string, pair controlledPairManifest) error {
	p.manifestWrites++
	if err := p.record("write_manifest"); err != nil {
		return err
	}
	if p.failManifestDirectorySync || p.failManifestDirectorySyncAt == p.manifestWrites {
		p.failManifestDirectorySync = false
		encoded, err := json.MarshalIndent(pair, "", "  ")
		if err != nil {
			return err
		}
		temp, err := os.CreateTemp(filepath.Dir(path), ".tmp-evaluation-*")
		if err != nil {
			return err
		}
		tempName := temp.Name()
		defer func() { _ = os.Remove(tempName) }()
		if err := temp.Chmod(0o600); err != nil {
			_ = temp.Close()
			return err
		}
		if _, err := temp.Write(append(encoded, '\n')); err != nil {
			_ = temp.Close()
			return err
		}
		if err := temp.Sync(); err != nil {
			_ = temp.Close()
			return err
		}
		if err := temp.Close(); err != nil {
			return err
		}
		if err := os.Rename(tempName, path); err != nil {
			return err
		}
		p.operations = append(p.operations, "manifest_sync_failure")
		return errors.New("injected controlled pair manifest directory sync failure")
	}
	return p.delegate.WriteManifest(path, pair)
}

func (p *recordingControlledPairPersistence) Rename(source, destination string) error {
	if err := p.record("rename"); err != nil {
		return err
	}
	return p.delegate.Rename(source, destination)
}

func assertOperationBefore(t *testing.T, operations []string, first, second string) {
	t.Helper()
	positions := map[string]int{first: -1, second: -1}
	for index, operation := range operations {
		if _, tracked := positions[operation]; tracked && positions[operation] == -1 {
			positions[operation] = index
		}
	}
	if positions[first] < 0 || positions[second] < 0 || positions[first] >= positions[second] {
		t.Fatalf("operation order %s before %s not satisfied: %v", first, second, operations)
	}
}

func assertPairProjectionCount(t *testing.T, runs []Run, pair controlledPairManifest, want int) {
	t.Helper()
	count := 0
	for _, run := range runs {
		if run.ID == pair.BaselineRunID || run.ID == pair.CandidateRunID {
			count++
		}
	}
	if count != want {
		t.Fatalf("pair projection count=%d, want %d", count, want)
	}
}

func assertPairProjectionStatus(t *testing.T, runs []Run, pair controlledPairManifest, want RunStatus) {
	t.Helper()
	count := 0
	for _, run := range runs {
		if run.ID == pair.BaselineRunID || run.ID == pair.CandidateRunID {
			count++
			if run.Status != want {
				t.Fatalf("pair projection exposed status %s, want %s", run.Status, want)
			}
		}
	}
	if count != 2 {
		t.Fatalf("pair projection exposed %d members, want 2", count)
	}
}

func assertStrictPendingPair(t *testing.T, store *Store, pair controlledPairManifest) {
	t.Helper()
	baseline, baselineErr := store.GetRun(pair.BaselineRunID)
	candidate, candidateErr := store.GetRun(pair.CandidateRunID)
	if baselineErr != nil || candidateErr != nil || baseline.Status != StatusPending ||
		candidate.Status != StatusPending || candidate.BaselineRunID != baseline.ID ||
		!baseline.CreatedAt.Before(candidate.CreatedAt) {
		t.Fatalf("stored pair is not strict pending: baseline=%+v/%v candidate=%+v/%v", baseline, baselineErr, candidate, candidateErr)
	}
}

func assertControlledPairAbsent(t *testing.T, store *Store, pair controlledPairManifest) {
	t.Helper()
	for _, runID := range []string{pair.BaselineRunID, pair.CandidateRunID} {
		if _, err := store.GetRun(runID); !errors.Is(err, ErrNotFound) {
			t.Fatalf("controlled pair run %s unexpectedly exists: %v", runID, err)
		}
	}
	if _, err := store.readControlledPair(pair.PairID); !errors.Is(err, ErrNotFound) {
		t.Fatalf("controlled pair aggregate unexpectedly exists: %v", err)
	}
	staged, err := filepath.Glob(filepath.Join(store.runsRoot, stagedRunBundlePrefix+"*"))
	if err != nil || len(staged) != 0 {
		t.Fatalf("staged controlled pair bundles remained: %v err=%v", staged, err)
	}
}

func newControlledPairExecutionTestService(
	t *testing.T,
	process Process,
	maxConcurrent int,
) (*Service, string, string) {
	t.Helper()
	root := t.TempDir()
	storeRoot := filepath.Join(root, "evaluation")
	if err := os.Mkdir(storeRoot, 0o700); err != nil {
		t.Fatalf("create controlled pair store: %v", err)
	}
	deploymentsRoot := filepath.Join(root, "deployments")
	if err := os.Mkdir(deploymentsRoot, 0o700); err != nil {
		t.Fatalf("create controlled pair deployment registry: %v", err)
	}
	baselineConfig := []byte(modelArmTestYAML)
	candidateConfig := []byte(strings.Replace(
		modelArmTestYAML, "routing:\n  modelCards:", "routing:\n  strategy: confidence\n  modelCards:", 1,
	))
	writeDeploymentRegistryFixture(t, deploymentsRoot, []evaluationDeploymentDefinition{
		{
			ID: "baseline", Name: "Baseline", ConfigFile: "baseline.yaml",
			RouterOrigin: "https://baseline-router.internal", EnvoyOrigin: "https://baseline-envoy.internal",
		},
		{
			ID: "candidate", Name: "Candidate", ConfigFile: "candidate.yaml",
			RouterOrigin: "https://candidate-router.internal", EnvoyOrigin: "https://candidate-envoy.internal",
		},
	}, map[string][]byte{"baseline.yaml": baselineConfig, "candidate.yaml": candidateConfig})
	service, err := newControlledPairTestService(Options{
		DataDir: storeRoot, PythonPath: "python3",
		ConfigPath: filepath.Join(deploymentsRoot, "baseline.yaml"), DeploymentsDir: deploymentsRoot,
		CodeRevision: testSourceRevision, MaxConcurrent: maxConcurrent, Process: process,
	})
	if err != nil {
		t.Fatalf("NewService with controlled pair deployments: %v", err)
	}
	deployments, err := LoadEvaluationDeploymentRegistry(deploymentsRoot, "")
	if err != nil || len(deployments) != 2 {
		t.Fatalf("reload controlled pair deployments: count=%d err=%v", len(deployments), err)
	}
	return service, deployments[0].TargetID, deployments[1].TargetID
}

func createSealedControlledPairSource(t *testing.T, service *Service, targetID string) Run {
	t.Helper()
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), CreateRunRequest{
		ClientRequestID: newTestClientRequestID(), Name: "controlled pair source",
		SuiteIDs: []string{controlledPairCampaignSuiteID}, TrackIDs: []TrackID{"routing", "model_pool", "joint"},
		Mode: ModeLive, TargetID: targetID, ChangeProfile: "recipe",
		SampleLimit: 64, Concurrency: 1, Seed: 17,
	})
	if createErr != nil {
		t.Fatalf("create controlled pair source: %v", createErr)
	}
	manifest, _, manifestErr := service.readDurableManifest(run.ID)
	if manifestErr != nil {
		t.Fatalf("read controlled pair source manifest: %v", manifestErr)
	}
	if err := os.WriteFile(filepath.Join(service.store.runsRoot, run.ID, "records.jsonl"), []byte{}, 0o600); err != nil {
		t.Fatalf("write controlled pair source records: %v", err)
	}
	attestation := validExecutionAttestation(t, run.ID)
	attestation.ManifestDigest = manifest.ManifestDigest
	attestation.TargetID = manifest.Target.ID
	attestation.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	attestation.BackendTopologyDigest = manifest.Target.BackendTopologyDigest
	routingRecipeReport := controlledPairRoutingRecipeReport(t, manifest, &attestation)
	refreshExecutionAttestationDigests(t, &attestation)
	if err := service.store.writeExecutionAttestation(attestation); err != nil {
		t.Fatalf("write controlled pair source attestation: %v", err)
	}
	report := reportForRun(run, nil)
	report.Provenance.CodeRevision = manifest.CodeRevision
	report.Provenance.BenchmarkRevisions = copyCampaignRevisionMap(manifest.SuiteRevisions)
	report.Provenance.PolicySnapshotDigest = manifest.PolicySnapshotDigest
	report.Provenance.PoolSnapshotDigest = run.Mixture.PoolDigest
	report.Provenance.BindingSnapshotDigest = run.Mixture.BindingDigest
	report.Provenance.WorkloadSnapshotDigest = digestString("controlled-pair-workload")
	report.Provenance.EnvironmentSnapshotDigest = digestString("controlled-pair-environment")
	report.Provenance.RedactionPolicy = manifest.RedactionPolicy
	report.RoutingRecipeReport = routingRecipeReport
	writeAnchoredControlledPairReport(t, service, run.ID, report)
	anchor, anchorErr := service.store.readReportAnchor(run.ID)
	if anchorErr != nil {
		t.Fatalf("read controlled pair source anchor: %v", anchorErr)
	}
	if err := os.Remove(filepath.Join(service.store.runsRoot, run.ID, reportAnchorFileName)); err != nil {
		t.Fatalf("replace controlled pair source anchor: %v", err)
	}
	anchor.ExecutionAttestationDigest = attestation.Digest
	if err := service.store.writeReportAnchor(run.ID, anchor); err != nil {
		t.Fatalf("write controlled pair source anchor: %v", err)
	}
	completed, completedErr := service.store.GetRun(run.ID)
	if completedErr != nil {
		t.Fatalf("read completed controlled pair source: %v", completedErr)
	}
	return completed
}

func writeAnchoredControlledPairReport(t *testing.T, service *Service, runID string, report Report) {
	t.Helper()
	writeAnchoredTestReport(t, service, runID, report)
	if report.RoutingRecipeReport == nil {
		return
	}
	data, readErr := service.store.ReadReport(runID)
	if readErr != nil {
		t.Fatalf("read controlled-pair report before server aggregate injection: %v", readErr)
	}
	var sealed Report
	if err := json.Unmarshal(data, &sealed); err != nil {
		t.Fatalf("decode controlled-pair report before server aggregate injection: %v", err)
	}
	sealed.RoutingRecipeReport = report.RoutingRecipeReport
	if err := service.store.WriteReport(runID, sealed); err != nil {
		t.Fatalf("inject controlled-pair server routing aggregate: %v", err)
	}
	anchor, anchorErr := service.store.readReportAnchor(runID)
	if anchorErr != nil {
		t.Fatalf("read controlled-pair report anchor before aggregate injection: %v", anchorErr)
	}
	if err := os.Remove(filepath.Join(service.store.runsRoot, runID, reportAnchorFileName)); err != nil {
		t.Fatalf("replace controlled-pair report anchor after aggregate injection: %v", err)
	}
	data, readErr = service.store.ReadReport(runID)
	if readErr != nil {
		t.Fatalf("read controlled-pair report after aggregate injection: %v", readErr)
	}
	anchor.ReportDigest, anchor.ReportSize = digestAndSize(data)
	if err := service.store.writeReportAnchor(runID, anchor); err != nil {
		t.Fatalf("write controlled-pair report anchor after aggregate injection: %v", err)
	}
}

func controlledPairRoutingRecipeReport(
	t *testing.T,
	manifest RunManifest,
	attestation *executionAttestation,
) *RoutingRecipeEvaluationReport {
	t.Helper()
	if manifest.Target.Mixture == nil || !containsTrack(manifest.TrackIDs, "routing") {
		t.Fatal("controlled-pair routing source requires a frozen live Mixture routing plan")
	}
	plan := manifest.Target.Mixture.RoutingRecipePlan
	fetchedAt := attestation.StartedAt.Add(attestation.CompletedAt.Sub(attestation.StartedAt) / 2).UTC()
	requestID := uint64(1)
	for range attestation.Entries {
		requestID++
	}
	statusCode := 500
	requestedModel, recipe := manifest.Target.Mixture.EntrypointModel, manifest.Target.Mixture.RecipeName
	signals := make([]RoutingRecipeObservedInput, 0, len(plan.Signals))
	for _, signal := range plan.Signals {
		signals = append(signals, RoutingRecipeObservedInput{ID: signal.ID, State: "missing"})
	}
	projections := make([]RoutingRecipeObservedInput, 0, len(plan.Projections))
	for _, projection := range plan.Projections {
		projections = append(projections, RoutingRecipeObservedInput{ID: projection.ID, State: "missing"})
	}
	eligibility := make([]RoutingRecipeEligibility, 0, len(plan.ArmIDs))
	for _, armID := range plan.ArmIDs {
		eligibility = append(eligibility, RoutingRecipeEligibility{
			ArmID: armID, State: "unavailable", ReasonCode: "router_error",
		})
	}
	decision := RoutingRecipeDecisionSnapshot{
		ContractVersion: RoutingDecisionEvidenceContractVersion,
		DecisionID:      routingRecipeBrokerDecisionID(requestID),
		PlanDigest:      plan.PlanDigest,
		CaseID:          "case-1",
		ObservedAt:      fetchedAt,
		Signals:         signals,
		Projections:     projections,
		Eligibility:     eligibility,
		RankedArmIDs:    []string{},
		SelectionStatus: "error",
	}
	attestation.Entries = append(attestation.Entries, executionAttestationEntry{
		RequestID: requestID, Operation: workerBrokerRouterEvaluate,
		TrackID: "routing", CaseID: "case-1", AttemptID: "attempt-case-1",
		RequestDigest:     digestString("controlled-pair-routing-request:" + manifest.RunID),
		ResponseDigest:    digestString("controlled-pair-routing-response:" + manifest.RunID),
		UpstreamAttempted: true, StatusCode: &statusCode, LatencyMicroseconds: 100,
		FetchedAt: &fetchedAt, Headers: map[string]string{}, RequestedModel: &requestedModel,
		Recipe: &recipe, RoutingRecipeDecision: &decision,
	})
	report, err := ReduceRoutingRecipeEvaluation(RoutingRecipeReductionInput{
		Plan: plan, ExpectedCaseIDs: []string{"case-1"},
		Decisions: []RoutingRecipeDecisionSnapshot{decision}, Outcomes: []RoutingRecipeOutcome{},
	})
	if err != nil {
		t.Fatalf("reduce controlled-pair routing recipe report: %v", err)
	}
	return &report
}

func refreshTestManifestDigest(t *testing.T, manifest *RunManifest) {
	t.Helper()
	manifest.ManifestDigest = ""
	digest, err := manifestSemanticDigest(*manifest)
	if err != nil {
		t.Fatalf("refresh test manifest digest: %v", err)
	}
	manifest.ManifestDigest = digest
}

var (
	_ Process                         = (*controlledPairStoreTestProcess)(nil)
	_ controlledPairCredentialFreezer = (*controlledPairStoreTestProcess)(nil)
)
