package evaluationplane

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func reportForRun(run Run, artifacts []Artifact) Report {
	if artifacts == nil {
		artifacts = []Artifact{}
	}
	run.Status = StatusCompleted
	run.Error = ""
	return Report{
		SchemaVersion:       SchemaVersion,
		AttestationRevision: ServerAttestationRevision,
		Run:                 run,
		Summary: ReportSummary{
			Verdict: "pass", Coverage: Coverage{Evaluated: 1, Total: 1, Fraction: 1}, PassedGates: 1,
		},
		Tracks:          []TrackReport{},
		Metrics:         []Metric{},
		Gates:           []Gate{},
		Costs:           CostLedgers{Runtime: CostAmount{Currency: "USD"}, EvaluationOverhead: CostAmount{Currency: "USD"}, CapacityTCO: CostAmount{Currency: "USD"}},
		Recommendations: []string{},
		Provenance: Provenance{
			SchemaVersion: SchemaVersion, GeneratedAt: time.Now().UTC(), CodeRevision: testSourceRevision,
			TargetID: run.TargetID, Seed: run.Seed, RedactionPolicy: "evaluation-default-v1",
			WorkloadSnapshotDigest: "sha256:test-workload", PoolSnapshotDigest: "sha256:test-pool",
			EnvironmentSnapshotDigest: "sha256:test-environment", PolicySnapshotDigest: "sha256:test-policy",
			BindingSnapshotDigest: "sha256:test-binding", BenchmarkRevisions: map[string]string{"evaluation-smoke": "builtin-v1"},
		},
		Artifacts: artifacts,
	}
}

func writeTestPrivateReceipt(t *testing.T, service *Service, runID string) string {
	t.Helper()
	runDir := filepath.Join(service.store.runsRoot, runID)
	var receipt strings.Builder
	for _, name := range workerRunArtifactNames {
		if name == privateChecksumArtifactName || name == reportFileName {
			continue
		}
		data, err := readEvidenceBytes(filepath.Join(runDir, name), workerArtifactLimit(name))
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			t.Fatalf("read test evidence %s: %v", name, err)
		}
		hexDigest := strings.TrimPrefix(digestBytes(data), "sha256:")
		receipt.WriteString(hexDigest)
		receipt.WriteString("  ")
		receipt.WriteString(name)
		receipt.WriteByte('\n')
		objectPath := filepath.Join(service.store.root, "objects", "sha256", hexDigest)
		file, createErr := os.OpenFile(objectPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if createErr == nil {
			if _, createErr = file.Write(data); createErr == nil {
				createErr = file.Close()
			} else {
				_ = file.Close()
			}
		}
		if createErr != nil && !os.IsExist(createErr) {
			t.Fatalf("write test CAS object %s: %v", name, createErr)
		}
	}
	path := filepath.Join(runDir, privateChecksumArtifactName)
	if err := os.WriteFile(path, []byte(receipt.String()), 0o600); err != nil {
		t.Fatalf("write private receipt: %v", err)
	}
	return digestBytes([]byte(receipt.String()))
}

func sealTestReport(t *testing.T, service *Service, runID string) {
	t.Helper()
	run, err := service.store.GetRun(runID)
	if err != nil {
		t.Fatalf("read run for test seal: %v", err)
	}
	now := time.Now().UTC()
	if terminalStatus(run.Status) {
		if run.Status != StatusCompleted || run.CompletedAt == nil {
			t.Fatalf("seal test report requires a completed run, got %s", run.Status)
		}
		now = run.CompletedAt.UTC()
	} else {
		run.Status = StatusCompleted
		if run.StartedAt == nil {
			run.StartedAt = &now
		}
		run.CompletedAt = &now
		run.Error = ""
		run.Progress = RunProgress{Percent: 100, Completed: len(run.TrackIDs), Total: len(run.TrackIDs), Message: "Evaluation completed"}
		if updateErr := service.store.updateRunFixture(run); updateErr != nil {
			t.Fatalf("complete test run: %v", updateErr)
		}
	}
	reportBytes, reportErr := service.store.ReadReport(runID)
	if reportErr != nil {
		t.Fatalf("read report before test seal: %v", reportErr)
	}
	reportValue, decodeErr := decodeWorkerReportStrict(runID, reportBytes)
	if decodeErr != nil {
		t.Fatalf("decode report before test seal: %v", decodeErr)
	}
	ensureTestPublicReceipt(t, service, runID, &reportValue)
	canonicalizeReportRun(run, &reportValue, now)
	reportValue.AttestationRevision = ServerAttestationRevision
	if writeErr := service.store.WriteReport(runID, reportValue); writeErr != nil {
		t.Fatalf("canonicalize report before test seal: %v", writeErr)
	}
	privateDigest := writeTestPrivateReceipt(t, service, runID)
	checksums, err := service.validatePrivateReceipt(runID)
	if err != nil {
		t.Fatalf("validate private receipt: %v", err)
	}
	evidenceFiles, err := service.buildSealedEvidenceSnapshot(runID, checksums)
	if err != nil {
		t.Fatalf("build sealed evidence snapshot: %v", err)
	}
	report, err := service.store.ReadReport(runID)
	if err != nil {
		t.Fatalf("read report for test seal: %v", err)
	}
	manifestValue, manifest, err := service.readDurableManifest(runID)
	if err != nil {
		t.Fatalf("read manifest for test seal: %v", err)
	}
	reportDigest, reportSize := digestAndSize(report)
	manifestArtifactDigest, _ := digestAndSize(manifest)
	_ = os.Remove(filepath.Join(service.store.runsRoot, runID, reportAnchorFileName))
	if err := service.store.writeReportAnchor(runID, reportAnchor{
		SchemaVersion: SchemaVersion, AttestationRevision: reportValue.AttestationRevision,
		RunID: runID, ReportDigest: reportDigest, ReportSize: reportSize,
		ManifestSemanticDigest: manifestValue.ManifestDigest,
		ManifestArtifactDigest: manifestArtifactDigest,
		PrivateReceiptDigest:   privateDigest, EvidenceFiles: evidenceFiles, CreatedAt: now,
	}); err != nil {
		t.Fatalf("seal test report: %v", err)
	}
}

func writeAnchoredTestReport(t *testing.T, service *Service, runID string, report Report) {
	t.Helper()
	if err := service.store.WriteReport(runID, workerReportFromReport(report)); err != nil {
		t.Fatalf("write test report: %v", err)
	}
	sealTestReport(t, service, runID)
}

func artifactForBytes(id, name, mediaType string, content []byte) Artifact {
	kind := strings.TrimPrefix(filepath.Ext(name), ".")
	if contract, ok := publicArtifactContracts[name]; ok {
		kind = contract.Kind
		mediaType = contract.MediaType
	}
	return Artifact{
		ID: id, Name: name, Kind: kind, URI: name,
		Digest: digestBytes(content), MediaType: mediaType, SizeBytes: int64(len(content)),
	}
}

func ensureTestPublicReceipt(t *testing.T, service *Service, runID string, report *Report) {
	t.Helper()
	if _, present := findArtifactByName(*report, publicChecksumArtifactName); present {
		return
	}
	artifacts := reportArtifacts(*report)
	if len(artifacts) == 0 {
		metrics, err := json.Marshal(reportMetricEvidenceFile{SchemaVersion: SchemaVersion, Metrics: report.Metrics})
		if err != nil {
			t.Fatalf("encode public metric fixture: %v", err)
		}
		metrics = append(metrics, '\n')
		if err := os.WriteFile(filepath.Join(service.store.runsRoot, runID, "metrics.json"), metrics, 0o600); err != nil {
			t.Fatalf("write public metric fixture: %v", err)
		}
		report.Artifacts = append(report.Artifacts, artifactForBytes("metrics", "metrics.json", "application/json", metrics))
		artifacts = reportArtifacts(*report)
	}
	var receipt strings.Builder
	for _, artifact := range artifacts {
		receipt.WriteString(strings.TrimPrefix(artifact.Digest, "sha256:"))
		receipt.WriteString("  ")
		receipt.WriteString(artifact.Name)
		receipt.WriteByte('\n')
	}
	receiptBytes := []byte(receipt.String())
	if err := os.WriteFile(
		filepath.Join(service.store.runsRoot, runID, publicChecksumArtifactName), receiptBytes, 0o600,
	); err != nil {
		t.Fatalf("write public checksum fixture: %v", err)
	}
	report.Artifacts = append(report.Artifacts, artifactForBytes(
		"checksums", publicChecksumArtifactName, "text/plain", receiptBytes,
	))
}

func prepareOpenableArtifact(t *testing.T, service *Service, root string) (Run, []byte) {
	t.Helper()
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	content := []byte("{\"routing.accuracy\":1}\n")
	if err := os.WriteFile(filepath.Join(root, "runs", run.ID, "metrics.json"), content, 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, []Artifact{
		artifactForBytes("metrics", "metrics.json", "application/json", content),
	})
	return run, content
}

func writeReportWithPublicReceipt(t *testing.T, service *Service, run Run, artifacts []Artifact) []Artifact {
	t.Helper()
	report := reportForRun(run, append([]Artifact(nil), artifacts...))
	ensureTestPublicReceipt(t, service, run.ID, &report)
	if err := service.store.WriteReport(run.ID, workerReportFromReport(report)); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	sealTestReport(t, service, run.ID)
	return report.Artifacts
}

func TestArtifactDownloadRequiresReportAndCanonicalAllowlist(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	content := []byte("{\"routing.accuracy\":1}\n")
	if err := os.WriteFile(filepath.Join(runDir, "metrics.json"), content, 0o600); err != nil {
		t.Fatalf("write artifact: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, []Artifact{
		artifactForBytes("metrics", "metrics.json", "application/json", content),
	})
	opened, openErr := service.OpenArtifactAs(SystemActor(), run.ID, "metrics")
	if openErr != nil {
		t.Fatalf("OpenArtifact: %v", openErr)
	}
	t.Cleanup(func() {
		if err := opened.File.Close(); err != nil {
			t.Errorf("close opened artifact: %v", err)
		}
	})
	got, readErr := io.ReadAll(opened.File)
	if readErr != nil || string(got) != string(content) {
		t.Fatalf("artifact bytes=%q err=%v", got, readErr)
	}
	if opened.MediaType != "application/json" {
		t.Fatalf("artifact media type=%q, want application/json", opened.MediaType)
	}
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "not-in-report"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unlisted artifact error=%v, want ErrNotFound", err)
	}

	if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, []Artifact{{
		ID: "manifest", Name: manifestFileName, Kind: "json", URI: manifestFileName,
	}}))); err != nil {
		t.Fatalf("WriteReport manifest: %v", err)
	}
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "manifest"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("protected manifest error=%v, want ErrInvalid", err)
	}

	if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, []Artifact{{
		ID: "traversal", Name: "../metrics.json", Kind: "json", URI: "../metrics.json",
		Digest: digestBytes(content), SizeBytes: int64(len(content)),
	}}))); err != nil {
		t.Fatalf("WriteReport traversal: %v", err)
	}
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "traversal"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("traversal error=%v, want ErrInvalid", err)
	}
	if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, []Artifact{{
		ID: "normalized-traversal", Name: "nested/../metrics.json", Kind: "json", URI: "nested/../metrics.json",
		Digest: digestBytes(content), SizeBytes: int64(len(content)),
	}}))); err != nil {
		t.Fatalf("WriteReport normalized traversal: %v", err)
	}
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "normalized-traversal"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("normalized traversal error=%v, want ErrInvalid", err)
	}
}

func TestArtifactDownloadVerifiesReportAndPublicChecksumEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(t *testing.T, runDir string, artifacts []Artifact) []Artifact
	}{
		{
			name: "artifact bytes",
			mutate: func(t *testing.T, runDir string, artifacts []Artifact) []Artifact {
				t.Helper()
				if err := os.WriteFile(filepath.Join(runDir, "metrics.json"), []byte("tampered\n"), 0o600); err != nil {
					t.Fatalf("tamper artifact: %v", err)
				}
				return artifacts
			},
		},
		{
			name: "reported size",
			mutate: func(_ *testing.T, _ string, artifacts []Artifact) []Artifact {
				artifacts[0].SizeBytes++
				return artifacts
			},
		},
		{
			name: "reported digest",
			mutate: func(_ *testing.T, _ string, artifacts []Artifact) []Artifact {
				artifacts[0].Digest = digestString("different artifact")
				return artifacts
			},
		},
		{
			name: "checksum bytes",
			mutate: func(t *testing.T, runDir string, artifacts []Artifact) []Artifact {
				t.Helper()
				if err := os.WriteFile(filepath.Join(runDir, publicChecksumArtifactName), []byte("tampered\n"), 0o600); err != nil {
					t.Fatalf("tamper checksum receipt: %v", err)
				}
				return artifacts
			},
		},
		{
			name: "checksum entry",
			mutate: func(t *testing.T, runDir string, artifacts []Artifact) []Artifact {
				t.Helper()
				badReceipt := []byte(strings.Repeat("0", 64) + "  metrics.json\n")
				if err := os.WriteFile(filepath.Join(runDir, publicChecksumArtifactName), badReceipt, 0o600); err != nil {
					t.Fatalf("replace checksum receipt: %v", err)
				}
				artifacts[1] = artifactForBytes("checksums", publicChecksumArtifactName, "text/plain", badReceipt)
				return artifacts
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			service, root := newTestService(t, &controlledProcess{}, 1)
			run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
			if err != nil {
				t.Fatalf("CreateRun: %v", err)
			}
			runDir := filepath.Join(root, "runs", run.ID)
			content := []byte("{\"routing.accuracy\":1}\n")
			if err := os.WriteFile(filepath.Join(runDir, "metrics.json"), content, 0o600); err != nil {
				t.Fatalf("write artifact: %v", err)
			}
			artifacts := writeReportWithPublicReceipt(t, service, run, []Artifact{
				artifactForBytes("metrics", "metrics.json", "application/json", content),
			})
			artifacts = test.mutate(t, runDir, artifacts)
			if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, artifacts))); err != nil {
				t.Fatalf("rewrite report: %v", err)
			}
			if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "metrics"); !errors.Is(err, ErrInvalid) {
				t.Fatalf("OpenArtifact error=%v, want ErrInvalid", err)
			}
		})
	}
}

func TestArtifactSensitivityAllowlistExcludesCasesGradingAndConnectivity(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if createErr != nil {
		t.Fatalf("CreateRun: %v", createErr)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	traceBytes := []byte(`{"schema_version":"evaluation.v1","case_id":"case-1","plugins":[],"recommended_models":[],"traces":[],"signals":[],"applied_unknown_policies":[]}` + "\n")
	caseBytes := []byte(`{"schema_version":"evaluation.v1","id":"case-1","track_ids":["routing"],"messages":[{"role":"user","content":"test"}],"modality":"text","tags":[]}` + "\n")
	artifacts := []Artifact{
		artifactForBytes("traces", "routing-traces.jsonl", "application/x-ndjson", traceBytes),
		artifactForBytes("capacity", "capacity-profile.json", "application/json", []byte("{}\n")),
		artifactForBytes("summary", "failure-summary.json", "application/json", []byte("{}\n")),
		artifactForBytes("records", "records.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("cases", "cases.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("lineage", "lineage.json", "application/json", []byte("{}\n")),
		artifactForBytes("private-checksums", "private-checksums.sha256", "text/plain", []byte("{}\n")),
	}
	for _, artifact := range artifacts {
		data := []byte("{}\n")
		switch artifact.URI {
		case "routing-traces.jsonl":
			data = traceBytes
		case "cases.jsonl":
			data = caseBytes
		}
		if err := os.WriteFile(filepath.Join(runDir, artifact.URI), data, 0o600); err != nil {
			t.Fatalf("write %s: %v", artifact.URI, err)
		}
	}
	public := writeReportWithPublicReceipt(t, service, run, artifacts[1:3])
	_ = public
	for _, id := range []string{"capacity", "summary", "checksums"} {
		opened, err := service.OpenArtifactAs(SystemActor(), run.ID, id)
		if err != nil {
			t.Fatalf("safe artifact %s rejected: %v", id, err)
		}
		_ = opened.File.Close()
	}
	for _, id := range []string{"traces", "records", "cases", "lineage", "private-checksums"} {
		if _, err := service.OpenArtifactAs(SystemActor(), run.ID, id); !errors.Is(err, ErrNotFound) {
			t.Fatalf("sensitive artifact %s error=%v, want ErrNotFound", id, err)
		}
	}
}

func TestArtifactAndReportRejectSymlinks(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	outside := filepath.Join(root, "outside.jsonl")
	if err := os.WriteFile(outside, []byte("private\n"), 0o600); err != nil {
		t.Fatalf("write outside file: %v", err)
	}
	if err := os.Symlink(outside, filepath.Join(runDir, "metrics.json")); err != nil {
		t.Fatalf("symlink artifact: %v", err)
	}
	if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, []Artifact{{
		ID: "metrics", Name: "metrics.json", Kind: "json", URI: "metrics.json",
		Digest: digestBytes([]byte("private\n")), SizeBytes: int64(len("private\n")),
	}}))); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	if _, err := service.store.OpenArtifact(run.ID, "metrics.json"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("symlink artifact error=%v, want ErrInvalid", err)
	}

	reportPath := filepath.Join(runDir, reportFileName)
	if err := os.Remove(reportPath); err != nil {
		t.Fatalf("remove report: %v", err)
	}
	outsideReport := filepath.Join(root, "outside-report.json")
	if err := writeJSONAtomic(outsideReport, reportForRun(run, nil)); err != nil {
		t.Fatalf("write outside report: %v", err)
	}
	if err := os.Symlink(outsideReport, reportPath); err != nil {
		t.Fatalf("symlink report: %v", err)
	}
	run = completeTestRun(t, service, run)
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); err == nil {
		t.Fatal("symlink report was accepted")
	}
}

func TestArtifactDownloadRejectsInBundleSymlinkAndSharedPermissions(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	target := filepath.Join(runDir, "cases.jsonl")
	if err := os.WriteFile(target, []byte("private\n"), 0o600); err != nil {
		t.Fatalf("write artifact target: %v", err)
	}
	linked := filepath.Join(runDir, "metrics.json")
	if err := os.Symlink("cases.jsonl", linked); err != nil {
		t.Fatalf("symlink in-bundle artifact: %v", err)
	}
	if err := service.store.WriteReport(run.ID, workerReportFromReport(reportForRun(run, []Artifact{{
		ID: "metrics", Name: "metrics.json", Kind: "json", URI: "metrics.json",
		Digest: digestBytes([]byte("private\n")), SizeBytes: int64(len("private\n")),
	}}))); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	if _, err := service.store.OpenArtifact(run.ID, "metrics.json"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("in-bundle symlink error=%v, want ErrInvalid", err)
	}

	if err := os.Remove(linked); err != nil {
		t.Fatalf("remove symlink: %v", err)
	}
	if err := os.WriteFile(linked, []byte("shared\n"), 0o640); err != nil {
		t.Fatalf("write shared artifact: %v", err)
	}
	if _, err := service.store.OpenArtifact(run.ID, "metrics.json"); err == nil {
		t.Fatal("group-readable artifact was accepted")
	}
}

func TestCompletedReportRejectsPostSealPrivateEvidenceMutation(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	records := []byte("{\"case_id\":\"case-1\"}\n")
	if err := os.WriteFile(filepath.Join(runDir, "records.jsonl"), records, 0o600); err != nil {
		t.Fatalf("write records: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, nil)
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("read sealed report: %v", err)
	}
	tampered := append([]byte(nil), records...)
	tampered[len(tampered)-2] = '2'
	if err := os.WriteFile(filepath.Join(runDir, "records.jsonl"), tampered, 0o600); err != nil {
		t.Fatalf("tamper records: %v", err)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("post-seal private evidence mutation error=%v, want ErrInvalid", err)
	}
}

func TestCompletedReportRejectsMutationWithRestoredSizeAndMtime(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	recordsPath := filepath.Join(root, "runs", run.ID, "records.jsonl")
	records := []byte("{\"case_id\":\"case-1\"}\n")
	if writeErr := os.WriteFile(recordsPath, records, 0o600); writeErr != nil {
		t.Fatalf("write records: %v", writeErr)
	}
	writeReportWithPublicReceipt(t, service, run, nil)
	info, statErr := os.Stat(recordsPath)
	if statErr != nil {
		t.Fatalf("stat sealed records: %v", statErr)
	}
	tampered := append([]byte(nil), records...)
	tampered[len(tampered)-2] = '2'
	if err := os.WriteFile(recordsPath, tampered, 0o600); err != nil {
		t.Fatalf("tamper records: %v", err)
	}
	if err := os.Chtimes(recordsPath, info.ModTime(), info.ModTime()); err != nil {
		t.Fatalf("restore records mtime: %v", err)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("metadata-preserving evidence mutation error=%v, want ErrInvalid", err)
	}
}

func TestCompletedReportSurvivesPrivatePermissionRepair(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	recordsPath := filepath.Join(runDir, "records.jsonl")
	if err := os.WriteFile(recordsPath, []byte("{\"case_id\":\"case-1\"}\n"), 0o600); err != nil {
		t.Fatalf("write records: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, nil)
	if err := os.Chmod(recordsPath, 0o600); err != nil {
		t.Fatalf("repair private permissions: %v", err)
	}
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("read report after private permission repair: %v", err)
	}
}

func TestReportAndArtifactReadsAreGloballyBounded(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRunAs(context.Background(), SystemActor(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	content := []byte("{}\n")
	if err := os.WriteFile(filepath.Join(runDir, "metrics.json"), content, 0o600); err != nil {
		t.Fatalf("write metrics: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, []Artifact{
		artifactForBytes("metrics", "metrics.json", "application/json", content),
	})
	for index := 0; index < cap(service.activity.evidenceReads); index++ {
		service.activity.evidenceReads <- struct{}{}
	}
	defer func() {
		for len(service.activity.evidenceReads) > 0 {
			<-service.activity.evidenceReads
		}
	}()
	if _, err := service.ReportJSONAs(SystemActor(), run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("saturated report read error=%v, want ErrConflict", err)
	}
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "metrics"); !errors.Is(err, ErrConflict) {
		t.Fatalf("saturated artifact read error=%v, want ErrConflict", err)
	}
}

func TestArtifactVerificationReleasesEvidenceCapacityBeforeStreaming(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, _ := prepareOpenableArtifact(t, service, root)
	opened := make([]*OpenedArtifact, 0, cap(service.activity.evidenceReads)+1)
	for index := 0; index <= cap(service.activity.evidenceReads); index++ {
		artifact, err := service.OpenArtifactAs(SystemActor(), run.ID, "metrics")
		if err != nil {
			t.Fatalf("open verified artifact %d: %v", index, err)
		}
		t.Cleanup(func() { _ = artifact.File.Close() })
		opened = append(opened, artifact)
	}
	if got := len(service.activity.evidenceReads); got != 0 {
		t.Fatalf("verified artifact streams retained %d evidence leases", got)
	}
	if !service.operationMu.TryLock() {
		t.Fatal("verified artifact stream retained its Service operation lease")
	}
	service.operationMu.Unlock()
	for _, artifact := range opened {
		if err := artifact.File.Close(); err != nil {
			t.Fatalf("close verified artifact: %v", err)
		}
	}
}

func TestArtifactOpenFailureReleasesAllLeases(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, _ := prepareOpenableArtifact(t, service, root)
	if _, err := service.OpenArtifactAs(SystemActor(), run.ID, "missing-artifact"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("missing artifact error=%v, want ErrNotFound", err)
	}
	if got := len(service.activity.evidenceReads); got != 0 {
		t.Fatalf("failed artifact open leaked %d evidence read leases", got)
	}
	if !service.operationMu.TryLock() {
		t.Fatal("failed artifact open leaked its service operation lease")
	}
	service.operationMu.Unlock()
}

func TestVerifiedArtifactDescriptorSurvivesConcurrentDeletion(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, _ := prepareOpenableArtifact(t, service, root)
	opened, openErr := service.OpenArtifactAs(SystemActor(), run.ID, "metrics")
	if openErr != nil {
		t.Fatalf("open artifact: %v", openErr)
	}
	t.Cleanup(func() { _ = opened.File.Close() })
	if err := service.DeleteRunAs(SystemActor(), run.ID); err != nil {
		t.Fatalf("delete run while verified descriptor is open: %v", err)
	}
	data, readErr := io.ReadAll(opened.File)
	if readErr != nil || string(data) != "{\"routing.accuracy\":1}\n" {
		t.Fatalf("pinned artifact after deletion data=%q err=%v", data, readErr)
	}
}

func TestVerifiedArtifactDescriptorDoesNotBlockServiceClose(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, _ := prepareOpenableArtifact(t, service, root)
	opened, openErr := service.OpenArtifactAs(SystemActor(), run.ID, "metrics")
	if openErr != nil {
		t.Fatalf("open artifact: %v", openErr)
	}
	t.Cleanup(func() { _ = opened.File.Close() })
	if err := service.Close(); err != nil {
		t.Fatalf("close service while verified descriptor is open: %v", err)
	}
	data, readErr := io.ReadAll(opened.File)
	if readErr != nil || string(data) != "{\"routing.accuracy\":1}\n" {
		t.Fatalf("pinned artifact after Service close data=%q err=%v", data, readErr)
	}
}
