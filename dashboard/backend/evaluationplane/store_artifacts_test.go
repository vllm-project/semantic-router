package evaluationplane

import (
	"context"
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
		SchemaVersion: SchemaVersion,
		Run:           run,
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
		if name == "events.jsonl" || name == privateChecksumArtifactName || name == reportFileName {
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
	run.Status = StatusCompleted
	if run.StartedAt == nil {
		run.StartedAt = &now
	}
	run.CompletedAt = &now
	if updateErr := service.store.UpdateRun(run); updateErr != nil {
		t.Fatalf("complete test run: %v", updateErr)
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
	_, manifest, err := service.readDurableManifest(runID)
	if err != nil {
		t.Fatalf("read manifest for test seal: %v", err)
	}
	reportDigest, reportSize := digestAndSize(report)
	manifestDigest, _ := digestAndSize(manifest)
	_ = os.Remove(filepath.Join(service.store.runsRoot, runID, reportAnchorFileName))
	if err := service.store.writeReportAnchor(runID, reportAnchor{
		SchemaVersion: SchemaVersion, RunID: runID, ReportDigest: reportDigest, ReportSize: reportSize,
		ManifestDigest: manifestDigest, PrivateReceiptDigest: privateDigest, EvidenceFiles: evidenceFiles, CreatedAt: now,
	}); err != nil {
		t.Fatalf("seal test report: %v", err)
	}
}

func writeAnchoredTestReport(t *testing.T, service *Service, runID string, report Report) {
	t.Helper()
	if err := service.store.WriteReport(runID, report); err != nil {
		t.Fatalf("write test report: %v", err)
	}
	sealTestReport(t, service, runID)
}

func artifactForBytes(id, name, mediaType string, content []byte) Artifact {
	return Artifact{
		ID: id, Name: name, Kind: filepath.Ext(name), URI: name,
		Digest: digestBytes(content), MediaType: mediaType, SizeBytes: int64(len(content)),
	}
}

func writeReportWithPublicReceipt(t *testing.T, service *Service, run Run, artifacts []Artifact) []Artifact {
	t.Helper()
	artifacts = append([]Artifact(nil), artifacts...)
	var receipt strings.Builder
	for _, artifact := range artifacts {
		receipt.WriteString(strings.TrimPrefix(artifact.Digest, "sha256:"))
		receipt.WriteString("  ")
		receipt.WriteString(artifact.Name)
		receipt.WriteByte('\n')
	}
	receiptBytes := []byte(receipt.String())
	runDir, err := service.store.checkedRunDir(run.ID)
	if err != nil {
		t.Fatalf("resolve run directory: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runDir, publicChecksumArtifactName), receiptBytes, 0o600); err != nil {
		t.Fatalf("write public checksum receipt: %v", err)
	}
	artifacts = append(artifacts, artifactForBytes(
		"checksums", publicChecksumArtifactName, "text/plain", receiptBytes,
	))
	if err := service.store.WriteReport(run.ID, reportForRun(run, artifacts)); err != nil {
		t.Fatalf("WriteReport: %v", err)
	}
	sealTestReport(t, service, run.ID)
	return artifacts
}

func TestArtifactDownloadRequiresReportAndCanonicalAllowlist(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, createErr := service.CreateRun(context.Background(), validCreateRequest())
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
	opened, openErr := service.OpenArtifact(run.ID, "metrics")
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
	if _, err := service.OpenArtifact(run.ID, "not-in-report"); !errors.Is(err, ErrNotFound) {
		t.Fatalf("unlisted artifact error=%v, want ErrNotFound", err)
	}

	if err := service.store.WriteReport(run.ID, reportForRun(run, []Artifact{{
		ID: "manifest", Name: manifestFileName, Kind: "json", URI: manifestFileName,
	}})); err != nil {
		t.Fatalf("WriteReport manifest: %v", err)
	}
	if _, err := service.OpenArtifact(run.ID, "manifest"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("protected manifest error=%v, want ErrInvalid", err)
	}

	if err := service.store.WriteReport(run.ID, reportForRun(run, []Artifact{{
		ID: "traversal", Name: "../metrics.json", Kind: "json", URI: "../metrics.json",
		Digest: digestBytes(content), SizeBytes: int64(len(content)),
	}})); err != nil {
		t.Fatalf("WriteReport traversal: %v", err)
	}
	if _, err := service.OpenArtifact(run.ID, "traversal"); !errors.Is(err, ErrInvalid) {
		t.Fatalf("traversal error=%v, want ErrInvalid", err)
	}
	if err := service.store.WriteReport(run.ID, reportForRun(run, []Artifact{{
		ID: "normalized-traversal", Name: "nested/../metrics.json", Kind: "json", URI: "nested/../metrics.json",
		Digest: digestBytes(content), SizeBytes: int64(len(content)),
	}})); err != nil {
		t.Fatalf("WriteReport normalized traversal: %v", err)
	}
	if _, err := service.OpenArtifact(run.ID, "normalized-traversal"); !errors.Is(err, ErrInvalid) {
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
			run, err := service.CreateRun(context.Background(), validCreateRequest())
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
			if err := service.store.WriteReport(run.ID, reportForRun(run, artifacts)); err != nil {
				t.Fatalf("rewrite report: %v", err)
			}
			if _, err := service.OpenArtifact(run.ID, "metrics"); !errors.Is(err, ErrInvalid) {
				t.Fatalf("OpenArtifact error=%v, want ErrInvalid", err)
			}
		})
	}
}

func TestArtifactSensitivityAllowlistExcludesCasesGradingAndConnectivity(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	artifacts := []Artifact{
		artifactForBytes("traces", "routing-traces.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("capacity", "capacity-profile.json", "application/json", []byte("{}\n")),
		artifactForBytes("summary", "failure-summary.json", "application/json", []byte("{}\n")),
		artifactForBytes("records", "records.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("failures", "failure-cases.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("cases", "cases.jsonl", "application/x-ndjson", []byte("{}\n")),
		artifactForBytes("lineage", "lineage.json", "application/json", []byte("{}\n")),
		artifactForBytes("private-checksums", "private-checksums.sha256", "text/plain", []byte("{}\n")),
	}
	for _, artifact := range artifacts {
		if err := os.WriteFile(filepath.Join(runDir, artifact.URI), []byte("{}\n"), 0o600); err != nil {
			t.Fatalf("write %s: %v", artifact.URI, err)
		}
	}
	public := writeReportWithPublicReceipt(t, service, run, artifacts[:3])
	_ = public
	for _, id := range []string{"traces", "capacity", "summary", "checksums"} {
		opened, err := service.OpenArtifact(run.ID, id)
		if err != nil {
			t.Fatalf("safe artifact %s rejected: %v", id, err)
		}
		_ = opened.File.Close()
	}
	for _, id := range []string{"records", "failures", "cases", "lineage", "private-checksums"} {
		if _, err := service.OpenArtifact(run.ID, id); !errors.Is(err, ErrNotFound) {
			t.Fatalf("sensitive artifact %s error=%v, want ErrNotFound", id, err)
		}
	}
}

func TestArtifactAndReportRejectSymlinks(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
	if err := service.store.WriteReport(run.ID, reportForRun(run, []Artifact{{
		ID: "metrics", Name: "metrics.json", Kind: "json", URI: "metrics.json",
		Digest: digestBytes([]byte("private\n")), SizeBytes: int64(len("private\n")),
	}})); err != nil {
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
	run.Status = StatusCompleted
	if err := service.store.UpdateRun(run); err != nil {
		t.Fatalf("complete run for report read: %v", err)
	}
	if _, err := service.ReportJSON(run.ID); err == nil {
		t.Fatal("symlink report was accepted")
	}
}

func TestArtifactDownloadRejectsInBundleSymlinkAndSharedPermissions(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
	if err := service.store.WriteReport(run.ID, reportForRun(run, []Artifact{{
		ID: "metrics", Name: "metrics.json", Kind: "json", URI: "metrics.json",
		Digest: digestBytes([]byte("private\n")), SizeBytes: int64(len("private\n")),
	}})); err != nil {
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
	run, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	runDir := filepath.Join(root, "runs", run.ID)
	records := []byte("{\"case_id\":\"case-1\"}\n")
	if err := os.WriteFile(filepath.Join(runDir, "records.jsonl"), records, 0o600); err != nil {
		t.Fatalf("write records: %v", err)
	}
	writeReportWithPublicReceipt(t, service, run, nil)
	if _, err := service.ReportJSON(run.ID); err != nil {
		t.Fatalf("read sealed report: %v", err)
	}
	tampered := append([]byte(nil), records...)
	tampered[len(tampered)-2] = '2'
	if err := os.WriteFile(filepath.Join(runDir, "records.jsonl"), tampered, 0o600); err != nil {
		t.Fatalf("tamper records: %v", err)
	}
	if _, err := service.ReportJSON(run.ID); !errors.Is(err, ErrInvalid) {
		t.Fatalf("post-seal private evidence mutation error=%v, want ErrInvalid", err)
	}
}

func TestCompletedReportSurvivesPrivatePermissionRepair(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
	if _, err := service.ReportJSON(run.ID); err != nil {
		t.Fatalf("read report after private permission repair: %v", err)
	}
}

func TestSealedEvidenceAcceptsLegacyMetadataVersionWhenDigestMatches(t *testing.T) {
	path := filepath.Join(t.TempDir(), "evidence.json")
	content := []byte("{\"sealed\":true}\n")
	if err := os.WriteFile(path, content, 0o600); err != nil {
		t.Fatalf("write evidence: %v", err)
	}
	sealed, err := sealEvidenceFile("run", "evidence.json", path, digestBytes(content), maxStructuredArtifactBytes)
	if err != nil {
		t.Fatalf("seal evidence: %v", err)
	}
	sealed.FileVersion = digestString("legacy-ctime-based-file-version")
	if err := verifyEvidenceFileMetadata(sealed, path); err != nil {
		t.Fatalf("verify legacy metadata version by content digest: %v", err)
	}
}

func TestReportAndArtifactReadsAreGloballyBounded(t *testing.T) {
	service, root := newTestService(t, &controlledProcess{}, 1)
	run, err := service.CreateRun(context.Background(), validCreateRequest())
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
	for index := 0; index < cap(service.evidenceReads); index++ {
		service.evidenceReads <- struct{}{}
	}
	defer func() {
		for len(service.evidenceReads) > 0 {
			<-service.evidenceReads
		}
	}()
	if _, err := service.ReportJSON(run.ID); !errors.Is(err, ErrConflict) {
		t.Fatalf("saturated report read error=%v, want ErrConflict", err)
	}
	if _, err := service.OpenArtifact(run.ID, "metrics"); !errors.Is(err, ErrConflict) {
		t.Fatalf("saturated artifact read error=%v, want ErrConflict", err)
	}
}
