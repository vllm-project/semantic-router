package evaluationplane

import (
	"fmt"
	"strings"
	"testing"
)

func TestResolvedLineageAcceptsOnlyCurrentVersionedEnvelope(t *testing.T) {
	resolved := fmt.Sprintf(
		`{"schema_version":%q,"manifest_digest":%q,"workload":{},"policy":{},"binding":{},"pool":{},"arms":[],"environment":{},"fixture_ref":null,"discovered_entrypoints":[],"executors":[]}`,
		SchemaVersion,
		"sha256:"+strings.Repeat("a", 64),
	)
	current := fmt.Sprintf(
		`{"schema_version":%q,"resolved_snapshot":%s,"normalized_suite_identities":null}`,
		SchemaVersion,
		resolved,
	)
	document, err := decodeLineageDocument([]byte(current))
	if err != nil {
		t.Fatalf("current lineage envelope rejected: %v", err)
	}
	if len(document.NormalizedSuiteIdentities) != 0 {
		t.Fatal("nullable normalized suite identities were not decoded as absent")
	}
	retiredAlias := fmt.Sprintf(
		`{"schema_version":%q,"resolved_snapshot":%s,"normalized_suite_aliases":{}}`,
		SchemaVersion,
		resolved,
	)
	if _, err := decodeLineageDocument([]byte(retiredAlias)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("retired lineage alias wrapper error=%v", err)
	}
	legacy := resolved
	if _, err := decodeLineageDocument([]byte(legacy)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("legacy bare lineage error=%v", err)
	}
	missingIdentities := fmt.Sprintf(
		`{"schema_version":%q,"resolved_snapshot":%s}`,
		SchemaVersion,
		resolved,
	)
	if _, err := decodeLineageDocument([]byte(missingIdentities)); err == nil || !strings.Contains(err.Error(), "omits required field") {
		t.Fatalf("missing normalized identities field error=%v", err)
	}
	unknownVersion := fmt.Sprintf(
		`{"schema_version":"evaluation.v0","resolved_snapshot":%s,"normalized_suite_identities":null}`,
		resolved,
	)
	if _, err := decodeLineageDocument([]byte(unknownVersion)); err == nil || !strings.Contains(err.Error(), "version is invalid") {
		t.Fatalf("unknown lineage envelope version error=%v", err)
	}
}

func TestLineageExecutorsMustExactlyMatchManifest(t *testing.T) {
	manifest := RunManifest{
		Mode: ModeLive, SuiteIDs: []string{"live-mom-core"},
		SuiteExecutors: map[string]string{"live-mom-core": "live-runtime.v1"},
		TrackIDs:       []TrackID{"routing"},
	}
	valid := []lineageExecutor{{
		SchemaVersion: SchemaVersion, TrackID: "routing",
		ExecutorID: "live-runtime.v1", Mode: ModeLive,
	}}
	if err := validateLineageExecutors(manifest, valid); err != nil {
		t.Fatalf("valid lineage executor rejected: %v", err)
	}
	for _, mutate := range []func([]lineageExecutor) []lineageExecutor{
		func([]lineageExecutor) []lineageExecutor { return nil },
		func(values []lineageExecutor) []lineageExecutor { values[0].ExecutorID = "other.v1"; return values },
		func(values []lineageExecutor) []lineageExecutor { values[0].TrackID = "capacity"; return values },
		func(values []lineageExecutor) []lineageExecutor { values[0].Mode = ModeReplay; return values },
	} {
		candidate := append([]lineageExecutor(nil), valid...)
		if err := validateLineageExecutors(manifest, mutate(candidate)); err == nil {
			t.Fatal("lineage executor drift was accepted")
		}
	}
}
