package evaluationplane

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestNormalizedSamplingRetainsInstalledPerturbationCohort(t *testing.T) {
	root := t.TempDir()
	objectRoot := filepath.Join(root, "objects", "grading", "sha256")
	if err := os.MkdirAll(objectRoot, 0o700); err != nil {
		t.Fatal(err)
	}
	row := []byte(`{"schema_version":"evaluation-suite.v1","pair_id":"pair-1","source_case_id":"case-a","perturbed_case_id":"case-b","relation":"invariant","slice_ids":[],"native_pair_count":1,"source_record_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}` + "\n")
	digest := suiteDocumentDigest(row)
	if err := os.WriteFile(filepath.Join(objectRoot, strings.TrimPrefix(digest, "sha256:")), row, 0o600); err != nil {
		t.Fatal(err)
	}
	artifacts, err := json.Marshal(map[string]any{
		"perturbations": suiteArtifactReference{
			SchemaVersion: SchemaVersion, Digest: digest,
			SizeBytes: int64(len(row)), MediaType: "application/x-ndjson",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	document := installedSuiteDocument{Manifest: suiteManifestProjection{
		Revision: "sha256:" + strings.Repeat("b", 64), Artifacts: artifacts,
	}}
	plans := map[string]installedVisibleCasePlan{
		"case-a": {Modality: "text", TrackIDs: []TrackID{"routing"}},
		"case-b": {Modality: "text", TrackIDs: []TrackID{"routing"}},
		"case-c": {Modality: "text", TrackIDs: []TrackID{"routing"}},
	}
	for _, test := range []struct {
		limit int
		want  []string
	}{
		{limit: 1, want: []string{"case-a"}},
		{limit: 2, want: []string{"case-a", "case-b"}},
	} {
		selected, err := selectedInstalledNormalizedSourceCases(root, document, plans, RunManifest{
			TrackIDs: []TrackID{"routing"}, SampleLimit: test.limit, Seed: 19,
		})
		if err != nil {
			t.Fatalf("sample limit %d: %v", test.limit, err)
		}
		if !reflect.DeepEqual(selected, test.want) {
			t.Fatalf("sample limit %d selected=%v, want installed perturbation cohort %v", test.limit, selected, test.want)
		}
	}
}
