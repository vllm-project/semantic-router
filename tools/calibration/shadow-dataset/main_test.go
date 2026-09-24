package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/shadowdataset"
)

func sha(text string) string {
	sum := sha256.Sum256([]byte(text))
	return hex.EncodeToString(sum[:])
}

// fixture writes a manifest, its judge tasks and one valid judgment set to a
// scratch directory, and returns their paths with the tasks it built.
func fixture(t *testing.T, seed string) (dir, manifestPath, tasksPath, judgmentsPath string, tasks shadowdataset.JudgeTaskSet) {
	t.Helper()
	record := store.Record{
		ID: "r1", RequestID: "req-r1", Recipe: "vault", Decision: "guard",
		Timestamp:     time.Date(2026, 9, 24, 9, 0, 0, 0, time.UTC),
		SelectedModel: "primary-model", RequestBody: "secret prompt", ResponseStatus: 200,
		LifecycleState: store.LifecycleCompleted,
		Outcomes: []store.Outcome{
			{Source: "primary_response", Verdict: "completed", Metadata: map[string]string{"response_sha256": sha("secret primary answer")}},
			{Source: "shadow_dispatch", Verdict: "completed", TargetRef: "candidate", Metadata: map[string]string{"response_sha256": sha("secret shadow answer")}},
		},
	}
	manifest, err := shadowdataset.Build([]store.Record{record},
		shadowdataset.Policy{Seed: seed, Splits: []shadowdataset.Split{{Name: "eval", Weight: 1}}})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	tasks, err = shadowdataset.BuildJudgeTasks(manifest, map[string]shadowdataset.ExampleText{
		manifest.Examples[0].ID: {Input: "secret prompt", Primary: "secret primary answer", Shadows: []string{"secret shadow answer"}},
	}, "key")
	if err != nil {
		t.Fatalf("BuildJudgeTasks: %v", err)
	}
	judgments := shadowdataset.JudgmentSet{
		Version:        shadowdataset.JudgmentsVersion,
		ManifestDigest: manifest.Digest,
		Judge:          shadowdataset.JudgeIdentity{Model: "judge", ModelRevision: "r1", RubricVersion: "v1"},
		Judgments: []shadowdataset.Judgment{
			{Task: tasks.Tasks[0].ID, Outcome: shadowdataset.JudgmentPreferred, Preferred: tasks.Tasks[0].First.Arm},
			{Task: tasks.Tasks[1].ID, Outcome: shadowdataset.JudgmentPreferred, Preferred: tasks.Tasks[1].First.Arm},
		},
	}

	dir = t.TempDir()
	write := func(name string, value any) string {
		path := filepath.Join(dir, name)
		body, marshalErr := json.Marshal(value)
		if marshalErr != nil {
			t.Fatal(marshalErr)
		}
		if writeErr := os.WriteFile(path, body, 0o644); writeErr != nil {
			t.Fatal(writeErr)
		}
		return path
	}
	return dir, write("manifest.json", manifest), write("tasks.json", tasks), write("judgments.json", judgments), tasks
}

func publishedTree(t *testing.T, root string) map[string]string {
	t.Helper()
	tree := map[string]string{}
	published := os.DirFS(root)
	err := fs.WalkDir(published, ".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil || entry.IsDir() {
			return err
		}
		body, readErr := fs.ReadFile(published, path)
		tree[path] = string(body)
		return readErr
	})
	if err != nil {
		t.Fatal(err)
	}
	return tree
}

// The manifest, the judgments and their report are published under names
// derived from content. The tasks are not, because they carry the prompt and
// both answers, and nothing published may.
func TestPublishWritesManifestJudgmentsAndReportButNoText(t *testing.T) {
	_, manifestPath, tasksPath, judgmentsPath, tasks := fixture(t, "seed")
	dest := t.TempDir()

	names, err := publish(DirectoryDestination{Root: dest}, manifestPath, tasksPath, judgmentsPath)
	if err != nil {
		t.Fatalf("publish: %v", err)
	}
	tree := publishedTree(t, dest)
	if len(names) != 3 || len(tree) != 3 {
		t.Fatalf("published %v, tree %v, want manifest, judgments and report", names, tree)
	}
	if names[0] != "manifests/"+tasks.ManifestDigest+".json" {
		t.Fatalf("manifest published as %q, want it named by its digest", names[0])
	}
	judgmentsName := names[1]
	if sha(tree[judgmentsName]) != strings.TrimSuffix(filepath.Base(judgmentsName), ".json") {
		t.Fatalf("judgments %q are not named by the digest of their bytes", judgmentsName)
	}
	if names[2] != strings.TrimSuffix(judgmentsName, ".json")+".report.json" {
		t.Fatalf("report %q is not named after its judgment set %q", names[2], judgmentsName)
	}
	for name, body := range tree {
		if strings.Contains(body, "secret") {
			t.Fatalf("%s carries captured text: %s", name, body)
		}
	}
	if !strings.Contains(tree[names[2]], `"slot_following_pairs": 1`) {
		t.Fatalf("report does not count the first-slot verdicts: %s", tree[names[2]])
	}
}

// Publishing the same inputs again is a no-op, and a name that already holds
// different bytes is never replaced.
func TestDirectoryDestinationNeverReplacesAPublishedFile(t *testing.T) {
	dest := DirectoryDestination{Root: t.TempDir()}
	if err := dest.Put("manifests/a.json", []byte("one")); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if err := dest.Put("manifests/a.json", []byte("one")); err != nil {
		t.Fatalf("republishing the same bytes failed: %v", err)
	}
	if err := dest.Put("manifests/a.json", []byte("two")); err == nil {
		t.Fatal("Put replaced a published file with different bytes")
	}
	body, err := os.ReadFile(filepath.Join(dest.Root, "manifests", "a.json"))
	if err != nil || string(body) != "one" {
		t.Fatalf("published file reads %q (%v), want the original", body, err)
	}
	entries, err := os.ReadDir(filepath.Join(dest.Root, "manifests"))
	if err != nil || len(entries) != 1 {
		t.Fatalf("manifests directory holds %v (%v), want no temporary files left", entries, err)
	}
}

// Every input is validated before anything is written, so a bad judgment set
// does not leave its manifest published on its own.
func TestPublishWritesNothingWhenAnInputFails(t *testing.T) {
	dir, manifestPath, tasksPath, judgmentsPath, _ := fixture(t, "seed")
	reasoned := filepath.Join(dir, "reasoned.json")
	body, err := os.ReadFile(judgmentsPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(reasoned, []byte(strings.Replace(string(body), `"outcome"`, `"reasoning":"longer","outcome"`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}
	edited := filepath.Join(dir, "edited.json")
	manifestBody, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(edited, []byte(strings.Replace(string(manifestBody), `"eval"`, `"train"`, 1)), 0o644); err != nil {
		t.Fatal(err)
	}

	_, otherManifest, _, _, _ := fixture(t, "another-seed")

	for name, inputs := range map[string][3]string{
		"judgment carrying reasoning": {manifestPath, tasksPath, reasoned},
		"edited manifest":             {edited, "", ""},
		"judgments without tasks":     {manifestPath, "", judgmentsPath},
		"tasks from another manifest": {otherManifest, tasksPath, judgmentsPath},
	} {
		t.Run(name, func(t *testing.T) {
			dest := t.TempDir()
			if _, err := publish(DirectoryDestination{Root: dest}, inputs[0], inputs[1], inputs[2]); err == nil {
				t.Fatal("publish accepted the inputs")
			}
			if tree := publishedTree(t, dest); len(tree) != 0 {
				t.Fatalf("publish wrote %v before failing", tree)
			}
		})
	}
}
