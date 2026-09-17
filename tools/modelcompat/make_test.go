package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestQualificationMakeRequiresCommittedSources(t *testing.T) {
	requireMakeFixtureTools(t)
	repo := t.TempDir()
	runGit := func(args ...string) {
		t.Helper()
		command := exec.Command("git", args...)
		command.Dir = repo
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, output)
		}
	}
	runGit("init", "-q")
	data, err := os.ReadFile("../../tools/make/models.mk")
	if err != nil {
		t.Fatal(err)
	}
	data = append(data, []byte("\nrust-ci:\n\t@:\n")...)
	writeMakeFixtureFile(t, filepath.Join(repo, "Makefile"), data)
	runGit("add", "Makefile")
	runGit("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
		"commit", "-q", "-s", "-m", "Test fixture")
	runMake := func(args ...string) ([]byte, error) {
		command := exec.Command("make", args...)
		command.Dir = repo
		return command.CombinedOutput()
	}
	if output, err := runMake("check-candle-qualification-source"); err != nil {
		t.Fatalf("clean source rejected: %v\n%s", err, output)
	}
	for target, command := range map[string]string{
		"qualify-candle-cpu": "go run ../../tools/modelcompat/main.go qualify-candle-cpu",
		"test":               "go test -race -count=1 -v ../../tools/modelcompat/*.go",
	} {
		if output, err := runMake("--dry-run", target); err != nil || !strings.Contains(string(output), command) {
			t.Fatalf("make %s must run the tools entrypoint: %v\n%s", target, err, output)
		}
	}
	writeMakeFixtureFile(t, filepath.Join(repo, "Makefile"), append(data, []byte("\n# dirty\n")...))
	if output, err := runMake("qualify-candle-cpu"); err == nil || !strings.Contains(string(output), "tracked changes found") {
		t.Fatalf("dirty source must fail before building: %v\n%s", err, output)
	}
	writeMakeFixtureFile(t, filepath.Join(repo, "Makefile"), data)
	for _, source := range []string{"src/semantic-router", "candle-binding", "tools/modelcompat"} {
		if err := os.MkdirAll(filepath.Join(repo, source), 0o700); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(repo, source, "untracked-source")
		writeMakeFixtureFile(t, path, []byte("new source"))
		if output, err := runMake("qualify-candle-cpu"); err == nil || !strings.Contains(string(output), "untracked source files found") {
			t.Fatalf("untracked source must fail before building: %v\n%s", err, output)
		}
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}
}

func writeMakeFixtureFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func requireMakeFixtureTools(t *testing.T) {
	t.Helper()
	for _, name := range []string{"git", "make"} {
		if _, err := exec.LookPath(name); err != nil {
			t.Skipf("%s is required: %v", name, err)
		}
	}
}
