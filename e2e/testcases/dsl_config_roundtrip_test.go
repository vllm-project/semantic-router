package testcases

import (
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

func TestMaintainedConfigDecompilesToValidDSL(t *testing.T) {
	_, testFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test file path")
	}
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(testFile), "../.."))
	routerRoot := filepath.Join(repoRoot, "src", "semantic-router")
	binary := filepath.Join(t.TempDir(), "sr-dsl")

	build := exec.Command("go", "build", "-o", binary, "./cmd/dsl")
	build.Dir = routerRoot
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build sr-dsl: %v\n%s", err, output)
	}

	dslPath := filepath.Join(t.TempDir(), "config.dsl")
	decompile := exec.Command(
		binary,
		"decompile",
		"-o",
		dslPath,
		filepath.Join(repoRoot, "config", "config.yaml"),
	)
	if output, err := decompile.CombinedOutput(); err != nil {
		t.Fatalf("decompile maintained config: %v\n%s", err, output)
	}

	validate := exec.Command(binary, "validate", dslPath)
	if output, err := validate.CombinedOutput(); err != nil {
		t.Fatalf("validate decompiled DSL: %v\n%s", err, output)
	}
}
