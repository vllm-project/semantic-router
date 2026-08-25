package modeldownload

import (
	"slices"
	"testing"
)

func assertContainsAllModelSpecs(t *testing.T, specs []ModelSpec, wantPaths ...string) {
	t.Helper()
	gotPaths := make([]string, 0, len(specs))
	for _, spec := range specs {
		gotPaths = append(gotPaths, spec.LocalPath)
	}
	for _, want := range wantPaths {
		if !slices.Contains(gotPaths, want) {
			t.Fatalf("BuildModelSpecs() missing %q; got %v", want, gotPaths)
		}
	}
}
