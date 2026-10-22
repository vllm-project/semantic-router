package config

import (
	"strings"
	"testing"
)

func TestRegistrySourceRevisionIsExactAndUnambiguous(t *testing.T) {
	revision := strings.Repeat("a", 40)
	models := []ModelSpec{{RepoID: "owner/source-model", Revision: revision}}
	if got := registryInfoByRevision(models, revision); got == nil || got.RepoID != "owner/source-model" {
		t.Fatalf("registered source identity missing: %+v", got)
	}
	for _, unknown := range []string{"", "main", revision[:16], strings.Repeat("b", 40), strings.Repeat("z", 40)} {
		if got := registryInfoByRevision(models, unknown); got != nil {
			t.Fatalf("unverified revision %q identified a model: %+v", unknown, got)
		}
	}
	models = append(models, ModelSpec{RepoID: "another/copied-model", Revision: revision})
	if got := registryInfoByRevision(models, revision); got != nil {
		t.Fatalf("ambiguous revision identified a model: %+v", got)
	}
}
