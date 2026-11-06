package config

import (
	"regexp"
	"testing"
)

var deferredModelPath = regexp.MustCompile(`model "([^"]+)"`)

// effectiveModel is the model that answers consumer: its deployment's
// artifact, or the local path a deprecated model field names.
func effectiveModel(t *testing.T, cfg *RouterConfig, consumer string) string {
	t.Helper()
	resolved, ok, err := cfg.ResolveTask(consumer)
	if IsDeferredModelError(err) {
		if match := deferredModelPath.FindStringSubmatch(err.Error()); match != nil {
			return match[1]
		}
	}
	if err != nil || !ok {
		t.Fatalf("no deployment answers %s: %v", consumer, err)
	}
	if path := resolved.Deployment.LegacyModelPath(); path != "" && RegisteredModel(path) == nil {
		return path
	}
	return resolved.Deployment.Artifact
}

// sameModel reports whether two references name one model: they are equal
// or name the same registry repository.
func sameModel(a, b string) bool {
	if a == b {
		return true
	}
	left, right := RegisteredModel(a), RegisteredModel(b)
	return left != nil && right != nil && left.RepoID == right.RepoID
}
