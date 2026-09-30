package testcases

import (
	"encoding/base64"
	"testing"
)

func TestWorkflowRedisRecipeNamespaceMatchesProductionEncoding(t *testing.T) {
	t.Parallel()

	wantDefault := base64.RawURLEncoding.EncodeToString([]byte("default"))
	if got := workflowRedisRecipeNamespace("default"); got != wantDefault {
		t.Fatalf("default namespace = %q, want %q", got, wantDefault)
	}
	if got := workflowRedisRecipeNamespace(""); got != wantDefault {
		t.Fatalf("empty namespace = %q, want %q", got, wantDefault)
	}
	if got := workflowRedisRecipeNamespace("default"); got == "default" {
		t.Fatal("default recipe was stored without the injective namespace encoding")
	}

	slash := workflowRedisRecipeNamespace("tenant/a")
	underscore := workflowRedisRecipeNamespace("tenant_a")
	if slash == underscore {
		t.Fatalf("namespace encoding collapsed tenant/a onto tenant_a: %q", slash)
	}
	if slash != base64.RawURLEncoding.EncodeToString([]byte("tenant/a")) {
		t.Fatalf("tenant/a namespace = %q", slash)
	}
}
