package dsl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMultiObjectiveConfigRoundTripsEntrypointsAndRecipes(t *testing.T) {
	original := loadMultiObjectiveConfig(t)
	source, err := Decompile(original)
	if err != nil {
		t.Fatalf("decompile multi-objective config: %v", err)
	}
	assertMultiObjectiveScopedDSL(t, source)

	recompiled, compileErrs := Compile(source)
	if len(compileErrs) > 0 {
		t.Fatalf("recompile multi-objective DSL: %v", compileErrs)
	}
	assertMultiObjectiveRoutingScopesEqual(t, original, recompiled)
}

func loadMultiObjectiveConfig(t *testing.T) *config.RouterConfig {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("resolve test path")
	}
	configPath := filepath.Join(
		filepath.Dir(filename),
		"..", "..", "..", "..",
		"config", "recipes", "multi-objective", "config.yaml",
	)
	data, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatalf("read multi-objective config: %v", err)
	}
	original, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("parse multi-objective config: %v", err)
	}
	return original
}

func assertMultiObjectiveScopedDSL(t *testing.T, source string) {
	t.Helper()
	if got := strings.Count(source, "ENTRYPOINT {"); got != 5 {
		t.Fatalf("decompiled entrypoints = %d, want 5", got)
	}
	if got := strings.Count(source, "\nRECIPE "); got != 5 {
		t.Fatalf("decompiled recipes = %d, want 5", got)
	}
	for _, expected := range []string{
		`mode: "insert"`,
		`max_completion_tokens: 2048`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("decompiled multi-objective DSL missing %q", expected)
		}
	}
}

func assertMultiObjectiveRoutingScopesEqual(
	t *testing.T,
	original, recompiled *config.RouterConfig,
) {
	t.Helper()
	if !reflect.DeepEqual(original.Entrypoints, recompiled.Entrypoints) {
		t.Fatalf("entrypoints changed after YAML -> DSL -> config round trip")
	}
	recipeDiff := cmp.Diff(
		original.Recipes,
		recompiled.Recipes,
		cmp.Comparer(func(left, right config.StructuredPayload) bool {
			var leftValue, rightValue interface{}
			if json.Unmarshal(left.Raw, &leftValue) != nil ||
				json.Unmarshal(right.Raw, &rightValue) != nil {
				return false
			}
			return reflect.DeepEqual(leftValue, rightValue)
		}),
		cmp.FilterPath(func(path cmp.Path) bool {
			field, ok := path.Last().(cmp.StructField)
			return ok && field.Name() == "OnError"
		}, cmp.Ignore()),
	)
	if recipeDiff != "" {
		t.Fatalf(
			"recipes changed after YAML -> DSL -> config round trip (-want +got):\n%s",
			recipeDiff,
		)
	}
}
