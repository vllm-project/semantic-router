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
	"github.com/google/go-cmp/cmp/cmpopts"

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
	return parseMaintainedConfigBytes(t, "multi-objective config", data)
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
		`mode: "dynamic"`,
		`template: "micro_agent"`,
	} {
		if !strings.Contains(source, expected) {
			t.Fatalf("decompiled multi-objective DSL missing %q", expected)
		}
	}
	if strings.Contains(source, `planner: { model:`) {
		t.Fatalf("Recipe DSL leaked physical planner Model selection:\n%s", source)
	}
}

func assertMultiObjectiveRoutingScopesEqual(
	t *testing.T,
	original, recompiled *config.RouterConfig,
) {
	t.Helper()
	originalEntrypoints := persistedEntrypointContracts(original.Entrypoints)
	recompiledEntrypoints := persistedEntrypointContracts(recompiled.Entrypoints)
	if !reflect.DeepEqual(originalEntrypoints, recompiledEntrypoints) {
		t.Fatalf("entrypoints changed after YAML -> DSL -> config round trip:\n%s", cmp.Diff(originalEntrypoints, recompiledEntrypoints))
	}
	recipeDiff := cmp.Diff(
		original.Recipes,
		recompiled.Recipes,
		cmpopts.IgnoreUnexported(config.RoutingRecipe{}),
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

// EntrypointMapping also carries a prepared, runtime-only derived recipe view.
// Round-trip equality must cover the complete persisted contract without
// treating that immutable cache (whose profile may contain normalized runtime
// defaults) as serialized DSL state.
type persistedEntrypointContract struct {
	ID         string
	Revision   int64
	Name       string
	ModelNames []string
	Rules      []persistedEntrypointRule
}

type persistedEntrypointRule struct {
	ID      string
	Name    string
	Matches []config.EntrypointMatch
	Action  config.EntrypointRuleAction
}

func persistedEntrypointContracts(entrypoints []config.EntrypointMapping) []persistedEntrypointContract {
	result := make([]persistedEntrypointContract, 0, len(entrypoints))
	for _, entrypoint := range entrypoints {
		rules := make([]persistedEntrypointRule, 0, len(entrypoint.Rules))
		for _, rule := range entrypoint.Rules {
			rules = append(rules, persistedEntrypointRule{ID: rule.ID, Name: rule.Name, Matches: rule.Matches, Action: rule.Action})
		}
		result = append(result, persistedEntrypointContract{
			ID:         entrypoint.ID,
			Revision:   entrypoint.Revision,
			Name:       entrypoint.Name,
			ModelNames: entrypoint.ModelNames,
			Rules:      rules,
		})
	}
	return result
}
