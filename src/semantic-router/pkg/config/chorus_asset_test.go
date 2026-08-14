package config

import (
	"slices"
	"testing"
)

const chorusAsset = "config/recipes/built-in/latest/chorus-v1/config.yaml"

func TestChorusRuntimeContract(t *testing.T) {
	cfg, err := ParseYAMLBytes(mustReadRepoFile(t, chorusAsset))
	if err != nil {
		t.Fatalf("parse vLLM-SR Chorus recipe: %v", err)
	}

	if got := cfg.Looper.Endpoint; got != "http://vllm-sr-envoy-container:8899/v1/chat/completions" {
		t.Fatalf("Looper endpoint must re-enter Envoy, got %q", got)
	}
	if len(cfg.Entrypoints) != 5 {
		t.Fatalf("entrypoint count = %d, want 5", len(cfg.Entrypoints))
	}

	decisionCount := 0
	for _, recipeName := range []string{"balance", "speed", "cost", "accuracy", "vault"} {
		recipe, ok := cfg.RecipeByName(RecipeName(recipeName))
		if !ok {
			t.Fatalf("missing recipe %q", recipeName)
		}
		decisionCount += len(recipe.Profile.Decisions)
	}
	if decisionCount != 43 {
		t.Fatalf("decision count = %d, want 43", decisionCount)
	}

	assertChorusMistralReasoning(t, cfg)
	assertChorusOrchestrationBudgets(t, cfg)
	assertChorusReplayBoundary(t, cfg)
	assertChorusManagementBoundary(t, cfg)
}

func assertChorusManagementBoundary(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	assertChorusManagementListener(t, cfg.ManagementAPI)
	assertChorusDashboardToken(t, cfg.ManagementAPI.Auth.Tokens)
	assertChorusDashboardPermissions(t, cfg.ManagementAPI.Auth.Roles["dashboard_control_plane"])
	if cfg.Observability.Tracing.Enabled {
		t.Fatal("vLLM-SR Chorus must not emit traces to an undeclared collector")
	}
}

func assertChorusManagementListener(t *testing.T, management ManagementAPIConfig) {
	t.Helper()
	if management.BindAddress != "0.0.0.0" || management.Port != 8080 || !management.RemoteExposure ||
		management.Auth.Mode != ManagementAuthModeBearer {
		t.Fatalf("vLLM-SR Chorus management API is not remotely authenticated: %+v", management)
	}
}

func assertChorusDashboardToken(t *testing.T, tokens []ManagementAPITokenRef) {
	t.Helper()
	foundToken := false
	for _, token := range tokens {
		if token.Env == "VLLM_SR_DASHBOARD_RECIPE_TOKEN" && token.Role == "dashboard_control_plane" {
			foundToken = true
		}
	}
	if !foundToken {
		t.Fatalf("vLLM-SR Chorus management API is missing its Dashboard token reference: %+v", tokens)
	}
}

func assertChorusDashboardPermissions(t *testing.T, permissions []string) {
	t.Helper()
	expectedPermissions := []string{
		"cache.invalidate",
		"cache.manage",
		"cache.read",
		"classify.invoke",
		"compression.manage",
		"compression.preview",
		"compression.read",
		"config.read",
		"config.write",
		"learning.ingest",
		"ready.read",
		"replay.detail",
		"replay.read",
	}
	if !slices.Equal(permissions, expectedPermissions) {
		t.Fatalf("vLLM-SR Chorus Dashboard role = %v, want %v", permissions, expectedPermissions)
	}
	for _, permission := range []string{"*", "secret_view", "data.write"} {
		if slices.Contains(permissions, permission) {
			t.Fatalf("vLLM-SR Chorus Dashboard role contains broad permission %q: %v", permission, permissions)
		}
	}
}

func assertChorusReplayBoundary(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	assertChorusReplayStore(t, cfg.RouterReplay)
	assertChorusVaultReplayDisabled(t, cfg)
	if replay := cfg.EffectiveRouterReplayConfigForDecision("balance_standard"); replay == nil || !replay.Enabled {
		t.Fatalf("ordinary vLLM-SR Chorus decisions must inherit enabled replay: %+v", replay)
	}
}

func assertChorusReplayStore(t *testing.T, replayConfig RouterReplayConfig) {
	t.Helper()
	if !replayConfig.Enabled || replayConfig.StoreBackend != "redis" || replayConfig.Redis == nil {
		t.Fatalf("vLLM-SR Chorus replay service = %+v, want enabled Redis", replayConfig)
	}
	if replayConfig.TTLSeconds != 604800 || replayConfig.AsyncWrites {
		t.Fatalf("vLLM-SR Chorus replay durability = ttl %d / async %v", replayConfig.TTLSeconds, replayConfig.AsyncWrites)
	}
	if replayConfig.Redis.Address != "redis:6379" || replayConfig.Redis.DB != 2 ||
		replayConfig.Redis.KeyPrefix != "chorus-v1:router-replay:" {
		t.Fatalf("vLLM-SR Chorus replay Redis isolation = %+v", replayConfig.Redis)
	}
}

func assertChorusVaultReplayDisabled(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	vault, ok := cfg.RecipeByName(RecipeName("vault"))
	if !ok {
		t.Fatal("missing Vault recipe")
	}
	if len(vault.Profile.Decisions) != 8 {
		t.Fatalf("Vault decision count = %d, want 8", len(vault.Profile.Decisions))
	}
	for index := range vault.Profile.Decisions {
		decision := &vault.Profile.Decisions[index]
		if replay := cfg.EffectiveRouterReplayConfig(decision); replay != nil {
			t.Fatalf("Vault decision %q unexpectedly records replay: %+v", decision.Name, replay)
		}
	}
}

func assertChorusMistralReasoning(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	family := cfg.ReasoningFamilies["mistral"]
	if family.Type != ReasoningFamilyTypeTopLevelReasoningEffort || family.Parameter != "reasoning_effort" {
		t.Fatalf("Mistral reasoning family = %+v", family)
	}
	if got := cfg.ModelConfig["local/mistral-small-4"].ReasoningFamily; got != "mistral" {
		t.Fatalf("Mistral model reasoning_family = %q, want mistral", got)
	}

	mistralRefs := 0
	for _, recipe := range cfg.Recipes {
		for _, decision := range recipe.Profile.Decisions {
			for _, modelRef := range decision.ModelRefs {
				if modelRef.Model != "local/mistral-small-4" {
					continue
				}
				mistralRefs++
				if modelRef.UseReasoning == nil || !*modelRef.UseReasoning || modelRef.ReasoningEffort != "high" {
					t.Fatalf("%s Mistral modelRef must use explicit high effort: %+v", decision.Name, modelRef)
				}
			}
		}
	}
	if mistralRefs == 0 {
		t.Fatal("vLLM-SR Chorus recipe has no Mistral modelRefs")
	}
}

func assertChorusOrchestrationBudgets(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	assertChorusWorkflowBudget(t, cfg)
	assertChorusFusionBudgets(t, cfg)
	assertChorusReMoMBudget(t, cfg)
}

func assertChorusWorkflowBudget(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	workflow := chorusDecision(t, cfg, "accuracy", "accuracy_dynamic_workflow")
	if workflow.Algorithm == nil || workflow.Algorithm.Workflows == nil ||
		workflow.Algorithm.Workflows.MaxCompletionTokens != 2048 ||
		workflow.Algorithm.Workflows.Planner.MaxCompletionTokens != 2048 {
		t.Fatalf("accuracy workflow output budgets changed: %+v", workflow.Algorithm)
	}
}

func assertChorusFusionBudgets(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	decision := chorusDecision(t, cfg, "accuracy", "accuracy_expert_fusion")
	if decision.Algorithm == nil || decision.Algorithm.Fusion == nil ||
		decision.Algorithm.Fusion.MaxCompletionTokens != 2048 {
		t.Fatalf("accuracy Fusion output budget changed: %+v", decision.Algorithm)
	}
}

func assertChorusReMoMBudget(t *testing.T, cfg *RouterConfig) {
	t.Helper()
	remom := chorusDecision(t, cfg, "accuracy", "accuracy_multi_round_exploration")
	if remom.Algorithm == nil || remom.Algorithm.ReMoM == nil ||
		remom.Algorithm.ReMoM.MaxCompletionTokens == nil ||
		*remom.Algorithm.ReMoM.MaxCompletionTokens != 2048 {
		t.Fatalf("accuracy ReMoM output budget changed: %+v", remom.Algorithm)
	}
}

func chorusDecision(t *testing.T, cfg *RouterConfig, recipeName, decisionName string) Decision {
	t.Helper()
	recipe, ok := cfg.RecipeByName(RecipeName(recipeName))
	if !ok {
		t.Fatalf("missing recipe %q", recipeName)
	}
	for _, decision := range recipe.Profile.Decisions {
		if decision.Name == decisionName {
			return decision
		}
	}
	t.Fatalf("recipe %q is missing decision %q", recipeName, decisionName)
	return Decision{}
}
