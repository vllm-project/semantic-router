package config

import (
	"flag"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"
)

func TestResolveConfigPathsHonorsDashboardBaseDirectoryOverride(t *testing.T) {
	configRoot := t.TempDir()
	baseDirectory := t.TempDir()
	t.Setenv("DASHBOARD_CONFIG_DIR", baseDirectory)
	cfg := &Config{ConfigFile: filepath.Join(configRoot, ".vllm-sr", "runtime-config.stack.yaml")}
	if err := resolveConfigPaths(cfg); err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(baseDirectory)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.ConfigDir != want {
		t.Fatalf("ConfigDir = %q, want %q", cfg.ConfigDir, want)
	}
	if cfg.AbsConfigPath == "" || filepath.Base(cfg.AbsConfigPath) != "runtime-config.stack.yaml" {
		t.Fatalf("AbsConfigPath = %q", cfg.AbsConfigPath)
	}
}

func TestParseAllowedOrigins(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []string
	}{
		{name: "empty"},
		{name: "only separators", raw: " , , "},
		{name: "single", raw: "http://localhost:3001", want: []string{"http://localhost:3001"}},
		{
			name: "trims, lowercases, drops blanks",
			raw:  " HTTP://Dash.Example ,, https://Other.Example:8443 ",
			want: []string{"http://dash.example", "https://other.example:8443"},
		},
		{name: "drops a trailing slash", raw: "http://localhost:3001/", want: []string{"http://localhost:3001"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseAllowedOrigins(tc.raw); !slices.Equal(got, tc.want) {
				t.Fatalf("parseAllowedOrigins(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}

func TestEvaluationRuntimeEnvironmentResolvesTypedServerConfiguration(t *testing.T) {
	t.Setenv("EVALUATION_DEPLOYMENTS_DIR", "/srv/evaluation-deployments")
	t.Setenv("EVALUATION_ROUTER_API_KEY_ENV", "ROUTER_EVAL_TOKEN")
	t.Setenv("EVALUATION_ENVOY_API_KEY_ENV", "ENVOY_EVAL_TOKEN")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_URL", "https://agent-task-ledger.internal")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_API_KEY_ENV", "AGENT_TASK_LEDGER_TOKEN")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_TIMEOUT", "40s")
	t.Setenv("EVALUATION_FAULT_RECOVERY_LEDGER_URL", "https://fault-ledger.internal")
	t.Setenv("EVALUATION_FAULT_RECOVERY_LEDGER_API_KEY_ENV", "FAULT_LEDGER_TOKEN")
	t.Setenv("EVALUATION_FAULT_RECOVERY_LEDGER_TIMEOUT", "45s")
	t.Setenv("EVALUATION_HARD_POLICY_LEDGER_URL", "https://policy-ledger.internal")
	t.Setenv("EVALUATION_HARD_POLICY_LEDGER_API_KEY_ENV", "POLICY_LEDGER_TOKEN")
	t.Setenv("EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_URL", "https://experiment-ledger.internal")
	t.Setenv("EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_API_KEY_ENV", "EXPERIMENT_LEDGER_TOKEN")
	t.Setenv("EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_TIMEOUT", "2m")

	previous := flag.CommandLine
	flag.CommandLine = flag.NewFlagSet("evaluation-config-test", flag.ContinueOnError)
	t.Cleanup(func() { flag.CommandLine = previous })
	flags := bindFeatureFlags(bindCoreFlags())
	if err := flag.CommandLine.Parse(nil); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{}
	applyCoreConfig(cfg, flags)
	if err := applyFeatureConfig(cfg, flags); err != nil {
		t.Fatal(err)
	}
	if err := validateEvaluationRuntimeConfig(cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.EvaluationRouterAPIKeyEnv != "ROUTER_EVAL_TOKEN" ||
		cfg.EvaluationEnvoyAPIKeyEnv != "ENVOY_EVAL_TOKEN" ||
		cfg.EvaluationDeploymentsDir != "/srv/evaluation-deployments" {
		t.Fatalf("evaluation credential refs = (%q, %q)", cfg.EvaluationRouterAPIKeyEnv, cfg.EvaluationEnvoyAPIKeyEnv)
	}
	if got := cfg.EvaluationAgentTaskLedger; got.URL != "https://agent-task-ledger.internal" ||
		got.APIKeyEnv != "AGENT_TASK_LEDGER_TOKEN" || got.Timeout != 40*time.Second {
		t.Fatalf("agent task config = %+v", got)
	}
	if got := cfg.EvaluationFaultRecoveryLedger; got.URL != "https://fault-ledger.internal" ||
		got.APIKeyEnv != "FAULT_LEDGER_TOKEN" || got.Timeout != 45*time.Second {
		t.Fatalf("fault recovery config = %+v", got)
	}
	if got := cfg.EvaluationHardPolicyLedger; got.URL != "https://policy-ledger.internal" ||
		got.APIKeyEnv != "POLICY_LEDGER_TOKEN" || got.Timeout != defaultEvaluationLedgerTimeout {
		t.Fatalf("hard policy config = %+v", got)
	}
	if got := cfg.EvaluationProductionExperimentLedger; got.URL != "https://experiment-ledger.internal" ||
		got.APIKeyEnv != "EXPERIMENT_LEDGER_TOKEN" || got.Timeout != 2*time.Minute {
		t.Fatalf("production experiment config = %+v", got)
	}
}

func TestLoadConfigIgnoresStaleEvaluationConfigurationWhenDisabled(t *testing.T) {
	t.Setenv("EVALUATION_ENABLED", "false")
	t.Setenv("EVALUATION_DEPLOYMENTS_DIR", " invalid ")
	t.Setenv("EVALUATION_ROUTER_API_KEY_ENV", "not-a-reference")
	t.Setenv("EVALUATION_ENVOY_API_KEY_ENV", "not-another-reference")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_URL", "https://ledger.internal/path")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_API_KEY_ENV", "")
	t.Setenv("EVALUATION_AGENT_TASK_LEDGER_TIMEOUT", "not-a-duration")
	t.Setenv("TARGET_ROUTER_API_URL", "https://router.internal/management")
	t.Setenv("TARGET_ENVOY_URL", "https://envoy.internal/")

	previousFlags := flag.CommandLine
	previousArgs := os.Args
	flag.CommandLine = flag.NewFlagSet("disabled-evaluation-config-test", flag.ContinueOnError)
	os.Args = []string{"dashboard-backend"}
	t.Cleanup(func() {
		flag.CommandLine = previousFlags
		os.Args = previousArgs
	})

	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("disabled Evaluation rejected stale configuration: %v", err)
	}
	if cfg.EvaluationEnabled {
		t.Fatal("Evaluation unexpectedly enabled")
	}
	if cfg.EvaluationDeploymentsDir != "" || cfg.EvaluationRouterAPIKeyEnv != "" ||
		cfg.EvaluationEnvoyAPIKeyEnv != "" || cfg.EvaluationAgentTaskLedger.Configured() {
		t.Fatalf("disabled Evaluation retained runtime configuration: %+v", cfg)
	}
	if cfg.RouterAPIURL != "https://router.internal/management" ||
		cfg.EnvoyURL != "https://envoy.internal/" {
		t.Fatalf("core target configuration changed: router=%q envoy=%q", cfg.RouterAPIURL, cfg.EnvoyURL)
	}
}

func TestEvaluationRuntimeFlagsOverrideEmptyEnvironment(t *testing.T) {
	for _, name := range []string{
		"EVALUATION_ROUTER_API_KEY_ENV", "EVALUATION_ENVOY_API_KEY_ENV",
		"EVALUATION_AGENT_TASK_LEDGER_URL", "EVALUATION_AGENT_TASK_LEDGER_API_KEY_ENV",
		"EVALUATION_AGENT_TASK_LEDGER_TIMEOUT",
		"EVALUATION_FAULT_RECOVERY_LEDGER_URL", "EVALUATION_FAULT_RECOVERY_LEDGER_API_KEY_ENV",
		"EVALUATION_FAULT_RECOVERY_LEDGER_TIMEOUT", "EVALUATION_HARD_POLICY_LEDGER_URL",
		"EVALUATION_HARD_POLICY_LEDGER_API_KEY_ENV", "EVALUATION_HARD_POLICY_LEDGER_TIMEOUT",
		"EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_URL",
		"EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_API_KEY_ENV",
		"EVALUATION_PRODUCTION_EXPERIMENT_LEDGER_TIMEOUT",
		"EVALUATION_DEPLOYMENTS_DIR",
	} {
		t.Setenv(name, "")
	}
	previous := flag.CommandLine
	flag.CommandLine = flag.NewFlagSet("evaluation-cli-flags-test", flag.ContinueOnError)
	t.Cleanup(func() { flag.CommandLine = previous })
	flags := bindFeatureFlags(bindCoreFlags())
	if err := flag.CommandLine.Parse([]string{
		"--evaluation-deployments", "/srv/paired-deployments",
		"--evaluation-router-api-key-env", "ROUTER_EVAL_TOKEN",
		"--evaluation-envoy-api-key-env", "ENVOY_EVAL_TOKEN",
		"--evaluation-hard-policy-ledger-url", "https://policy.internal",
		"--evaluation-hard-policy-ledger-api-key-env", "POLICY_TOKEN",
		"--evaluation-hard-policy-ledger-timeout", "90s",
	}); err != nil {
		t.Fatal(err)
	}
	cfg := &Config{}
	applyCoreConfig(cfg, flags)
	if err := applyFeatureConfig(cfg, flags); err != nil {
		t.Fatal(err)
	}
	if err := validateEvaluationRuntimeConfig(cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.EvaluationRouterAPIKeyEnv != "ROUTER_EVAL_TOKEN" ||
		cfg.EvaluationEnvoyAPIKeyEnv != "ENVOY_EVAL_TOKEN" ||
		cfg.EvaluationDeploymentsDir != "/srv/paired-deployments" {
		t.Fatalf("evaluation credential refs = (%q, %q)", cfg.EvaluationRouterAPIKeyEnv, cfg.EvaluationEnvoyAPIKeyEnv)
	}
	if got := cfg.EvaluationHardPolicyLedger; got.URL != "https://policy.internal" ||
		got.APIKeyEnv != "POLICY_TOKEN" || got.Timeout != 90*time.Second {
		t.Fatalf("hard policy flag config = %+v", got)
	}
}

func TestEvaluationDeploymentsDirectoryRejectsSurroundingWhitespace(t *testing.T) {
	cfg := Config{EvaluationDeploymentsDir: " /srv/evaluation-deployments"}
	if err := validateEvaluationRuntimeConfig(&cfg); err == nil {
		t.Fatal("evaluation deployments directory with surrounding whitespace was accepted")
	}
}

func TestEvaluationEndpointConfigurationIsZeroOrComplete(t *testing.T) {
	if got, err := resolveEvaluationEndpoint("ledger", "", "", ""); err != nil || got != (EvaluationServiceEndpointConfig{}) {
		t.Fatalf("zero endpoint = %+v err=%v", got, err)
	}
	for name, values := range map[string][3]string{
		"missing URL":        {"", "LEDGER_TOKEN", "30s"},
		"missing credential": {"https://ledger.internal", "", "30s"},
		"URL path":           {"https://ledger.internal/sealed", "LEDGER_TOKEN", "30s"},
		"literal key":        {"https://ledger.internal", "not-a-ref", "30s"},
		"zero timeout":       {"https://ledger.internal", "LEDGER_TOKEN", "0s"},
		"unbounded timeout":  {"https://ledger.internal", "LEDGER_TOKEN", "11m"},
	} {
		t.Run(name, func(t *testing.T) {
			rawURL, ref, timeout := values[0], values[1], values[2]
			if _, err := resolveEvaluationEndpoint("ledger", rawURL, ref, timeout); err == nil {
				t.Fatalf("unsafe endpoint accepted: url=%q ref=%q timeout=%q", rawURL, ref, timeout)
			}
		})
	}
	for name, endpoint := range map[string]EvaluationServiceEndpointConfig{
		"direct missing URL":     {APIKeyEnv: "LEDGER_TOKEN", Timeout: time.Second},
		"direct missing key":     {URL: "https://ledger.internal", Timeout: time.Second},
		"direct missing timeout": {URL: "https://ledger.internal", APIKeyEnv: "LEDGER_TOKEN"},
	} {
		t.Run(name, func(t *testing.T) {
			cfg := Config{EvaluationFaultRecoveryLedger: endpoint}
			if err := validateEvaluationRuntimeConfig(&cfg); err == nil {
				t.Fatalf("partial typed endpoint accepted: %+v", endpoint)
			}
		})
	}
}

func TestEvaluationRuntimeRejectsSharedOriginsAndCredentials(t *testing.T) {
	base := Config{
		RouterAPIURL:              "https://router.internal",
		EnvoyURL:                  "https://envoy.internal",
		EvaluationRouterAPIKeyEnv: "ROUTER_EVAL_TOKEN",
		EvaluationEnvoyAPIKeyEnv:  "ENVOY_EVAL_TOKEN",
		EvaluationAgentTaskLedger: EvaluationServiceEndpointConfig{
			URL: "https://agent-task.internal", APIKeyEnv: "AGENT_TASK_TOKEN", Timeout: time.Second,
		},
		EvaluationFaultRecoveryLedger: EvaluationServiceEndpointConfig{
			URL: "https://fault.internal", APIKeyEnv: "FAULT_TOKEN", Timeout: time.Second,
		},
		EvaluationHardPolicyLedger: EvaluationServiceEndpointConfig{
			URL: "https://policy.internal", APIKeyEnv: "POLICY_TOKEN", Timeout: time.Second,
		},
	}
	if err := validateEvaluationRuntimeConfig(&base); err != nil {
		t.Fatalf("valid distinct Evaluation config rejected: %v", err)
	}
	sharedOrigin := base
	sharedOrigin.EvaluationHardPolicyLedger.URL = base.RouterAPIURL
	if err := validateEvaluationRuntimeConfig(&sharedOrigin); err == nil {
		t.Fatal("ledger sharing the Router origin was accepted")
	}
	for name, routerURL := range map[string]string{
		"trailing slash": "https://router.internal/",
		"path":           "https://router.internal/api",
		"default port":   "https://router.internal:443/api",
	} {
		t.Run("shared Router origin with "+name, func(t *testing.T) {
			sharedOriginVariant := base
			sharedOriginVariant.RouterAPIURL = routerURL
			sharedOriginVariant.EvaluationHardPolicyLedger.URL = "https://router.internal"
			if err := validateEvaluationRuntimeConfig(&sharedOriginVariant); err == nil {
				t.Fatalf("ledger sharing Router origin through %q was accepted", routerURL)
			}
		})
	}
	sharedLedgerOrigin := base
	sharedLedgerOrigin.EvaluationHardPolicyLedger.URL = "https://agent-task.internal:443"
	if err := validateEvaluationRuntimeConfig(&sharedLedgerOrigin); err == nil {
		t.Fatal("ledgers sharing an effective origin through the default port were accepted")
	}
	sharedCredential := base
	sharedCredential.EvaluationHardPolicyLedger.APIKeyEnv = base.EvaluationRouterAPIKeyEnv
	if err := validateEvaluationRuntimeConfig(&sharedCredential); err == nil {
		t.Fatal("ledger sharing the Router credential ref was accepted")
	}
	managementCredential := base
	managementCredential.EvaluationRouterAPIKeyEnv = evaluationManagementKeyEnv
	if err := validateEvaluationRuntimeConfig(&managementCredential); err == nil {
		t.Fatal("Dashboard management credential was accepted for Router evaluation")
	}
	sharedRuntimeCredential := base
	sharedRuntimeCredential.EvaluationEnvoyAPIKeyEnv = base.EvaluationRouterAPIKeyEnv
	if err := validateEvaluationRuntimeConfig(&sharedRuntimeCredential); err == nil {
		t.Fatal("Router and Envoy sharing an evaluation credential ref was accepted")
	}
}
