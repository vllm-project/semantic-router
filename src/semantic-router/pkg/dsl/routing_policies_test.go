package dsl

import (
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRoutingPoliciesRoundTrip(t *testing.T) {
	source := `ROUTING {candidate_requirements: {capabilities: declared}}
MODEL local {context_window_size: 8192 max_output_tokens: 1024 capabilities: ["chat"]}
RECIPE secondary {
 ROUTING {candidate_requirements: {context: known_limits}}
}
ENTRYPOINT {model_names: ["secondary"] recipe: secondary}
`
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	text, err := DecompileConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	again, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if !reflect.DeepEqual(cfg.Recipes, again.Recipes) || !reflect.DeepEqual(cfg.CandidateRequirements, again.CandidateRequirements) || again.ModelConfig["local"].MaxOutputTokens != 1024 {
		t.Fatalf("DSL roundtrip lost contract:\n%s", text)
	}
	ast := ProgramToJSON(DecompileRoutingToAST(cfg))
	if !reflect.DeepEqual(ast.CandidateRequirements, cfg.CandidateRequirements) {
		t.Fatal("builder AST lost policies")
	}
	data, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := config.ParseYAMLBytes(append([]byte("version: v0.3\n"), data...))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(parsed.CandidateRequirements, cfg.CandidateRequirements) {
		t.Fatal("YAML lost candidate requirements")
	}
	if _, crdErr := EmitCRD(cfg, "test", "default"); crdErr == nil {
		t.Fatal("CRD silently discarded named routing")
	}
	defaultOnly, errs := Compile(`ROUTING {candidate_requirements: {context: known_limits}}`)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	crd, err := EmitCRD(defaultOnly, "test", "default")
	if err != nil || !strings.Contains(string(crd), "candidate_requirements:") {
		t.Fatalf("default CRD lost policy: %v\n%s", err, crd)
	}
}

func TestReplayPluginOverridesRoundTrip(t *testing.T) {
	for _, fields := range []string{
		"", "enabled: false", "capture_personal_data: false",
		"capture_request_body: false capture_response_body: false max_tool_trace_bytes: 0 max_tool_trace_steps: 0",
		"enabled: true capture_personal_data: true max_records: 500 max_body_bytes: 2048",
	} {
		t.Run(fields, func(t *testing.T) {
			cfg, errs := Compile("ROUTE everything {PRIORITY 1 MODEL \"local\" PLUGIN router_replay {" + fields + "}}")
			if len(errs) > 0 {
				t.Fatal(errs)
			}
			original, err := cfg.Decisions[0].Plugins[0].Configuration.AsStringMap()
			if err != nil {
				t.Fatal(err)
			}
			text, err := DecompileConfig(cfg)
			if err != nil {
				t.Fatal(err)
			}
			again, errs := Compile(text)
			if len(errs) > 0 {
				t.Fatal(errs)
			}
			got, err := again.Decisions[0].Plugins[0].Configuration.AsStringMap()
			if err != nil || !reflect.DeepEqual(got, original) {
				t.Fatalf("plugin overrides changed: %v\n%s", err, text)
			}
			astConfig, errs := CompileAST(DecompileRoutingToAST(cfg))
			if len(errs) > 0 {
				t.Fatal(errs)
			}
			ast, err := astConfig.Decisions[0].Plugins[0].Configuration.AsStringMap()
			if err != nil || !reflect.DeepEqual(ast, original) {
				t.Fatalf("AST overrides changed: %v %+v", err, ast)
			}
		})
	}
	for _, fields := range []string{`capture_personal_data: "false"`, `capture_request_body: 1`, `enabled: true unknown_field: false`} {
		if _, errs := Compile("ROUTE invalid {PRIORITY 1 MODEL \"local\" PLUGIN router_replay {" + fields + "}}"); len(errs) == 0 {
			t.Fatalf("accepted invalid replay override: %s", fields)
		}
	}
}

func TestRoutingPoliciesRejectInvalidDSL(t *testing.T) {
	for _, source := range []string{
		`ROUTING {candidate_requirements: {context: bounded}}`,
		`ROUTING {candidate_requirements: {capability: declared}}`,
		`ROUTING {candidate_requirements: false}`,
		`ROUTING {data_policy: {replay: false}}`,
		`ROUTING {strategy: priority data_policy: {replay: false}}`,
		`ROUTING {data_policy: {replay: false, unknown: 1}}`,
	} {
		if _, errs := Compile(source); len(errs) == 0 {
			t.Errorf("accepted invalid policy: %s", source)
		}
	}
}

func TestMultiFactorObjectiveAndQualityRoundTrip(t *testing.T) {
	source := `ROUTE choose {
 PRIORITY 1
 MODEL "a", "b"
 ALGORITHM multi_factor {
  minimum_candidates: 1
  objective: {strategy: lexicographic, priorities: [{factor: latency, tolerance: 0.05}, {factor: cost}]}
  quality: {index: "vllm-sr/general@1.0.0", on_missing: exclude, min_coverage: 0.8, min_score: 0}
  latency_metric: ttft
  latency_percentile: 90
  on_no_candidates: fail
 }
}`
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	want := cfg.Decisions[0].Algorithm
	if want.MultiFactor.Objective == nil || want.MultiFactor.Quality.MinScore == nil || *want.MultiFactor.Quality.MinScore != 0 || want.MultiFactor.Quality.MinCoverage != 0.8 || want.MultiFactor.LatencyMetric != "ttft" {
		t.Fatal("compile lost fields")
	}
	output, err := DecompileConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	again, errs := Compile(output)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if !reflect.DeepEqual(want, again.Decisions[0].Algorithm) {
		t.Fatalf("algorithm changed:\n%s", output)
	}
	for _, bad := range []string{
		strings.Replace(source, "min_coverage: 0.8", "min_coverge: 0.8", 1),
		strings.Replace(source, "priorities: [{factor: latency, tolerance: 0.05}, {factor: cost}]", "priorities: false", 1),
	} {
		if _, errs := Compile(bad); len(errs) == 0 {
			t.Fatal("invalid algorithm field was silently discarded")
		}
	}
}

func TestRequestParamsDefaultRoundTrip(t *testing.T) {
	source := `ROUTE bounded {
 PRIORITY 1
 MODEL "local"
 PLUGIN request_params {default_max_tokens: 4096 max_tokens_limit: 8192}
}`
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	output, err := DecompileConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	again, errs := Compile(output)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	want := cfg.Decisions[0].GetRequestParamsConfig()
	if want.DefaultMaxTokens == nil || want.DefaultMaxTokens.Value != 4096 || !reflect.DeepEqual(want, again.Decisions[0].GetRequestParamsConfig()) {
		t.Fatalf("request default lost through DSL: %s", output)
	}
	ast := DecompileRoutingToAST(cfg)
	astCfg, errs := CompileAST(ast)
	if len(errs) > 0 || !reflect.DeepEqual(want, astCfg.Decisions[0].GetRequestParamsConfig()) {
		t.Fatalf("AST roundtrip changed output default: %v", errs)
	}
	for _, value := range []string{"0", "-1", "1.5", `"4096"`} {
		if _, errs := Compile(strings.Replace(source, "default_max_tokens: 4096", "default_max_tokens: "+value, 1)); len(errs) == 0 {
			t.Errorf("accepted invalid default %s", value)
		}
	}
}

func TestAutomaticOutputAndForecastRoundTrip(t *testing.T) {
	source := `ROUTE capacity {
 PRIORITY 1
 MODEL "local"
 ALGORITHM multi_factor {expected_output_tokens: 4096}
 PLUGIN request_params {default_max_tokens: "auto"}
}`
	cfg, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	text, err := DecompileConfig(cfg)
	if err != nil {
		t.Fatal(err)
	}
	parsed, errs := Compile(text)
	if len(errs) > 0 {
		t.Fatal(errs)
	}
	if !parsed.Decisions[0].GetRequestParamsConfig().DefaultMaxTokens.IsAuto() || *parsed.Decisions[0].Algorithm.MultiFactor.ExpectedOutputTokens != 4096 {
		t.Fatalf("auto/forecast lost: %s", text)
	}
	astConfig, errs := CompileAST(DecompileRoutingToAST(cfg))
	if len(errs) > 0 || !reflect.DeepEqual(cfg.Decisions[0].GetRequestParamsConfig(), astConfig.Decisions[0].GetRequestParamsConfig()) {
		t.Fatalf("AST lost automatic default: %v", errs)
	}
}
