package config

import (
	"flag"
	"os"
	"testing"
)

func TestSRBenchConfigRequiresServerOwnedOriginAndSecretReference(t *testing.T) {
	for _, origin := range []string{"http://127.0.0.1:8090", "https://bench.example"} {
		if err := ValidateSRBenchConfig(origin, "SR_BENCH_TOKEN"); err != nil {
			t.Fatalf("valid origin rejected: %v", err)
		}
	}
	for _, origin := range []string{"", "/local", "ftp://bench.example", "https://user:secret@bench.example", "https://bench.example/", "https://bench.example?target=x", "https://bench.example#x", " https://bench.example"} {
		if err := ValidateSRBenchConfig(origin, "SR_BENCH_TOKEN"); err == nil {
			t.Fatalf("invalid origin accepted: %q", origin)
		}
	}
	for _, ref := range []string{"", "token-with-hyphen", "literal-secret", " SR_BENCH_TOKEN"} {
		if err := ValidateSRBenchConfig("http://127.0.0.1:8090", ref); err == nil {
			t.Fatalf("invalid secret reference accepted: %q", ref)
		}
	}
}

func TestSRBenchFlagsAndEnvironmentResolveWithoutOldEvaluationContract(t *testing.T) {
	oldFlags, oldArgs := flag.CommandLine, os.Args
	t.Cleanup(func() { flag.CommandLine, os.Args = oldFlags, oldArgs })
	t.Setenv("SR_BENCH_URL", "http://127.0.0.1:8090")
	t.Setenv("SR_BENCH_TOKEN_ENV", "BENCH_SERVICE_TOKEN")
	flag.CommandLine = flag.NewFlagSet("sr-bench-test", flag.ContinueOnError)
	os.Args = []string{"dashboard", "--sr-bench-url=https://bench.example"}
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.SRBenchURL != "https://bench.example" || cfg.SRBenchTokenEnv != "BENCH_SERVICE_TOKEN" {
		t.Fatalf("unexpected sr-bench configuration: %q / %q", cfg.SRBenchURL, cfg.SRBenchTokenEnv)
	}
	if flag.Lookup("evaluation") != nil || flag.Lookup("evaluation-data") != nil {
		t.Fatal("retired Evaluation Plane flags remain registered")
	}
}
