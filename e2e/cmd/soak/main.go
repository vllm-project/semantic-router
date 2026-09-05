// Command soak drives a long-running load + memory-timeseries baseline against
// an already-running semantic-router stack.
//
// It is intentionally a standalone binary rather than an e2e testcase: soak
// results are reporting-only and must never gate the e2e suite.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/soak"
)

func main() {
	var (
		gatewayURL     = flag.String("gateway-url", "http://127.0.0.1:8801", "Envoy gateway base URL that fronts the router")
		metricsURL     = flag.String("metrics-url", "http://127.0.0.1:9190/metrics", "Router prometheus metrics endpoint")
		pprofURL       = flag.String("pprof-url", "http://127.0.0.1:6060", "Router pprof base URL (requires observability.profiling)")
		routerPID      = flag.Int("router-pid", 0, "Router PID; when set on linux, smaps_rollup is captured alongside heap dumps")
		out            = flag.String("out", "", "Output directory (default soak-results/<RFC3339 timestamp>)")
		model          = flag.String("model", "MoM", "Model name sent in each chat completion request")
		quick          = flag.Bool("quick", false, "Smoke mode (~9 min): 90s rounds, {8,32} concurrency sweep, 5k IDs; overrides -round-duration/-quiet-gap/-high-card-ids")
		concurrency    = flag.Int("concurrency", 100, "Concurrency for warmup and every soak round")
		rounds         = flag.Int("rounds", 3, "Number of fixed-concurrency soak rounds")
		roundDuration  = flag.Duration("round-duration", 25*time.Minute, "Duration of each soak round")
		quietGap       = flag.Duration("quiet-gap", 7*time.Minute, "Idle gap between rounds; sampling continues")
		sampleInterval = flag.Duration("sample-interval", 5*time.Second, "Metrics scrape interval")
		highCardIDs    = flag.Int("high-card-ids", 60000, "Unique session/user IDs used by the high-cardinality round")
	)
	flag.Parse()

	if *quick {
		set := map[string]bool{}
		flag.Visit(func(f *flag.Flag) { set[f.Name] = true })
		for _, name := range []string{"round-duration", "quiet-gap", "high-card-ids"} {
			if set[name] {
				fmt.Fprintf(os.Stderr, "soak: -quick overrides -%s; the explicit value is ignored\n", name)
			}
		}
	}

	outDir := *out
	if outDir == "" {
		outDir = filepath.Join("soak-results", time.Now().UTC().Format(time.RFC3339))
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fatal("create output dir: %v", err)
	}

	plan := soak.NewPlan(soak.Config{
		GatewayURL:     *gatewayURL,
		MetricsURL:     *metricsURL,
		PprofURL:       *pprofURL,
		RouterPID:      *routerPID,
		OutDir:         outDir,
		Model:          *model,
		Quick:          *quick,
		Concurrency:    *concurrency,
		Rounds:         *rounds,
		RoundDuration:  *roundDuration,
		QuietGap:       *quietGap,
		SampleInterval: *sampleInterval,
		HighCardIDs:    *highCardIDs,
	})

	runner, err := soak.NewRunner(plan)
	if err != nil {
		fatal("%v", err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	fmt.Printf("soak: writing results to %s (mode=%s, rounds=%d, concurrency=%d)\n",
		outDir, modeLabel(plan.Quick), plan.Rounds, plan.Concurrency)

	if err := runner.Run(ctx); err != nil {
		fatal("%v", err)
	}
	fmt.Printf("soak: done. summary=%s bench=%s\n",
		filepath.Join(outDir, "summary.json"),
		filepath.Join(outDir, "summary.bench"))
}

func modeLabel(quick bool) string {
	if quick {
		return "quick"
	}
	return "full"
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "Error: "+format+"\n", args...)
	os.Exit(1)
}
