package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"
)

func main() {
	inputs := flag.String("inputs", "", "Public/synthetic JSONL cases (required)")
	questionFile := flag.String("question", "", "Versioned Choice question JSON (required)")
	output := flag.String("output", "", "New JSONL output path; never overwritten (required)")
	live := flag.Bool("live", false, "Explicitly authorize paid calls to api.typesafe.ai")
	location := flag.String("location", "", "Coarse test region, no private hostnames (required)")
	revision := flag.String("revision", "", "Source revision used for this run (required)")
	timeout := flag.Duration("timeout", 10*time.Second, "Per-request timeout; no retries")
	limit := flag.Int("max-cases", 20, "Maximum number of paid requests")
	flag.Parse()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	err := run(ctx, runOptions{
		Inputs: *inputs, Question: *questionFile, Output: *output, Live: *live,
		Location: *location, Revision: *revision, Timeout: *timeout, MaxCases: *limit,
	})
	stop()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
