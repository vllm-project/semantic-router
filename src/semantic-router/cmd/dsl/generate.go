package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/nlgen"
)

func runGenerate() {
	os.Exit(doGenerate())
}

func doGenerate() int {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	apiURL := fs.String("api-url", envOrDefault("VLLM_API_URL", ""), "LLM API base URL (env: VLLM_API_URL)")
	model := fs.String("model", envOrDefault("VLLM_MODEL", ""), "Model name (env: VLLM_MODEL)")
	apiKey := fs.String("api-key", envOrDefault("VLLM_API_KEY", ""), "API key (env: VLLM_API_KEY)")
	temperature := fs.Float64("temperature", 0.1, "Sampling temperature")
	maxRetries := fs.Int("max-retries", 2, "Parse-repair retry count")
	timeout := fs.Int("timeout", 120, "Request timeout in seconds")
	output := fs.String("o", "", "Output file (default: stdout)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		return 1
	}

	if *apiURL == "" {
		fmt.Fprintln(os.Stderr, "Error: --api-url or VLLM_API_URL is required")
		return 1
	}
	if *model == "" {
		fmt.Fprintln(os.Stderr, "Error: --model or VLLM_MODEL is required")
		return 1
	}

	instruction := strings.Join(fs.Args(), " ")
	if instruction == "" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading stdin: %s\n", err)
			return 1
		}
		instruction = strings.TrimSpace(string(data))
	}
	if instruction == "" {
		fmt.Fprintln(os.Stderr, "Error: instruction required (positional arg or stdin)")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl generate [options] \"natural language description\"")
		return 1
	}

	client := nlgen.NewOpenAIClient(*apiURL, *model, *apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(*timeout)*time.Second)
	defer cancel()

	result, err := nlgen.GenerateFromNL(ctx, client, instruction,
		nlgen.WithTemperature(*temperature),
		nlgen.WithMaxRetries(*maxRetries),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		return 1
	}

	if result.ParseError != "" {
		fmt.Fprintf(os.Stderr, "Warning: output has parse errors after %d attempts: %s\n", result.Attempts, result.ParseError)
	}
	for _, w := range result.Warnings {
		fmt.Fprintf(os.Stderr, "Warning: %s\n", w)
	}
	if result.Attempts > 1 {
		fmt.Fprintf(os.Stderr, "Note: required %d attempts (repair loop)\n", result.Attempts)
	}

	if *output != "" {
		if err := os.WriteFile(*output, []byte(result.DSL+"\n"), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %s\n", err)
			return 1
		}
		fmt.Fprintf(os.Stderr, "Wrote DSL to %s\n", *output)
	} else {
		fmt.Println(result.DSL)
	}

	return 0
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
