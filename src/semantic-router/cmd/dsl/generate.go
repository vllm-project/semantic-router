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

type generateFlags struct {
	apiURL      string
	model       string
	apiKey      string
	temperature float64
	maxRetries  int
	timeout     int
	output      string
}

func parseGenerateFlags() (*generateFlags, []string, error) {
	fs := flag.NewFlagSet("generate", flag.ExitOnError)
	f := &generateFlags{}
	fs.StringVar(&f.apiURL, "api-url", envOrDefault("VLLM_API_URL", ""), "LLM API base URL (env: VLLM_API_URL)")
	fs.StringVar(&f.model, "model", envOrDefault("VLLM_MODEL", ""), "Model name (env: VLLM_MODEL)")
	fs.StringVar(&f.apiKey, "api-key", envOrDefault("VLLM_API_KEY", ""), "API key (env: VLLM_API_KEY)")
	fs.Float64Var(&f.temperature, "temperature", 0.1, "Sampling temperature")
	fs.IntVar(&f.maxRetries, "max-retries", 2, "Parse-repair retry count")
	fs.IntVar(&f.timeout, "timeout", 120, "Request timeout in seconds")
	fs.StringVar(&f.output, "o", "", "Output file (default: stdout)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		return nil, nil, err
	}
	return f, fs.Args(), nil
}

func readInstruction(args []string) (string, error) {
	if instruction := strings.Join(args, " "); instruction != "" {
		return instruction, nil
	}
	data, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("reading stdin: %w", err)
	}
	if s := strings.TrimSpace(string(data)); s != "" {
		return s, nil
	}
	return "", fmt.Errorf("instruction required (positional arg or stdin)")
}

func doGenerate() int {
	f, args, err := parseGenerateFlags()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		return 1
	}

	if f.apiURL == "" {
		fmt.Fprintln(os.Stderr, "Error: --api-url or VLLM_API_URL is required")
		return 1
	}
	if f.model == "" {
		fmt.Fprintln(os.Stderr, "Error: --model or VLLM_MODEL is required")
		return 1
	}

	instruction, err := readInstruction(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		return 1
	}

	client := nlgen.NewOpenAIClient(f.apiURL, f.model, f.apiKey)

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(f.timeout)*time.Second)
	defer cancel()

	result, err := nlgen.GenerateFromNL(ctx, client, instruction,
		nlgen.WithTemperature(f.temperature),
		nlgen.WithMaxRetries(f.maxRetries),
	)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		return 1
	}

	printWarnings(result)
	return writeOutput(f.output, result.DSL)
}

func printWarnings(result *nlgen.NLResult) {
	if result.ParseError != "" {
		fmt.Fprintf(os.Stderr, "Warning: output has parse errors after %d attempts: %s\n", result.Attempts, result.ParseError)
	}
	for _, w := range result.Warnings {
		fmt.Fprintf(os.Stderr, "Warning: %s\n", w)
	}
	if result.Attempts > 1 {
		fmt.Fprintf(os.Stderr, "Note: required %d attempts (repair loop)\n", result.Attempts)
	}
}

func writeOutput(path, content string) int {
	if path != "" {
		if err := os.WriteFile(path, []byte(content+"\n"), 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "Error writing output: %s\n", err)
			return 1
		}
		fmt.Fprintf(os.Stderr, "Wrote DSL to %s\n", path)
		return 0
	}
	fmt.Println(content)
	return 0
}

func envOrDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
