package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

const usage = `Usage: sr-dsl <command> [options]

Commands:
  compile    Compile routing DSL to YAML/CRD
  decompile  Convert YAML config to routing-only DSL
  validate   Validate a DSL file
  fmt        Format a DSL file
  generate   Generate DSL from natural language using an LLM

Examples:
  sr-dsl compile -o config.yaml --base providers.yaml privacy-router.dsl
  sr-dsl compile -o config.yaml config.dsl
  sr-dsl compile --format crd -o semanticrouter.yaml config.dsl
  sr-dsl decompile -o config.dsl config.yaml
  sr-dsl validate config.dsl
  sr-dsl validate --runtime-checks config.dsl
  sr-dsl fmt -o formatted.dsl config.dsl
  sr-dsl generate --api-url http://localhost:8090 --model Qwen2.5-72B "Route math to qwen-math, default to qwen2.5:3b"
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}

	cmd := os.Args[1]
	os.Args = append(os.Args[:1], os.Args[2:]...)

	switch cmd {
	case "compile":
		runCompile()
	case "decompile":
		runDecompile()
	case "validate":
		runValidate()
	case "fmt", "format":
		runFormat()
	case "generate":
		runGenerate()
	case "help", "-h", "--help":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		fmt.Fprint(os.Stderr, usage)
		os.Exit(1)
	}
}

func runCompile() {
	fs := flag.NewFlagSet("compile", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: stdout)")
	format := fs.String("format", "yaml", "Output format: yaml, crd")
	base := fs.String("base", "", "Base YAML config with infrastructure (version, listeners, providers); merged with compiled routing to produce a complete config")
	crdName := fs.String("name", "router", "CRD resource name (for --format crd)")
	crdNamespace := fs.String("namespace", "", "CRD namespace (for --format crd, default: \"default\")")
	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl compile [-o output.yaml] [--base providers.yaml] [--format yaml|crd] <input.dsl>")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLICompile(inputPath, *output, *format, *crdName, *crdNamespace, *base); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func runDecompile() {
	fs := flag.NewFlagSet("decompile", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: stdout)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl decompile [-o output.dsl] <input.yaml>")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLIDecompile(inputPath, *output); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}

func runValidate() {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	runtimeChecks := fs.Bool("runtime-checks", false, "Run native TEST block and projection partition runtime validation")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl validate [--runtime-checks] <input.dsl>")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	errCount := dsl.CLIValidateWithRunner(
		inputPath,
		os.Stdout,
		validateRunnerFactory(*runtimeChecks),
	)
	if errCount > 0 {
		os.Exit(1)
	}
}

func validateRunnerFactory(enableRuntimeChecks bool) dsl.TestBlockRunnerFactory {
	if !enableRuntimeChecks {
		return nil
	}
	return buildNativeTestBlockRunner
}

func runFormat() {
	fs := flag.NewFlagSet("fmt", flag.ExitOnError)
	output := fs.String("o", "", "Output file path (default: overwrite input)")

	if err := fs.Parse(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}

	if fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "Error: input file required")
		fmt.Fprintln(os.Stderr, "Usage: sr-dsl fmt [-o output.dsl] <input.dsl>")
		os.Exit(1)
	}

	inputPath := fs.Arg(0)
	if err := dsl.CLIFormat(inputPath, *output); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %s\n", err)
		os.Exit(1)
	}
}
