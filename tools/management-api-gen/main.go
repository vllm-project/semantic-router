// Command management-api-gen emits checked client artifacts from the Router's
// canonical Management API registry. It never starts a listener.
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapiartifact"
)

const (
	formatOpenAPI    = "openapi"
	formatTypeScript = "typescript"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "management-api-gen: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	format := flag.String("format", formatOpenAPI, "output format: openapi | typescript")
	out := flag.String("o", "", "write output to file instead of stdout")
	flag.Parse()

	var (
		data []byte
		err  error
	)
	switch *format {
	case formatOpenAPI:
		data, err = managementapi.GenerateOpenAPIJSON()
	case formatTypeScript:
		data = managementapiartifact.RenderTypeScriptContract()
	default:
		return fmt.Errorf("unsupported -format %q: want %q or %q", *format, formatOpenAPI, formatTypeScript)
	}
	if err != nil {
		return err
	}

	if *out == "" {
		_, err = os.Stdout.Write(data)
		return err
	}
	return os.WriteFile(*out, data, 0o644)
}
