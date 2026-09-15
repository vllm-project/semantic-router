//go:build !windows && cgo

// Command openapi-gen emits the Router Apiserver OpenAPI 3.0 specification and
// endpoint index documentation from the route catalog. It never starts the
// router: generation reuses the same in-process catalog the apiserver serves.
//
// Usage:
//
//	openapi-gen -format json  > website/static/openapi/apiserver/apiserver.openapi.json
//	openapi-gen -format index > generated endpoint index markdown
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiserver"
)

const (
	formatJSON  = "json"
	formatIndex = "index"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "openapi-gen: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	format := flag.String("format", formatJSON, "output format: json | index")
	out := flag.String("o", "", "write output to file instead of stdout")
	flag.Parse()

	var data []byte
	var err error
	switch *format {
	case formatJSON:
		data, err = renderSpecJSON()
	case formatIndex:
		data = renderIndexMarkdown()
	default:
		return fmt.Errorf("unsupported -format %q: want %q or %q", *format, formatJSON, formatIndex)
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

// renderSpecJSON marshals the served OpenAPI spec deterministically. Go's
// encoding/json orders map keys lexicographically, so two runs over the same
// catalog always produce byte-identical output.
func renderSpecJSON() ([]byte, error) {
	spec := apiserver.ExportOpenAPISpec()
	data, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal openapi spec: %w", err)
	}
	return append(data, '\n'), nil
}

// renderIndexMarkdown renders the grouped endpoint index table that
// website/docs/api/apiserver.md previously hand-maintained.
func renderIndexMarkdown() []byte {
	routes := apiserver.ExportedRoutes()
	var body strings.Builder

	for _, capability := range apiserver.ExportedCapabilities() {
		fmt.Fprintf(&body, "\n### %s\n\n%s\n", capability.Name, capability.Description)
		body.WriteString("\n| Method | Path | Description |\n| --- | --- | --- |\n")
		for _, route := range routes {
			if route.Contract.Capability != capability.Name {
				continue
			}
			fmt.Fprintf(&body, "| `%s` | `%s` | %s |\n", route.Method, route.Path, route.Description)
		}
	}

	return []byte(body.String())
}
