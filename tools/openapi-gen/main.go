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

// indexGroup names one section of the rendered endpoint index. match reports
// whether a route belongs in the group.
type indexGroup struct {
	heading string
	note    string
	match   func(path string) bool
}

// indexGroups mirrors the grouping previously maintained by hand in
// website/docs/api/apiserver.md. Routes that fit no group land in
// "Other endpoints" so the index never silently drops a catalog entry.
var indexGroups = []indexGroup{
	{
		heading: "Discovery and health",
		match: func(p string) bool {
			switch p {
			case "/health", "/ready", "/startup-status", "/api/v1", "/openapi.json", "/docs":
				return true
			}
			return false
		},
	},
	{
		heading: "Classification and signals",
		match: func(p string) bool {
			switch {
			case strings.HasPrefix(p, "/api/v1/classify"):
				return true
			case p == "/api/v1/eval", p == "/api/v1/nli", p == "/api/v1/embeddings",
				p == "/api/v1/similarity", p == "/api/v1/similarity/batch":
				return true
			}
			return false
		},
	},
	{
		heading: "Models and metrics",
		match: func(p string) bool {
			return strings.HasPrefix(p, "/info/") || strings.HasPrefix(p, "/metrics/") ||
				p == "/api/v1/embeddings/models" || p == "/v1/models"
		},
	},
	{
		heading: "Memory, vector stores, and files",
		note:    "These require the corresponding service to be enabled; otherwise the API returns `503`.",
		match: func(p string) bool {
			return strings.HasPrefix(p, "/v1/memory") || strings.HasPrefix(p, "/v1/vector_stores") ||
				strings.HasPrefix(p, "/v1/files")
		},
	},
}

// renderIndexMarkdown renders the grouped endpoint index table that
// website/docs/api/apiserver.md previously hand-maintained.
func renderIndexMarkdown() []byte {
	routes := apiserver.ExportedRoutes()
	assigned := make([]bool, len(routes))
	var body strings.Builder

	for _, group := range indexGroups {
		fmt.Fprintf(&body, "\n### %s\n", group.heading)
		if group.note != "" {
			fmt.Fprintf(&body, "\n%s\n", group.note)
		}
		body.WriteString("\n| Method | Path | Description |\n| --- | --- | --- |\n")
		for i, r := range routes {
			if assigned[i] || !group.match(r.Path) {
				continue
			}
			assigned[i] = true
			fmt.Fprintf(&body, "| `%s` | `%s` | %s |\n", r.Method, r.Path, r.Description)
		}
	}

	// Routes without a markdown home still get documented.
	var rest int
	for i, r := range routes {
		if assigned[i] {
			continue
		}
		if rest == 0 {
			body.WriteString("\n### Other endpoints\n\n| Method | Path | Description |\n| --- | --- | --- |\n")
		}
		rest++
		fmt.Fprintf(&body, "| `%s` | `%s` | %s |\n", r.Method, r.Path, r.Description)
	}

	return []byte(body.String())
}
