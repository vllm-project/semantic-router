// e2e/cmd/e2e-audit/main.go

// Command e2e-audit emits the runtime-derived E2E execution graph as
// deterministic JSON on stdout: registered testcases, canonical registered
// profiles, each profile's resolved GetTestCases selection, and the derived
// drift sets. The output is generated audit evidence for issue #2379 and is
// never committed as a source of truth.
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/e2e/pkg/verification"

	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

func main() {
	inventory, err := verification.BuildInventory()
	if err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: %v\n", err)
		os.Exit(1)
	}

	output := struct {
		verification.Inventory
		Unreachable []string `json:"registered_but_unreachable"`
	}{
		Inventory:   inventory,
		Unreachable: inventory.Unreachable(),
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(output); err != nil {
		fmt.Fprintf(os.Stderr, "e2e-audit: encoding inventory: %v\n", err)
		os.Exit(1)
	}
}
