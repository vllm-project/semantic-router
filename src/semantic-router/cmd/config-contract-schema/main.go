package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func main() {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetEscapeHTML(false)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(config.BuildCanonicalContractSchema()); err != nil {
		fmt.Fprintf(os.Stderr, "encode canonical config schema: %v\n", err)
		os.Exit(1)
	}
}
