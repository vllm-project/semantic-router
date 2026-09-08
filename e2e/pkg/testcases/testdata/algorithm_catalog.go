// This helper runs from the router module to export its public algorithm catalog.
package main

import (
	"encoding/json"
	"log"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func main() {
	if err := json.NewEncoder(os.Stdout).Encode(config.DecisionAlgorithmCatalog()); err != nil {
		log.Fatal(err)
	}
}
