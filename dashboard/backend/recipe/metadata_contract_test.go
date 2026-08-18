package recipe

import (
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestMetadataContractCasesMatchProductionDecoder(t *testing.T) {
	contractPath := filepath.Join("..", "..", "..", "config", "schemas", "recipe-metadata-v1.contract.yaml")
	data, err := os.ReadFile(contractPath)
	if err != nil {
		t.Fatalf("ReadFile(%s): %v", contractPath, err)
	}
	var contract struct {
		SchemaVersion string `yaml:"schema_version"`
		Cases         []struct {
			Name     string `yaml:"name"`
			Valid    bool   `yaml:"valid"`
			Document string `yaml:"document"`
		} `yaml:"cases"`
	}
	if err := yaml.Unmarshal(data, &contract); err != nil {
		t.Fatalf("decode contract: %v", err)
	}
	if contract.SchemaVersion != "vllm-sr/recipe-metadata-contract/v1" || len(contract.Cases) == 0 {
		t.Fatalf("invalid metadata contract header: %#v", contract)
	}

	for _, testCase := range contract.Cases {
		t.Run(testCase.Name, func(t *testing.T) {
			_, err := decodeMetadata([]byte(testCase.Document))
			if accepted := err == nil; accepted != testCase.Valid {
				t.Fatalf("accepted = %v, want %v; err=%v", accepted, testCase.Valid, err)
			}
		})
	}
}
