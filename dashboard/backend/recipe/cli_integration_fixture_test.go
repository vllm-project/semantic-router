package recipe

import (
	"path/filepath"
	"testing"
)

// The CLI integration suite imports this package through the Dashboard
// (e2e/testing/vllm-sr-cli/test_integration_recipe_bearer.py). It must stay a
// valid package, its recipe.dsl the canonical DSL of its config.yaml.
func TestTheCLIIntegrationBearerFixtureIsAValidPackage(t *testing.T) {
	directory := filepath.Join("..", "..", "..", "e2e", "testing", "vllm-sr-cli", "fixtures", "bearer-recipe")
	validation, err := validatePackageDirectory(directory)
	if err != nil {
		t.Fatalf("the fixture is not a valid Recipe package: %v", err)
	}
	if validation.metadata.ID != "bearer-fixture" || len(validation.warnings) != 0 {
		t.Fatalf("fixture metadata=%+v warnings=%+v", validation.metadata, validation.warnings)
	}
}
