package config

import "testing"

// entrypointsRecipesRequiredDocs ratchets the multi-recipe routing
// terminology (#2331) across the public config docs, mirroring the
// configContractRequiredDocs pattern in docs_contract_test.go.
var entrypointsRecipesRequiredDocs = []docNeedles{
	{
		path: "config/README.md",
		needles: []string{
			"`entrypoints`",
			"`recipes`",
			"tutorials/global/entrypoints-and-recipes.md",
		},
	},
	{
		path: repoRel("website", "docs", "installation", "configuration.md"),
		needles: []string{
			"`entrypoints[].recipe`",
			"`entrypoints[].model_names`",
			"`recipes[].routing`",
		},
	},
	{
		path: repoRel("website", "docs", "proposals", "unified-config-contract-v0-3.md"),
		needles: []string{
			"Entrypoints and multi-recipe routing",
			"`entrypoints[]`",
			"`recipes[]`",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "entrypoints-and-recipes.md"),
		needles: []string{
			"`entrypoints`",
			"`recipes`",
			"`vllm-sr/auto`",
			"`/v1/models`",
			"`providers.defaults.default_model`",
		},
	},
}

func TestEntrypointsRecipesDocsStayAligned(t *testing.T) {
	assertDocsContainAll(t, repoRootFromTestFile(t), entrypointsRecipesRequiredDocs)
}
