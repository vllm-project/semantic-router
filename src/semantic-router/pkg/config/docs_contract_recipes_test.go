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
			"`assignments`",
			"`entrypoints[].model_names`",
			"`recipes[].routing`",
		},
	},
	{
		path: repoRel("website", "docs", "tutorials", "global", "entrypoints-and-recipes.md"),
		needles: []string{
			"`entrypoints`",
			"`recipes`",
			"aliases",
			"backend_refs:",
			"`/v1/models`",
			"There is no implicit default Recipe or automatic alias",
		},
	},
}

func TestEntrypointsRecipesDocsStayAligned(t *testing.T) {
	assertDocsContainAll(t, repoRootFromTestFile(t), entrypointsRecipesRequiredDocs)
}
