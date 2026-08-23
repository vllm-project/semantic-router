package agentmanagement

const (
	ToolCatalogDescribe   = "router.catalog.describe"
	ToolSkillsRead        = "router.skills.read"
	ToolModelsList        = "router.models.list"
	ToolRecipesExamples   = "router.recipes.examples"
	ToolRecipeGet         = "router.recipe.get"
	ToolRecipePrepare     = "router.recipe.prepare"
	ToolRecipeValidate    = "router.recipe.validate"
	ToolRecipeProbe       = "router.recipe.probe"
	ToolRecipeEvaluate    = "router.recipe.evaluate"
	ToolEntrypointPrepare = "router.entrypoint.prepare"
	ToolPublishPrepare    = "router.publish.prepare"
)

// BuiltinBuilderToolNames is the single canonical allowlist shared by default
// Profile bootstrap and the production composite registry startup gate.
func BuiltinBuilderToolNames() []string {
	return []string{
		ToolCatalogDescribe,
		ToolSkillsRead,
		ToolModelsList,
		ToolRecipesExamples,
		ToolRecipeGet,
		ToolRecipePrepare,
		ToolRecipeValidate,
		ToolRecipeProbe,
		ToolRecipeEvaluate,
		ToolEntrypointPrepare,
		ToolPublishPrepare,
	}
}
