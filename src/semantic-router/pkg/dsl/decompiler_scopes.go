package dsl

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DecompileConfig emits the shared routing catalog followed by request-facing
// entrypoints and isolated recipe programs. Single-profile configs retain the
// original routing-only output.
func DecompileConfig(cfg *config.RouterConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("cannot decompile a nil config")
	}
	base, err := DecompileRouting(cfg)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	sb.WriteString(base)
	if len(cfg.Entrypoints) > 0 {
		writeEntrypoints(&sb, cfg.Entrypoints)
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		if recipe.Name == config.DefaultRecipeName {
			continue
		}
		body, bodyErr := decompileRecipeBody(cfg, recipe)
		if bodyErr != nil {
			return "", bodyErr
		}
		writeRecipe(&sb, recipe, body)
	}
	return canonicalDSLOutput(sb.String()), nil
}

func canonicalDSLOutput(input string) string {
	return strings.TrimRight(input, "\n") + "\n"
}

func writeEntrypoints(sb *strings.Builder, entrypoints []config.EntrypointMapping) {
	sb.WriteString("# =============================================================================\n")
	sb.WriteString("# ENTRYPOINTS\n")
	sb.WriteString("# =============================================================================\n\n")
	for _, entrypoint := range entrypoints {
		sb.WriteString("ENTRYPOINT {\n")
		fmt.Fprintf(sb, "  model_names: %s\n", formatStringArray(entrypoint.ModelNames))
		fmt.Fprintf(sb, "  recipe: %q\n", entrypoint.Recipe)
		sb.WriteString("}\n\n")
	}
}

func writeRecipe(sb *strings.Builder, recipe *config.RoutingRecipe, body string) {
	sb.WriteString("# =============================================================================\n")
	fmt.Fprintf(sb, "# RECIPE %s\n", recipe.Name)
	sb.WriteString("# =============================================================================\n\n")
	fmt.Fprintf(sb, "RECIPE %s", quoteName(string(recipe.Name)))
	if recipe.Description != "" {
		fmt.Fprintf(sb, " (description = %q)", recipe.Description)
	}
	sb.WriteString(" {\n")
	sb.WriteString(indentDSL(body, "  "))
	sb.WriteString("}\n\n")
}

func decompileRecipeBody(cfg *config.RouterConfig, recipe *config.RoutingRecipe) (string, error) {
	scoped := cfg.ConfigForRecipe(recipe)
	if scoped == nil {
		return "", fmt.Errorf("cannot construct routing view for recipe %q", recipe.Name)
	}
	d := &decompiler{cfg: scoped, pluginTemplates: make(map[string]*pluginTemplate)}
	d.extractPluginTemplates()
	d.decompileRoutingStrategy()
	d.writeSection("SIGNALS")
	d.decompileSignals()
	if len(d.pluginTemplates) > 0 {
		d.writeSection("PLUGINS")
		d.decompilePluginTemplates()
	}
	d.writeSection("ROUTES")
	d.decompileDecisions()
	return d.sb.String(), nil
}

func indentDSL(input, prefix string) string {
	if input == "" {
		return ""
	}
	lines := strings.Split(input, "\n")
	var sb strings.Builder
	for i, line := range lines {
		if i == len(lines)-1 && line == "" {
			continue
		}
		if line != "" {
			sb.WriteString(prefix)
		}
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	return sb.String()
}

func appendConfigScopesToAST(prog *Program, cfg *config.RouterConfig) {
	if prog == nil || cfg == nil {
		return
	}
	for _, entrypoint := range cfg.Entrypoints {
		prog.Entrypoints = append(prog.Entrypoints, &EntrypointDecl{
			ModelNames: append([]string(nil), entrypoint.ModelNames...),
			Recipe:     string(entrypoint.Recipe),
		})
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		if recipe.Name == config.DefaultRecipeName {
			continue
		}
		scoped := cfg.ConfigForRecipe(recipe)
		prog.Recipes = append(prog.Recipes, &RecipeDecl{
			Name:        string(recipe.Name),
			Description: recipe.Description,
			Program:     DecompileRoutingToAST(scoped),
		})
	}
}
