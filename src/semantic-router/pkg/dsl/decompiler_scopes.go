package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// DecompileConfig emits the shared Model catalog followed by request-facing
// Entrypoints and isolated Recipe programs. A narrow Recipe document remains
// available through DecompileRouting.
func DecompileConfig(cfg *config.RouterConfig) (string, error) {
	if cfg == nil {
		return "", fmt.Errorf("cannot decompile a nil config")
	}
	if len(cfg.Recipes) == 0 && len(cfg.Entrypoints) == 0 {
		base, err := DecompileRouting(cfg)
		if err != nil {
			return "", err
		}
		return canonicalDSLOutput(base), nil
	}

	var sb strings.Builder
	models := canonicalDSLModels(cfg)
	if len(models) > 0 {
		d := &decompiler{cfg: cfg}
		d.writeSection("MODELS")
		d.decompileRoutingModels(models)
		sb.WriteString(d.sb.String())
	}
	if len(cfg.Entrypoints) > 0 {
		writeEntrypoints(&sb, config.CanonicalConfigFromRouterConfig(cfg).Entrypoints)
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
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

func writeEntrypoints(sb *strings.Builder, entrypoints []config.AuthoringEntrypoint) {
	sb.WriteString("# =============================================================================\n")
	sb.WriteString("# ENTRYPOINTS\n")
	sb.WriteString("# =============================================================================\n\n")
	for _, entrypoint := range entrypoints {
		sb.WriteString("ENTRYPOINT {\n")
		fmt.Fprintf(sb, "  name: %q\n", entrypoint.Name)
		if len(entrypoint.Aliases) > 0 {
			fmt.Fprintf(sb, "  aliases: %s\n", formatStringArray(entrypoint.Aliases))
		}
		if len(entrypoint.Rules) == 0 {
			fmt.Fprintf(sb, "  recipe: %q\n", entrypoint.Recipe)
			writeEntrypointAssignments(sb, "  ", entrypoint.Assignments)
		} else {
			writeEntrypointRules(sb, entrypoint.Rules)
		}
		sb.WriteString("}\n\n")
	}
}

func writeEntrypointRules(sb *strings.Builder, rules []config.AuthoringEntrypointRule) {
	sb.WriteString("  rules: [\n")
	for _, rule := range rules {
		sb.WriteString("    {\n")
		fmt.Fprintf(sb, "      name: %q\n", rule.Name)
		if len(rule.Matches) > 0 {
			sb.WriteString("      matches: [")
			for index, match := range rule.Matches {
				if index > 0 {
					sb.WriteString(", ")
				}
				sb.WriteString(formatEntrypointMatch(match))
			}
			sb.WriteString("]\n")
		}
		fmt.Fprintf(sb, "      recipe: %q\n", rule.Recipe)
		writeEntrypointAssignments(sb, "      ", rule.Assignments)
		sb.WriteString("    },\n")
	}
	sb.WriteString("  ]\n")
}

func writeEntrypointAssignments(sb *strings.Builder, indent string, assignments map[string]config.AuthoringAssignmentSet) {
	fmt.Fprintf(sb, "%sassignments: [\n", indent)
	decisionNames := make([]string, 0, len(assignments))
	for decisionName := range assignments {
		decisionNames = append(decisionNames, decisionName)
	}
	sort.Strings(decisionNames)
	for _, decisionName := range decisionNames {
		fmt.Fprintf(sb, "%s  { decision: %q, models: [", indent, decisionName)
		assignmentSet := assignments[decisionName]
		for index, assignment := range assignmentSet.Models {
			if index > 0 {
				sb.WriteString(", ")
			}
			sb.WriteString(formatEntrypointAssignment(assignment))
		}
		sb.WriteString("]")
		if assignmentSet.Fallback != nil {
			fmt.Fprintf(sb, ", fallback: { strategy: %q, on: %s }", assignmentSet.Fallback.Strategy, formatStringArray(assignmentSet.Fallback.On))
		}
		sb.WriteString(" },\n")
	}
	fmt.Fprintf(sb, "%s]\n", indent)
}

func formatEntrypointMatch(match config.AuthoringEntrypointMatch) string {
	if match.Claim != nil {
		return fmt.Sprintf("{ claim: { name: %q, exact: %s } }", match.Claim.Name, formatEntrypointClaimValue(match.Claim.Exact))
	}
	if match.Path != nil && match.Path.Exact != "" {
		return fmt.Sprintf("{ path: { exact: %q } }", match.Path.Exact)
	}
	return fmt.Sprintf("{ path: { prefix: %q } }", match.Path.Prefix)
}

func formatEntrypointClaimValue(value interface{}) string {
	switch typed := value.(type) {
	case bool:
		return fmt.Sprintf("%t", typed)
	case int:
		return fmt.Sprintf("%d", typed)
	case int64:
		return fmt.Sprintf("%d", typed)
	default:
		return fmt.Sprintf("%q", fmt.Sprint(typed))
	}
}

func formatEntrypointAssignment(assignment config.AuthoringModelAssignment) string {
	fields := []string{fmt.Sprintf("model: %q", assignment.Model)}
	if assignment.Priority != 0 {
		fields = append(fields, fmt.Sprintf("priority: %d", assignment.Priority))
	}
	if assignment.Weight != "" {
		fields = append(fields, fmt.Sprintf("weight: %q", assignment.Weight))
	}
	if assignment.LoRAName != "" {
		fields = append(fields, fmt.Sprintf("lora: %q", assignment.LoRAName))
	}
	if assignment.Reasoning != nil {
		reasoning := []string{fmt.Sprintf("enabled: %t", assignment.Reasoning.Enabled)}
		if assignment.Reasoning.Effort != "" {
			reasoning = append(reasoning, fmt.Sprintf("effort: %q", assignment.Reasoning.Effort))
		}
		if assignment.Reasoning.Description != "" {
			reasoning = append(reasoning, fmt.Sprintf("description: %q", assignment.Reasoning.Description))
		}
		fields = append(fields, "reasoning: { "+strings.Join(reasoning, ", ")+" }")
	}
	return "{ " + strings.Join(fields, ", ") + " }"
}

func writeRecipe(sb *strings.Builder, recipe *config.RoutingRecipe, body string) {
	sb.WriteString("# =============================================================================\n")
	fmt.Fprintf(sb, "# RECIPE %s\n", recipe.Name)
	sb.WriteString("# =============================================================================\n\n")
	fmt.Fprintf(sb, "RECIPE %s", quoteName(string(recipe.Name)))
	var options []string
	if recipe.Description != "" {
		options = append(options, fmt.Sprintf("description = %q", recipe.Description))
	}
	if len(options) > 0 {
		fmt.Fprintf(sb, " (%s)", strings.Join(options, ", "))
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
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	for _, entrypoint := range canonical.Entrypoints {
		prog.Entrypoints = append(prog.Entrypoints, authoringEntrypointToDecl(entrypoint))
	}
	for i := range cfg.Recipes {
		recipe := &cfg.Recipes[i]
		scoped := cfg.ConfigForRecipe(recipe)
		recipeProgram := DecompileRoutingToAST(scoped)
		recipeProgram.Models = nil
		prog.Recipes = append(prog.Recipes, &RecipeDecl{
			Name:        string(recipe.Name),
			Description: recipe.Description,
			Program:     recipeProgram,
		})
	}
}

func authoringEntrypointToDecl(input config.AuthoringEntrypoint) *EntrypointDecl {
	return &EntrypointDecl{
		Name: input.Name, Aliases: append([]string(nil), input.Aliases...),
		Recipe: input.Recipe, Assignments: authoringAssignmentsToDecl(input.Assignments),
		Rules: authoringEntrypointRulesToDecl(input.Rules),
	}
}

func authoringEntrypointRulesToDecl(input []config.AuthoringEntrypointRule) []*EntrypointRuleDecl {
	output := make([]*EntrypointRuleDecl, 0, len(input))
	for _, rule := range input {
		decl := &EntrypointRuleDecl{
			Name: rule.Name, Recipe: rule.Recipe,
			Assignments: authoringAssignmentsToDecl(rule.Assignments),
		}
		for _, match := range rule.Matches {
			converted := &EntrypointMatchDecl{}
			if match.Claim != nil {
				var exact Value
				switch value := match.Claim.Exact.(type) {
				case bool:
					exact = BoolValue{V: value}
				case int:
					exact = IntValue{V: value}
				case int64:
					exact = IntValue{V: int(value)}
				default:
					exact = StringValue{V: fmt.Sprint(value)}
				}
				converted.Claim = &EntrypointClaimMatchDecl{Name: match.Claim.Name, Exact: exact}
			}
			if match.Path != nil {
				converted.Path = &EntrypointPathMatchDecl{Exact: match.Path.Exact, Prefix: match.Path.Prefix}
			}
			decl.Matches = append(decl.Matches, converted)
		}
		output = append(output, decl)
	}
	return output
}

func authoringAssignmentsToDecl(input map[string]config.AuthoringAssignmentSet) map[string]*EntrypointAssignmentSetDecl {
	if input == nil {
		return nil
	}
	output := make(map[string]*EntrypointAssignmentSetDecl, len(input))
	for decisionName, assignmentSet := range input {
		convertedSet := &EntrypointAssignmentSetDecl{Models: make([]*EntrypointAssignmentDecl, 0, len(assignmentSet.Models))}
		for _, assignment := range assignmentSet.Models {
			converted := &EntrypointAssignmentDecl{Model: assignment.Model, Priority: assignment.Priority, Weight: assignment.Weight, LoRAName: assignment.LoRAName}
			if assignment.Reasoning != nil {
				converted.Reasoning = &EntrypointReasoningDecl{Enabled: assignment.Reasoning.Enabled, Effort: assignment.Reasoning.Effort, Description: assignment.Reasoning.Description}
			}
			convertedSet.Models = append(convertedSet.Models, converted)
		}
		if assignmentSet.Fallback != nil {
			convertedSet.Fallback = &EntrypointFallbackDecl{Strategy: assignmentSet.Fallback.Strategy, On: append([]string(nil), assignmentSet.Fallback.On...)}
		}
		output[decisionName] = convertedSet
	}
	return output
}
