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
		writeEntrypoints(&sb, config.AuthoringEntrypointsFromRouterConfig(cfg))
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
		if entrypoint.Name == "" {
			continue
		}
		sb.WriteString("ENTRYPOINT {\n")
		fmt.Fprintf(sb, "  name: %q\n", entrypoint.Name)
		if len(entrypoint.Aliases) > 0 {
			fmt.Fprintf(sb, "  aliases: %s\n", formatStringArray(entrypoint.Aliases))
		}
		if len(entrypoint.Rules) > 0 {
			writeEntrypointRules(sb, entrypoint.Rules)
		} else {
			fmt.Fprintf(sb, "  recipe: %q\n", entrypoint.Recipe)
			writeEntrypointAssignments(sb, "  ", entrypoint.Assignments)
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
			fmt.Fprintf(sb, "      matches: %s\n", formatEntrypointMatches(rule.Matches))
		}
		fmt.Fprintf(sb, "      recipe: %q\n", rule.Recipe)
		writeEntrypointAssignments(sb, "      ", rule.Assignments)
		sb.WriteString("    },\n")
	}
	sb.WriteString("  ]\n")
}

func formatEntrypointMatches(matches []config.AuthoringEntrypointMatch) string {
	formatted := make([]string, 0, len(matches))
	for _, match := range matches {
		switch {
		case match.Claim != nil:
			formatted = append(formatted, fmt.Sprintf(
				"{ claim: { name: %q, exact: %s } }",
				match.Claim.Name,
				formatEntrypointMatchValue(match.Claim.Exact),
			))
		case match.Path != nil && match.Path.Exact != "":
			formatted = append(formatted, fmt.Sprintf("{ path: { exact: %q } }", match.Path.Exact))
		case match.Path != nil:
			formatted = append(formatted, fmt.Sprintf("{ path: { prefix: %q } }", match.Path.Prefix))
		}
	}
	return "[" + strings.Join(formatted, ", ") + "]"
}

func formatEntrypointMatchValue(value interface{}) string {
	switch typed := value.(type) {
	case string:
		return fmt.Sprintf("%q", typed)
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
	for _, entrypoint := range config.AuthoringEntrypointsFromRouterConfig(cfg) {
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
	decl := &EntrypointDecl{
		Name: input.Name, Aliases: append([]string(nil), input.Aliases...),
		Recipe: input.Recipe, Assignments: authoringAssignmentsToDecl(input.Assignments),
	}
	for _, rule := range input.Rules {
		converted := &EntrypointRuleDecl{
			Name: rule.Name, Recipe: rule.Recipe,
			Assignments: authoringAssignmentsToDecl(rule.Assignments),
		}
		for _, match := range rule.Matches {
			converted.Matches = append(converted.Matches, authoringMatchToDecl(match))
		}
		decl.Rules = append(decl.Rules, converted)
	}
	return decl
}

func authoringMatchToDecl(input config.AuthoringEntrypointMatch) *EntrypointMatchDecl {
	if input.Claim != nil {
		var exact Value
		switch value := input.Claim.Exact.(type) {
		case string:
			exact = StringValue{V: value}
		case bool:
			exact = BoolValue{V: value}
		case int:
			exact = IntValue{V: value}
		case int64:
			exact = IntValue{V: int(value)}
		}
		return &EntrypointMatchDecl{Claim: &EntrypointClaimMatchDecl{
			Name: input.Claim.Name, Exact: exact,
		}}
	}
	if input.Path == nil {
		return &EntrypointMatchDecl{}
	}
	return &EntrypointMatchDecl{Path: &EntrypointPathMatchDecl{
		Exact: input.Path.Exact, Prefix: input.Path.Prefix,
	}}
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
