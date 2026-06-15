package handlers

import (
	"fmt"
	"strings"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type builderNLGenerationContext struct {
	baseYAML        string
	targetModelName string
	knownModelNames []string
}

func prepareBuilderNLGenerationContext(
	configPath string,
	connectionMode builderNLConnectionMode,
	runtimeOptions builderNLRuntimeOptions,
	reporter builderNLProgressReporter,
) (builderNLGenerationContext, error) {
	baseConfig, err := readBuilderNLBaseConfig(configPath)
	if err != nil {
		return builderNLGenerationContext{}, err
	}

	baseYAMLBytes, err := marshalYAMLBytes(baseConfig)
	if err != nil {
		return builderNLGenerationContext{}, fmt.Errorf("failed to build deploy base yaml: %w", err)
	}

	context := builderNLGenerationContext{
		baseYAML:        string(baseYAMLBytes),
		targetModelName: builderNLDraftTargetModelName(baseConfig),
		knownModelNames: builderNLConfiguredModelNames(baseConfig),
	}

	reportBuilderNLProgress(reporter, "context", builderNLProgressSuccess, "Loaded deploy base YAML without mutating runtime provider settings.", 0)
	reportBuilderNLProgress(
		reporter,
		"context",
		builderNLProgressInfo,
		fmt.Sprintf(
			"Runtime knobs: temperature %.2f, repair budget %d, per-call timeout %s.",
			runtimeOptions.Temperature,
			runtimeOptions.MaxRetries,
			runtimeOptions.TimeoutLabel,
		),
		0,
	)
	reportBuilderNLResolvedModels(reporter, connectionMode, context)
	return context, nil
}

func reportBuilderNLResolvedModels(
	reporter builderNLProgressReporter,
	connectionMode builderNLConnectionMode,
	context builderNLGenerationContext,
) {
	if len(context.knownModelNames) > 0 {
		contextMessage := fmt.Sprintf(
			"Resolved %d current router model card(s); preferred draft target is %q.",
			len(context.knownModelNames),
			context.targetModelName,
		)
		if connectionMode == builderNLConnectionModeDefault {
			contextMessage = fmt.Sprintf(
				"Resolved %d current router model card(s); Builder will still use %q to generate the draft while route references prefer %q.",
				len(context.knownModelNames),
				builderNLFallbackModelAlias,
				context.targetModelName,
			)
		}
		reportBuilderNLProgress(reporter, "context", builderNLProgressInfo, contextMessage, 0)
		return
	}

	contextMessage := fmt.Sprintf(
		"No current router model cards were found; Builder will fall back to %q.",
		context.targetModelName,
	)
	if connectionMode == builderNLConnectionModeDefault {
		contextMessage = fmt.Sprintf(
			"No current router model cards were found; Builder will use %q for both draft generation and fallback route references.",
			builderNLFallbackModelAlias,
		)
	}
	reportBuilderNLProgress(reporter, "context", builderNLProgressWarning, contextMessage, 0)
}

func readBuilderNLBaseConfig(configPath string) (*routerconfig.CanonicalConfig, error) {
	cfg, err := readCanonicalConfigFile(configPath)
	if err == nil {
		return cfg, nil
	}
	if strings.Contains(strings.ToLower(err.Error()), "no such file") {
		return &routerconfig.CanonicalConfig{}, nil
	}
	return nil, fmt.Errorf("failed to read builder deploy base: %w", err)
}

func builderNLConfiguredModelNames(config *routerconfig.CanonicalConfig) []string {
	if config == nil {
		return nil
	}

	names := make([]string, 0, len(config.Routing.ModelCards)+len(config.Providers.Models))
	seen := make(map[string]struct{}, len(config.Routing.ModelCards)+len(config.Providers.Models))
	appendName := func(raw string) {
		name := strings.TrimSpace(raw)
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		names = append(names, name)
	}

	appendName(config.Providers.Defaults.DefaultModel)
	for _, model := range config.Routing.ModelCards {
		appendName(model.Name)
	}
	for _, model := range config.Providers.Models {
		appendName(model.Name)
	}
	return names
}

func builderNLDraftTargetModelName(config *routerconfig.CanonicalConfig) string {
	names := builderNLConfiguredModelNames(config)
	if len(names) > 0 {
		return names[0]
	}
	return builderNLFallbackModelAlias
}

func builderNLKnownModelList(knownModelNames []string) string {
	if len(knownModelNames) == 0 {
		return "(none found in the current router config)"
	}
	return strings.Join(knownModelNames, ", ")
}

func builderNLConnectionModeOrDefault(mode builderNLConnectionMode) (builderNLConnectionMode, error) {
	if mode == "" {
		return builderNLConnectionModeDefault, nil
	}
	if mode != builderNLConnectionModeDefault && mode != builderNLConnectionModeCustom {
		return "", fmt.Errorf("unsupported connectionMode %q", mode)
	}
	return mode, nil
}

func buildBuilderNLTaskContext(
	request string,
	currentDSL string,
	targetModelName string,
	knownModelNames []string,
	connectionMode builderNLConnectionMode,
) string {
	preferredTarget := strings.TrimSpace(targetModelName)
	if preferredTarget == "" {
		preferredTarget = builderNLFallbackModelAlias
	}

	lines := []string{
		fmt.Sprintf("Original Builder routing request: %s", strings.TrimSpace(request)),
		fmt.Sprintf("Preferred target model for route references: %s", preferredTarget),
		fmt.Sprintf("Connection mode: %s", connectionMode),
		fmt.Sprintf("Fallback Builder alias only when no current router model is available: %s.", builderNLFallbackModelAlias),
		fmt.Sprintf("Known current router model cards: %s", builderNLKnownModelList(knownModelNames)),
		fmt.Sprintf(
			"If you emit SIGNAL domain declarations without mmlu_categories, the signal name must be one of these supported routing domains: %s.",
			strings.Join(routerconfig.SupportedRoutingDomainNames(), ", "),
		),
		`If a DSL signal name or route reference contains spaces, quote it, for example SIGNAL domain "computer science" { ... } and WHEN domain("computer science").`,
		`If you prefer an identifier-style signal name such as computer_science, declare mmlu_categories explicitly, for example mmlu_categories: ["computer science"].`,
		"Preserve unrelated valid declarations from the current DSL unless the request clearly replaces them.",
		"Do not emit YAML, providers/global blocks, or deploy-time endpoint configuration.",
	}
	if trimmedDSL := strings.TrimSpace(currentDSL); trimmedDSL != "" {
		lines = append(lines, "Current Builder DSL context:\n"+trimmedDSL)
	} else {
		lines = append(lines, "No current DSL is available. Build a fresh draft.")
	}
	return strings.Join(lines, "\n")
}
