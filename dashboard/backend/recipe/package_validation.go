package recipe

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

const recipeEnvironmentAllowlist = "VLLM_SR_RECIPE_ENV_ALLOWLIST"

var (
	pureEnvironmentReference = regexp.MustCompile(`^\$\{[A-Z_][A-Z0-9_]*\}$`)
	recipeEnvironmentName    = regexp.MustCompile(`^[A-Z_][A-Z0-9_]*$`)
	knowledgeBasePathPart    = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*$`)
)

var forbiddenRecipeEnvironmentNames = map[string]struct{}{
	"DASHBOARD_PLATFORM":               {},
	"DASHBOARD_READONLY":               {},
	"DASHBOARD_SETUP_MODE":             {},
	"DISABLE_DASHBOARD":                {},
	"HOME":                             {},
	"LOGNAME":                          {},
	"PATH":                             {},
	"PWD":                              {},
	"SHELL":                            {},
	"USER":                             {},
	"VLLM_SR_ALGORITHM_OVERRIDE":       {},
	"VLLM_SR_ACTIVE_RECIPE_DIR":        {},
	"VLLM_SR_CONFIG_BASE_DIR":          {},
	"VLLM_SR_MANAGED_STORAGE_BACKENDS": {},
	"VLLM_SR_PLATFORM":                 {},
	"VLLM_SR_RECIPE_ENV_ALLOWLIST":     {},
	"VLLM_SR_RECIPE_STORE_DIR":         {},
	"VLLM_SR_RUNTIME_CONFIG_PATH":      {},
	"VLLM_SR_SETUP_MODE":               {},
	"VLLM_SR_SOURCE_CONFIG_PATH":       {},
	"VLLM_SR_STACK_NAME":               {},
	"VLLM_SR_STATE_ROOT_DIR":           {},
}

type environmentBinding struct {
	name string
	path string
}

type packageValidation struct {
	metadata Metadata
	counts   Counts
	warnings []PackageWarning
}

func validatePackageDirectory(directory string) (packageValidation, error) {
	files, err := readRecipeObject(directory)
	if err != nil {
		return packageValidation{}, invalidRecipeError(err)
	}
	for _, document := range []struct {
		filename string
		data     []byte
	}{
		{filename: "metadata.yaml", data: files["metadata"]},
		{filename: "config.yaml", data: files["config"]},
		{filename: "probes.yaml", data: files["probes"]},
	} {
		if validationErr := rejectYAMLIndirection(document.filename, document.data); validationErr != nil {
			return packageValidation{}, invalidRecipeError(validationErr)
		}
	}
	service := NewService(Options{Directory: directory})
	descriptor, err := service.Describe()
	if err != nil || !descriptor.Managed || descriptor.Metadata == nil || descriptor.SourceHealth.Status != "ready" {
		if err == nil {
			err = errors.New("managed Recipe descriptor is not ready")
		}
		return packageValidation{}, invalidRecipeError(err)
	}
	credentialWarnings := detectCredentialWarnings(files["config"])
	if len(credentialWarnings) > 0 {
		return packageValidation{}, invalidRecipeError(fmt.Errorf("config.yaml contains a literal credential-like value at %s", credentialWarnings[0].Path))
	}
	if err := validateRuntimeConfig(files["config"]); err != nil {
		return packageValidation{}, invalidRecipeError(err)
	}
	if err := validateRoutingAssets(files["probes"], descriptor.Metadata.ID); err != nil {
		return packageValidation{}, invalidRecipeError(err)
	}
	if err := validateCanonicalDSL(files["config"], files["dsl"]); err != nil {
		return packageValidation{}, invalidRecipeError(err)
	}
	return packageValidation{
		metadata: *descriptor.Metadata,
		counts:   descriptor.Counts,
		warnings: detectEnvironmentBindingWarnings(files["config"]),
	}, nil
}

func validateRuntimeConfig(data []byte) error {
	if err := rejectYAMLIndirection("config.yaml", data); err != nil {
		return err
	}
	if setupModeEnabled(data) {
		return errors.New("config.yaml must not enable setup mode")
	}
	config, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return fmt.Errorf("config.yaml: %w", err)
	}
	if err := validatePackageKnowledgeBasePaths(config); err != nil {
		return err
	}
	if err := validatePackageMCPTransport(config); err != nil {
		return err
	}
	if _, err := extractEnvironmentBindings(data); err != nil {
		return err
	}
	return nil
}

func rejectYAMLIndirection(filename string, data []byte) error {
	var document yaml.Node
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	if err := decoder.Decode(&document); err != nil {
		return err
	}
	var extra yaml.Node
	if err := decoder.Decode(&extra); err == nil {
		return fmt.Errorf("%s must contain exactly one YAML document", filename)
	} else if !errors.Is(err, io.EOF) {
		return err
	}
	var walk func(*yaml.Node) error
	walk = func(node *yaml.Node) error {
		if node == nil {
			return nil
		}
		if node.Style&yaml.TaggedStyle != 0 || node.Anchor != "" || node.Alias != nil || node.Kind == yaml.AliasNode ||
			(node.Kind == yaml.ScalarNode && node.Value == "<<" && node.Tag == "!!merge") {
			return fmt.Errorf("%s must not contain YAML anchors, aliases, merge keys, or explicit tags", filename)
		}
		for _, child := range node.Content {
			if err := walk(child); err != nil {
				return err
			}
		}
		return nil
	}
	return walk(&document)
}

func validatePackageMCPTransport(config *routerconfig.RouterConfig) error {
	mcp := config.MCPCategoryModel
	if !mcp.Enabled {
		return nil
	}
	transport := strings.ToLower(strings.TrimSpace(mcp.TransportType))
	if transport == "" {
		if strings.TrimSpace(mcp.URL) != "" && strings.TrimSpace(mcp.Command) == "" {
			transport = "streamable-http"
		} else {
			transport = "stdio"
		}
	}
	if transport == "stdio" || strings.TrimSpace(mcp.Command) != "" || len(mcp.Args) != 0 || len(mcp.Env) != 0 {
		return errors.New("config.yaml must not enable a local-process MCP transport")
	}
	return nil
}

func validatePackageKnowledgeBasePaths(config *routerconfig.RouterConfig) error {
	for _, knowledgeBase := range config.KnowledgeBases {
		raw := strings.TrimSpace(knowledgeBase.Source.Path)
		if raw == "" || strings.Contains(raw, "\\") || strings.HasPrefix(raw, "/") {
			return fmt.Errorf("global.model_catalog.kbs[%q]: source.path must use the canonical knowledge_bases/<name>/ form", knowledgeBase.Name)
		}
		trimmed := strings.TrimSuffix(raw, "/")
		cleaned := path.Clean(trimmed)
		parts := strings.Split(cleaned, "/")
		if cleaned != trimmed || len(parts) < 2 || parts[0] != "knowledge_bases" {
			return fmt.Errorf("global.model_catalog.kbs[%q]: source.path must use the canonical knowledge_bases/<name>/ form", knowledgeBase.Name)
		}
		for _, part := range parts[1:] {
			if !knowledgeBasePathPart.MatchString(part) {
				return fmt.Errorf("global.model_catalog.kbs[%q]: source.path contains an unsafe path component", knowledgeBase.Name)
			}
		}
	}
	return nil
}

func setupModeEnabled(data []byte) bool {
	var document struct {
		Setup *struct {
			Mode bool `yaml:"mode"`
		} `yaml:"setup"`
	}
	return yaml.Unmarshal(data, &document) == nil && document.Setup != nil && document.Setup.Mode
}

func validateRoutingAssets(data []byte, recipeID string) error {
	manifest, err := decodeProbes(data)
	if err != nil {
		return err
	}
	prefix := "config/recipes/" + recipeID + "/"
	if manifest.RoutingAssets.YAML != prefix+"config.yaml" || manifest.RoutingAssets.DSL != prefix+"recipe.dsl" {
		return errors.New("probes.yaml routing_assets must reference this Recipe's canonical config.yaml and recipe.dsl")
	}
	return nil
}

func validateCanonicalDSL(configData, dslData []byte) error {
	source := string(dslData)
	diagnostics, parseErrors := dsl.Validate(source)
	if len(parseErrors) > 0 {
		return fmt.Errorf("recipe.dsl parse failed: %v", parseErrors)
	}
	if len(diagnostics) > 0 {
		return fmt.Errorf("recipe.dsl validation failed: %s", diagnostics[0].String())
	}
	runtimeConfig, err := routerconfig.ParseYAMLBytesWithoutEnvExpansion(configData)
	if err != nil {
		return err
	}
	canonical, err := dsl.Decompile(runtimeConfig)
	if err != nil {
		return fmt.Errorf("recipe.dsl canonicalization failed: %w", err)
	}
	if canonical != source {
		return errors.New("recipe.dsl is not the canonical DSL generated from config.yaml")
	}
	compiled, compileErrors := dsl.Compile(source)
	if len(compileErrors) > 0 || compiled == nil {
		return fmt.Errorf("recipe.dsl compile failed: %v", compileErrors)
	}
	return nil
}

func detectCredentialWarnings(configData []byte) []PackageWarning {
	var document yaml.Node
	if err := yaml.Unmarshal(configData, &document); err != nil || len(document.Content) == 0 {
		return []PackageWarning{}
	}
	warnings := []PackageWarning{}
	walkCredentialNodes(document.Content[0], "", "", &warnings)
	return warnings
}

func detectEnvironmentBindingWarnings(configData []byte) []PackageWarning {
	bindings, err := extractEnvironmentBindings(configData)
	if err != nil {
		return []PackageWarning{}
	}
	warnings := make([]PackageWarning, 0, len(bindings))
	for _, binding := range bindings {
		warnings = append(warnings, PackageWarning{
			Code:    "environment_binding_required",
			Path:    binding.path,
			Message: fmt.Sprintf("Recipe requires environment variable %s to be explicitly authorized before activation.", binding.name),
		})
	}
	return warnings
}

// ValidateEnvironmentBindings enforces the process-scoped allowlist populated
// by `vllm-sr serve --recipe-env`. It never returns or logs secret values.
func ValidateEnvironmentBindings(configData []byte) error {
	bindings, err := extractEnvironmentBindings(configData)
	if err != nil {
		return wrapPackageError(ErrorEnvironmentBinding, http.StatusUnprocessableEntity, "Recipe environment bindings are invalid.", err)
	}
	allowlist, err := parseRecipeEnvironmentAllowlist(os.Getenv(recipeEnvironmentAllowlist))
	if err != nil {
		return wrapPackageError(ErrorEnvironmentBinding, http.StatusConflict, "Required Recipe environment bindings are not authorized or unavailable.", err)
	}
	for _, binding := range bindings {
		value, present := os.LookupEnv(binding.name)
		if _, allowed := allowlist[binding.name]; !allowed || !present || value == "" {
			return wrapPackageError(ErrorEnvironmentBinding, http.StatusConflict, "Required Recipe environment bindings are not authorized or unavailable.", fmt.Errorf("environment binding %s is unavailable", binding.name))
		}
	}
	return nil
}

func parseRecipeEnvironmentAllowlist(raw string) (map[string]struct{}, error) {
	result := map[string]struct{}{}
	if strings.TrimSpace(raw) == "" {
		return result, nil
	}
	for _, value := range strings.Split(raw, ",") {
		name := strings.TrimSpace(value)
		if name == "" || !recipeEnvironmentName.MatchString(name) {
			return nil, errors.New("recipe environment allowlist is invalid")
		}
		if _, forbidden := forbiddenRecipeEnvironmentNames[name]; forbidden {
			return nil, errors.New("recipe environment allowlist contains a reserved name")
		}
		result[name] = struct{}{}
	}
	return result, nil
}

func extractEnvironmentBindings(configData []byte) ([]environmentBinding, error) {
	var document yaml.Node
	if err := yaml.Unmarshal(configData, &document); err != nil || len(document.Content) == 0 {
		if err == nil {
			err = errors.New("config.yaml is empty")
		}
		return nil, err
	}
	bindings := map[string]environmentBinding{}
	if err := walkEnvironmentBindingNodes(document.Content[0], "", bindings); err != nil {
		return nil, err
	}
	result := make([]environmentBinding, 0, len(bindings))
	for _, binding := range bindings {
		result = append(result, binding)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].path != result[j].path {
			return result[i].path < result[j].path
		}
		return result[i].name < result[j].name
	})
	return result, nil
}

func walkEnvironmentBindingNodes(node *yaml.Node, pointer string, bindings map[string]environmentBinding) error {
	if node == nil {
		return nil
	}
	switch node.Kind {
	case yaml.MappingNode:
		for index := 0; index+1 < len(node.Content); index += 2 {
			key, value := strings.TrimSpace(node.Content[index].Value), node.Content[index+1]
			childPointer := pointer + "/" + escapeJSONPointer(key)
			if key == "api_key_env" && value.Kind == yaml.ScalarNode && strings.TrimSpace(value.Value) != "" {
				if err := addEnvironmentBinding(strings.TrimSpace(value.Value), childPointer, bindings); err != nil {
					return err
				}
			}
			if err := walkEnvironmentBindingNodes(value, childPointer, bindings); err != nil {
				return err
			}
		}
	case yaml.SequenceNode:
		for index, child := range node.Content {
			if err := walkEnvironmentBindingNodes(child, fmt.Sprintf("%s/%d", pointer, index), bindings); err != nil {
				return err
			}
		}
	case yaml.ScalarNode:
		for _, name := range requiredEnvironmentReferences(node.Value) {
			if err := addEnvironmentBinding(name, pointer, bindings); err != nil {
				return err
			}
		}
	}
	return nil
}

func addEnvironmentBinding(name, pointer string, bindings map[string]environmentBinding) error {
	if !recipeEnvironmentName.MatchString(name) {
		return fmt.Errorf("config.yaml contains an invalid Recipe environment binding at %s", pointer)
	}
	if _, forbidden := forbiddenRecipeEnvironmentNames[name]; forbidden {
		return fmt.Errorf("config.yaml contains a reserved Recipe environment binding at %s", pointer)
	}
	bindings[pointer+"\x00"+name] = environmentBinding{name: name, path: pointer}
	return nil
}

func requiredEnvironmentReferences(value string) []string {
	result := []string{}
	for index := 0; index < len(value); {
		if value[index] != '$' {
			index++
			continue
		}
		if index+1 < len(value) && value[index+1] == '$' {
			index += 2
			continue
		}
		if index+1 < len(value) && value[index+1] == '{' {
			end := strings.IndexByte(value[index+2:], '}')
			if end < 0 {
				index++
				continue
			}
			end += index + 2
			inner := value[index+2 : end]
			name := inner
			if separator := strings.Index(inner, ":-"); separator > 0 {
				name = inner[:separator]
			} else if separator := strings.Index(inner, "-"); separator > 0 {
				name = inner[:separator]
			}
			if name != "" {
				result = append(result, name)
			}
			index = end + 1
			continue
		}
		end := index + 1
		for end < len(value) && isEnvironmentNameByte(value[end]) {
			end++
		}
		if end > index+1 {
			result = append(result, value[index+1:end])
			index = end
			continue
		}
		index++
	}
	return result
}

func isEnvironmentNameByte(value byte) bool {
	return (value >= 'A' && value <= 'Z') ||
		(value >= 'a' && value <= 'z') ||
		(value >= '0' && value <= '9') ||
		value == '_'
}

func walkCredentialNodes(node *yaml.Node, path, parentKey string, warnings *[]PackageWarning) {
	if node == nil {
		return
	}
	switch node.Kind {
	case yaml.MappingNode:
		for index := 0; index+1 < len(node.Content); index += 2 {
			keyNode, valueNode := node.Content[index], node.Content[index+1]
			key := strings.TrimSpace(keyNode.Value)
			childPath := path + "/" + escapeJSONPointer(key)
			if credentialValueShouldWarn(key, parentKey, valueNode) || scalarURLHasEmbeddedCredential(valueNode) {
				*warnings = append(*warnings, PackageWarning{
					Code:    "literal_credential",
					Path:    childPath,
					Message: "Configuration contains a literal credential-like value; review it before activation.",
				})
			}
			walkCredentialNodes(valueNode, childPath, strings.ToLower(key), warnings)
		}
	case yaml.SequenceNode:
		for index, child := range node.Content {
			walkCredentialNodes(child, fmt.Sprintf("%s/%d", path, index), parentKey, warnings)
		}
	}
}

func scalarURLHasEmbeddedCredential(value *yaml.Node) bool {
	if value == nil || value.Kind != yaml.ScalarNode || !strings.Contains(value.Value, "://") {
		return false
	}
	parsed, err := url.Parse(value.Value)
	if err != nil || !parsed.IsAbs() || parsed.Host == "" {
		return false
	}
	if parsed.User != nil {
		return true
	}
	query, queryErr := url.ParseQuery(parsed.RawQuery)
	if parsed.RawQuery != "" && queryErr != nil {
		return true
	}
	for key := range query {
		normalized := strings.ToLower(strings.ReplaceAll(key, "-", "_"))
		for _, marker := range []string{"key", "token", "secret", "password", "credential", "auth", "signature"} {
			if strings.Contains(normalized, marker) {
				return true
			}
		}
	}
	return false
}

func credentialValueShouldWarn(key, parentKey string, value *yaml.Node) bool {
	if value == nil || value.Kind != yaml.ScalarNode || strings.TrimSpace(value.Value) == "" {
		return false
	}
	if pureEnvironmentReference.MatchString(strings.TrimSpace(value.Value)) {
		return false
	}
	normalized := strings.ToLower(strings.ReplaceAll(key, "-", "_"))
	if strings.HasSuffix(normalized, "_env") {
		return false
	}
	if credentialKey(normalized) {
		return true
	}
	switch normalized {
	case "authorization", "proxy_authorization", "cookie", "set_cookie", "x_api_key", "xapikey":
		return parentKey == "extra_headers" || parentKey == "headers"
	}
	return false
}

func credentialKey(normalized string) bool {
	for _, suffix := range []string{
		"api_key", "apikey", "access_key", "accesskey", "client_secret", "clientsecret",
		"password", "secret", "token", "private_key", "privatekey",
	} {
		if strings.HasSuffix(normalized, suffix) {
			return true
		}
	}
	return false
}

func escapeJSONPointer(value string) string {
	return strings.ReplaceAll(strings.ReplaceAll(value, "~", "~0"), "/", "~1")
}

func invalidRecipeError(err error) error {
	return wrapPackageError(ErrorInvalidRecipe, http.StatusUnprocessableEntity, "Recipe package failed managed Recipe validation.", err)
}
