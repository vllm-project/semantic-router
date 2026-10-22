package modeldownload

import (
	"fmt"
	"path/filepath"
	"slices"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// BuildModelSpecs lists registered artifacts needed by the default API owner
// and request-reachable recipes. Unregistered paths are supplied locally and
// validated by actual provider preparation, not by the download registry.
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	inventory := &modelInventory{registry: cfg.MoMRegistry, specs: map[string]ModelSpec{}}
	scopes := []*config.RouterConfig{}
	defaultScope := *cfg.ModelConsumerScope()
	defaultScope.Recipes, defaultScope.Entrypoints = nil, nil
	defaultScope.SemanticCache.Enabled = false
	defaultScope.Tools.Enabled, defaultScope.Memory.Enabled = false, false
	defaultScope.VectorStore = nil
	serviceScope := cfg.ConfigForGlobalModelServices()
	projectedServices, err := config.ProjectRecipeModelBindings(serviceScope, plan, config.GlobalModelScope)
	if err != nil {
		return nil, err
	}
	if err := inventory.addScope(projectedServices, plan); err != nil {
		return nil, err
	}

	scopes = append(scopes, &defaultScope)
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		if recipe.Name != config.DefaultRecipeName {
			scoped := cfg.ConfigForRecipe(recipe)
			scoped.SemanticCache.Enabled = false
			scopes = append(scopes, scoped)
		}
	}
	for _, scope := range scopes {
		projected, err := config.ProjectRecipeModelBindings(scope, plan, scope.RoutingScope)
		if err != nil {
			return nil, err
		}
		if err := inventory.addScope(projected, plan); err != nil {
			return nil, err
		}
	}
	keys := make([]string, 0, len(inventory.specs))
	for key := range inventory.specs {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	specs := make([]ModelSpec, 0, len(keys))
	for _, key := range keys {
		specs = append(specs, inventory.specs[key])
	}
	return specs, nil
}

type modelInventory struct {
	registry map[string]string
	specs    map[string]ModelSpec
}

func (i *modelInventory) addScope(cfg *config.RouterConfig, plan *config.ModelBindingPlan) error {
	primary := strings.ToLower(strings.TrimSpace(cfg.EmbeddingConfig.ModelType))
	if primary == "" {
		primary = "qwen3"
	}
	global := cfg.RoutingScope == config.GlobalModelScope
	sharedServices := global
	needed := config.EmbeddingModelsNeeded(cfg, primary, sharedServices)
	scoped := *cfg
	scoped.Recipes, scoped.Entrypoints = nil, nil
	paths := map[string]*string{"qwen3": &scoped.Qwen3ModelPath, "mmbert": &scoped.MmBertModelPath, "multimodal": &scoped.MultiModalModelPath}
	explicitEmbedding, hasEmbedding := plan.Lookup(cfg.RoutingScope, "embedding")
	if global {
		explicitEmbedding, hasEmbedding = plan.LookupGlobal("embedding")
	}
	for model, path := range paths {
		if !needed[model] || (cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() && !hasEmbedding) || (hasEmbedding && model == primary) {
			*path = ""
		}
	}
	if hasEmbedding {
		scoped.EmbeddingConfig.Backend = config.EmbeddingBackendModelRuntime
	}
	// A remote default does not suppress an explicit local primary deployment.
	active := map[string]bool{
		"domain_classifier": cfg.NeedsCategoryMappingForRouting(), "pii_classifier": cfg.NeedsPIIMappingForRouting(),
		"prompt_guard": cfg.NeedsJailbreakMappingForRouting(), "fact_check_classifier": cfg.NeedsFactCheckModelForAPI() || cfg.NeedsFactCheckModelForRouting(),
		"feedback_detector":      cfg.NeedsFeedbackModelForAPI() || cfg.NeedsFeedbackModelForRouting(),
		"hallucination_detector": cfg.NeedsLocalHallucinationModelsForRouting() || cfg.NeedsHallucinationDetectorForDefaultRuntime(),
		"modality_detector":      isModalityClassifierEnabled(cfg), "embedding": needed[primary],
		config.RAGRerankerConsumer: cfg.NeedsRAGReranker(),
	}
	for _, rule := range cfg.ClassifierRules {
		active["classifier."+rule.Name] = true
	}
	for _, rule := range cfg.SafetyRules {
		active["safety."+rule.Name] = true
		if rule.Hazard != nil {
			active["safety."+rule.Name+".hazard"] = true
		}
	}
	required := ExtractRequiredFilesByModel(&scoped)
	for path, files := range embeddingModelRequiredFiles(&scoped) {
		required[path] = append(required[path], files...)
	}
	excludes := embeddingModelExcludePatterns(&scoped)
	explicitPaths := map[string]bool{}
	// Built-in module models run through implicit model_runtime deployments,
	// and the runtime downloads them.
	servedPaths := scoped.RuntimeServedModelPaths()
	// The runtime downloads a runtime-provisioned catalog model (Vela Omni)
	// itself; every other embedding model is provisioned as native weights.
	for _, path := range paths {
		if catalog := config.GetModelByPath(*path); *path != "" && catalog != nil && catalog.RuntimeProvisioned {
			explicitPaths[config.ResolveModelPath(*path)] = true
		}
	}

	for name := range cfg.ModelBindings {
		spec, ok := plan.Lookup(cfg.RoutingScope, name)
		if global {
			spec, ok = plan.LookupGlobal(name)
		}
		if !ok || !active[name] {
			continue
		}
		// The runtime or the external service owns the model; only an
		// explicit label map is the router's. The projected consumer names the
		// runtime's artifact as its model, so it is not a module download.
		if spec.Deployment.IsModelRuntime() {
			explicitPaths[config.ResolveModelPath(spec.Deployment.Artifact)] = true
		}
		if spec.Binding.MappingPath != "" {
			if err := i.addFile(spec.Binding.MappingPath, spec.Deployment.Artifact, spec.Deployment.Revision); err != nil {
				return err
			}
		}
	}
	if hasEmbedding && needed[primary] && explicitEmbedding.Deployment.Provider != "http" {
		explicitPaths[config.ResolveModelPath(explicitEmbedding.Deployment.Artifact)] = true
	}
	for _, path := range filterDisabledOptionalModelPaths(&scoped, ExtractModelPaths(&scoped)) {
		if explicitPaths[config.ResolveModelPath(path)] || servedPaths[config.ResolveModelPath(path)] {
			continue
		}
		if err := i.addDefault(ModelSpec{LocalPath: config.ResolveModelPath(path), RequiredFiles: append(slices.Clone(DefaultRequiredFiles), required[path]...), ExcludePatterns: excludes[config.ResolveModelPath(path)]}); err != nil {
			return err
		}
	}
	mappings := []struct{ consumer, path, model string }{
		{"domain_classifier", cfg.CategoryMappingPath, cfg.CategoryModel.ModelID},
		{"pii_classifier", cfg.PIIMappingPath, cfg.PIIModel.ModelID},
		{"prompt_guard", cfg.PromptGuard.JailbreakMappingPath, cfg.PromptGuard.ModelID},
		{"feedback_detector", cfg.FeedbackDetector.FeedbackMappingPath, cfg.FeedbackDetector.ModelID},
	}
	for _, mapping := range mappings {
		if !active[mapping.consumer] || mapping.path == "" {
			continue
		}
		var err error
		if spec, ok := plan.Lookup(cfg.RoutingScope, mapping.consumer); ok {
			err = i.addFile(mapping.path, spec.Deployment.Artifact, spec.Deployment.Revision)
		} else {
			err = i.addDefaultFile(mapping.path, mapping.model)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func (i *modelInventory) addFile(path, artifact, revision string) error {
	path = filepath.Clean(path)
	root := ""
	for candidate := range i.registry {
		if relative, err := filepath.Rel(candidate, path); err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && len(candidate) > len(root) {
			root = candidate
		}
	}
	if root == "" {
		return nil
	}
	file, err := filepath.Rel(root, path)
	if err != nil {
		return err
	}
	if filepath.Clean(root) != filepath.Clean(artifact) && !config.SameModelRepo(i.registry[root], artifact) {
		revision = ""
	}
	return i.add(ModelSpec{LocalPath: root, Revision: revision, RequiredFiles: []string{file}, FilesOnly: true, CheckONNX: filepath.Ext(file) == ".onnx", Strict: true})
}

// addDefaultFile adds a companion file of a module's default model at the
// model's registered release, the revision the model runtime serves.
func (i *modelInventory) addDefaultFile(path, model string) error {
	model = config.ResolveModelPath(model)
	repo, err := i.registeredRepo(model)
	if err != nil {
		return err
	}
	return i.addFile(path, model, modelRevision(model, repo))
}

func (i *modelInventory) addDefault(spec ModelSpec) error {
	path := config.ResolveModelPath(spec.LocalPath)
	repo, err := i.registeredRepo(path)
	if err != nil {
		return err
	}
	spec.Revision = modelRevision(path, repo)
	return i.add(spec)
}

func (i *modelInventory) registeredRepo(path string) (string, error) {
	repo := i.registry[path]
	if repo == "" {
		for alias, candidate := range i.registry {
			if config.ResolveModelPath(alias) == path {
				if repo != "" && repo != candidate {
					return "", fmt.Errorf("registry aliases for %q disagree", path)
				}
				repo = candidate
			}
		}
	}
	return repo, nil
}

func (i *modelInventory) add(next ModelSpec) error {
	next.LocalPath = config.ResolveModelPath(next.LocalPath)
	repo, err := i.registeredRepo(next.LocalPath)
	if err != nil {
		return err
	}
	if repo == "" {
		return nil
	}
	next.RepoID = repo

	if previous, exists := i.specs[next.LocalPath]; exists {
		// One directory cannot hold two simultaneously declared revisions.
		if previous.Revision != "" && next.Revision != "" && previous.Revision != next.Revision {
			return fmt.Errorf("artifact %q is required at conflicting revisions %q and %q", next.LocalPath, previous.Revision, next.Revision)
		}
		if next.Revision == "" {
			next.Revision = previous.Revision
		}
		next.RequiredFiles = append(previous.RequiredFiles, next.RequiredFiles...)
		next.RequiredFileGroups = append(previous.RequiredFileGroups, next.RequiredFileGroups...)
		// Companion files do not introduce another execution format. Only
		// two model consumers intersect their provider exclusion policies.
		switch {
		case next.FilesOnly:
			next.ExcludePatterns = previous.ExcludePatterns
		case previous.FilesOnly:
		default:
			next.ExcludePatterns = intersectStrings(previous.ExcludePatterns, next.ExcludePatterns)
		}
		next.FilesOnly = previous.FilesOnly && next.FilesOnly
		next.CheckONNX = previous.CheckONNX || next.CheckONNX
		next.Strict = previous.Strict || next.Strict
	}
	next.ExcludePatterns = modelDownloadExcludePatterns(next.LocalPath, repo, next.ExcludePatterns)
	next.RequiredFiles = uniqueStrings(next.RequiredFiles)
	i.specs[next.LocalPath] = next
	return nil
}

func uniqueStrings(values []string) []string {
	out := []string{}
	for _, value := range values {
		if value != "" && !slices.Contains(out, value) {
			out = append(out, value)
		}
	}
	return out
}

func intersectStrings(a, b []string) []string {
	var out []string
	for _, value := range a {
		if slices.Contains(b, value) {
			out = append(out, value)
		}
	}
	return out
}

// Explicit bindings retain their selected revision. Implicit built-in modules
// use the registry pin only when the local path still names that repository.
func modelRevision(path, repoID string) string {
	if model := config.GetModelByPath(path); model != nil && config.SameModelRepo(model.RepoID, repoID) && model.Revision != "" {
		return model.Revision
	}
	return "main"
}

func modelDownloadExcludePatterns(path, repoID string, runtimePatterns []string) []string {
	patterns := slices.Clone(runtimePatterns)
	if model := config.GetModelByPath(path); model != nil && config.SameModelRepo(model.RepoID, repoID) {
		for _, pattern := range model.DownloadExcludePatterns {
			if !slices.Contains(patterns, pattern) {
				patterns = append(patterns, pattern)
			}
		}
	}
	return patterns
}
