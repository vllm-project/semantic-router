package modeldownload

import (
	"fmt"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// BuildModelSpecs lists registered artifacts needed by the default API owner
// and request-reachable recipes. Unregistered paths are supplied locally and
// validated by actual provider preparation, not by the download registry.
func BuildModelSpecs(cfg *config.RouterConfig) ([]ModelSpec, error) {
	cfg = config.WithModelResources(cfg)
	inventory := &modelInventory{registry: cfg.MoMRegistry, specs: map[string]ModelSpec{}}
	scopes := []*config.RouterConfig{}
	defaultScope := *cfg.ModelConsumerScope()
	defaultScope.Recipes, defaultScope.Entrypoints = nil, nil
	defaultScope.SemanticCache.Enabled = false
	defaultScope.Memory.Enabled = false
	defaultScope.VectorStore = nil
	if err := inventory.addScope(cfg.ConfigForGlobalModelServices()); err != nil {
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
		if err := inventory.addScope(scope); err != nil {
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

// addScope adds the model paths a scope names that the Router provisions.
// The runtime downloads the models of model_runtime deployments, embedding
// deployments included, at their pinned revisions.
func (i *modelInventory) addScope(cfg *config.RouterConfig) error {
	scoped := *cfg
	scoped.Recipes, scoped.Entrypoints = nil, nil
	required := ExtractRequiredFilesByModel(&scoped)
	servedPaths := scoped.RuntimeServedModelPaths()
	for _, path := range ExtractModelPaths(&scoped) {
		if servedPaths[config.ResolveModelPath(path)] {
			continue
		}
		if err := i.addDefault(ModelSpec{LocalPath: config.ResolveModelPath(path), RequiredFiles: append(slices.Clone(DefaultRequiredFiles), required[path]...)}); err != nil {
			return err
		}
	}
	return nil
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
