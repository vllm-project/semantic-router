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
	if provider, _ := config.DefaultModelExecution(true); provider != "candle" && provider != "ort" {
		return nil, fmt.Errorf("unsupported build default model provider %q", provider)
	}
	plan, err := config.CompileModelBindings(cfg)
	if err != nil {
		return nil, err
	}
	inventory := &modelInventory{registry: cfg.MoMRegistry, specs: map[string]ModelSpec{}}
	scopes := []*config.RouterConfig{}
	defaultScope := *cfg.ModelConsumerScope()
	defaultScope.Recipes, defaultScope.Entrypoints = nil, nil
	scopes = append(scopes, &defaultScope)
	for _, recipe := range cfg.ReachableRoutingRecipes() {
		if recipe.Name != config.DefaultRecipeName {
			scopes = append(scopes, cfg.ConfigForRecipe(recipe))
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
	needed := config.EmbeddingModelsNeeded(cfg, primary, cfg.RoutingScope == config.DefaultRecipeName)
	scoped := *cfg
	scoped.Recipes, scoped.Entrypoints = nil, nil
	paths := map[string]*string{"qwen3": &scoped.Qwen3ModelPath, "gemma": &scoped.GemmaModelPath, "mmbert": &scoped.MmBertModelPath, "multimodal": &scoped.MultiModalModelPath, "bert": &scoped.BertModelPath}
	explicitEmbedding, hasEmbedding := plan.Lookup(cfg.RoutingScope, "embedding")
	for model, path := range paths {
		if !needed[model] || (cfg.EmbeddingModels.UsesRemoteEmbeddingBackend() && !hasEmbedding) || (hasEmbedding && model == primary) {
			*path = ""
		}
	}
	if hasEmbedding {
		scoped.EmbeddingConfig.Backend = config.EmbeddingBackendCandle
	}
	// A remote default does not suppress an explicit local primary deployment.
	active := map[string]bool{
		"domain_classifier": cfg.NeedsCategoryMappingForRouting(), "pii_classifier": cfg.NeedsPIIMappingForRouting(),
		"prompt_guard": cfg.NeedsJailbreakMappingForRouting(), "fact_check_classifier": cfg.NeedsFactCheckModelForAPI() || cfg.NeedsFactCheckModelForRouting(),
		"feedback_detector":       cfg.NeedsFeedbackModelForAPI() || cfg.NeedsFeedbackModelForRouting(),
		"hallucination_detector":  cfg.NeedsLocalHallucinationModelsForRouting() || cfg.NeedsHallucinationDetectorForDefaultRuntime(),
		"hallucination_explainer": cfg.NeedsLocalHallucinationNLIForAPI() || cfg.NeedsLocalHallucinationNLIForRouting() || cfg.NeedsLocalNLIForSemanticCache(),
		"modality_detector":       isModalityClassifierEnabled(cfg), "embedding": needed[primary],
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
	defaultProvider, _ := config.DefaultModelExecution(cfg.EmbeddingModels.UseCPU)
	if defaultProvider == "candle" {
		for path, files := range candleEmbeddingModelRequiredFiles(&scoped) {
			required[path] = append(required[path], files...)
		}
	}
	excludes := candleEmbeddingModelExcludePatterns(&scoped)
	explicitPaths := map[string]bool{}
	// Implicit Safety and Hazard modules use the same artifact contract as
	// their typed owners. Explicit per-rule heads are added below from plan.
	for _, hazard := range []bool{false, true} {
		if !scoped.NeedsLocalSafetyHeadForRouting(hazard) {
			continue
		}
		head, contract := scoped.SafetyModels.Safety, config.RemoteClassifierContractLabelDistribution
		if hazard {
			head, contract = scoped.SafetyModels.Hazard, config.RemoteClassifierContractLabelScores
		}
		provider, device := config.DefaultModelExecution(head.UseCPU)
		spec := config.ResolvedModelBinding{
			Recipe:     cfg.RoutingScope,
			Binding:    config.ModelBinding{Adapter: "modernbert", Contract: contract},
			Deployment: config.ModelDeployment{Provider: provider, Device: device, Artifact: head.ModelID},
		}
		if err := i.addDefaultDeployment(cfg, spec); err != nil {
			return err
		}
		explicitPaths[config.ResolveModelPath(head.ModelID)] = true
	}
	if defaultProvider == "ort" && scoped.EmbeddingModels.EmbeddingBackend() == config.EmbeddingBackendCandle {
		// Resolve implicit embeddings with the same provider artifact contract
		// as explicit bindings. In particular, ROCm requires ONNX graphs and
		// their external tensors rather than Candle safetensors.
		for model, path := range paths {
			if *path == "" {
				continue
			}
			spec := config.ResolvedModelBinding{
				Recipe: cfg.RoutingScope, Name: "embedding",
				Binding:    config.ModelBinding{Adapter: model, Contract: "embedding.v1"},
				Deployment: config.ModelDeployment{Provider: defaultProvider, Artifact: *path},
			}
			if err := i.addDefaultDeployment(cfg, spec); err != nil {
				return err
			}
			explicitPaths[config.ResolveModelPath(*path)] = true
		}
	}
	for name := range cfg.ModelBindings {
		spec, ok := plan.Lookup(cfg.RoutingScope, name)
		if !ok || !active[name] {
			continue
		}
		if spec.Deployment.Provider == "http" {
			if spec.Binding.MappingPath != "" {
				if err := i.addFile(spec.Binding.MappingPath, "", ""); err != nil {
					return err
				}
			}
			continue
		}
		artifact := config.ResolveModelPath(spec.Deployment.Artifact)
		explicitPaths[artifact] = true
		if err := i.addDeployment(cfg, spec); err != nil {
			return err
		}
	}
	if hasEmbedding && needed[primary] && explicitEmbedding.Deployment.Provider != "http" {
		explicitPaths[config.ResolveModelPath(explicitEmbedding.Deployment.Artifact)] = true
	}
	for _, path := range filterDisabledOptionalModelPaths(&scoped, ExtractModelPaths(&scoped)) {
		if explicitPaths[config.ResolveModelPath(path)] {
			continue
		}
		if err := i.addDefault(ModelSpec{LocalPath: config.ResolveModelPath(path), RequiredFiles: append(slices.Clone(DefaultRequiredFiles), required[path]...), ExcludePatterns: excludes[config.ResolveModelPath(path)]}); err != nil {
			return err
		}
	}
	mappings := map[string]string{"domain_classifier": cfg.CategoryMappingPath, "pii_classifier": cfg.PIIMappingPath, "prompt_guard": cfg.PromptGuard.JailbreakMappingPath, "feedback_detector": cfg.FeedbackDetector.FeedbackMappingPath}
	for consumer, path := range mappings {
		if active[consumer] && path != "" {
			revision, artifact := "", ""
			if spec, ok := plan.Lookup(cfg.RoutingScope, consumer); ok {
				revision, artifact = spec.Deployment.Revision, spec.Deployment.Artifact
			}
			if err := i.addFile(path, artifact, revision); err != nil {
				return err
			}
		}
	}
	return nil
}

func (i *modelInventory) addDeployment(cfg *config.RouterConfig, spec config.ResolvedModelBinding) error {
	path := config.ResolveModelPath(spec.Deployment.Artifact)
	files := []string{"config.json", "tokenizer.json"}
	if spec.Binding.Contract == config.RelevanceScoresContract {
		files = append(files, "matryoshka_config.json")
	}
	groups := [][]string{}
	var rerankerSelections []config.PairScorerSelection
	excludes := []string(nil)
	if spec.Deployment.Provider == "ort" {
		if spec.Binding.OperatingPoint != nil {
			// A calibrated policy also binds its native source checkpoint.
			// An existing graph-only cache cannot satisfy that identity check.
			groups = append(groups, []string{"*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin"})
		}
		if spec.Binding.Head != "" {
			head := spec.Binding.Head
			if !filepath.IsAbs(head) {
				head = filepath.Join(path, head)
			}
			if err := i.addFile(head, path, spec.Deployment.Revision); err != nil {
				return err
			}
		} else if spec.Binding.Contract == config.RelevanceScoresContract {
			selection := config.PairScorerSelection{}
			if spec.Binding.PairScorer != nil {
				selection = *spec.Binding.PairScorer
			}
			rerankerSelections = append(rerankerSelections, selection)
		} else {
			groups = append(groups, []string{"*.onnx", "onnx/*.onnx", "onnx/layer-*/*.onnx"})
		}
		if spec.Binding.Contract == "embedding.v1" {
			if spec.Binding.Adapter == "multimodal" {
				groups = [][]string{{"text_encoder.onnx", "onnx/text_encoder.onnx"}, {"image_encoder.onnx", "onnx/image_encoder.onnx"}, {"audio_encoder.onnx", "onnx/audio_encoder.onnx"}}
			} else {
				// The maintained MMBERT primary export is layer 22; a flat
				// single-graph export remains supported by the provider.
				groups = append(groups, []string{"model.onnx", "onnx/model.onnx", "onnx/layer-22/model.onnx"})
				layers := []int{cfg.EmbeddingConfig.TargetLayer}
				if cfg.RoutingScope == config.DefaultRecipeName && cfg.SemanticCache.Enabled && config.SemanticCacheEmbeddingModel(cfg) == "mmbert" {
					layers = append(layers, 6)
				}
				for _, layer := range layers {
					if layer > 0 && layer != 22 {
						groups = append(groups, []string{
							fmt.Sprintf("onnx/layer-%d/model.onnx", layer),
							fmt.Sprintf("onnx/model_layer_%d.onnx", layer),
							fmt.Sprintf("model_layer_%d.onnx", layer),
						})
					}
				}
			}
		}
	} else {
		// A registered Candle snapshot must contain native weights, even if an
		// ONNX export already exists. Sharded safetensors remain valid.
		groups = append(groups, []string{"*.safetensors", "*.safetensors.index.json", "pytorch_model*.bin"})
		if spec.Binding.Contract == config.RelevanceScoresContract {
			files = append(files, "classification_heads.safetensors")
			groups = [][]string{{"model.safetensors", "model.safetensors.index.json"}}
		}
		excludes = slices.Clone(onnxWeightExcludePatterns)
		if spec.Binding.Contract == "embedding.v1" && spec.Binding.Adapter == "gemma" {
			files = append(files, gemmaDenseWeightFiles...)
		}
		if spec.Binding.Head != "" {
			if err := i.add(ModelSpec{LocalPath: config.ResolveModelPath(spec.Binding.Head), RequiredFiles: []string{"config.json", "tokenizer.json"}, RequiredFileGroups: groups, ExcludePatterns: excludes, CheckONNX: spec.Deployment.Provider == "ort", Strict: true}); err != nil {
				return err
			}
		}
	}
	if err := i.add(ModelSpec{LocalPath: path, Revision: spec.Deployment.Revision, RequiredFiles: files, RequiredFileGroups: groups, RerankerSelections: rerankerSelections, ExcludePatterns: excludes, CheckONNX: spec.Deployment.Provider == "ort", Strict: true}); err != nil {
		return err
	}
	if spec.Binding.MappingPath != "" {
		if err := i.addFile(spec.Binding.MappingPath, path, spec.Deployment.Revision); err != nil {
			return err
		}
	}
	if spec.Binding.OperatingPoint != nil {
		return i.addFile(spec.Binding.OperatingPoint.ResolvePath(path), path, spec.Deployment.Revision)
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
	if filepath.Clean(root) != filepath.Clean(artifact) {
		revision = ""
	}
	return i.add(ModelSpec{LocalPath: root, Revision: revision, RequiredFiles: []string{file}, FilesOnly: true, CheckONNX: filepath.Ext(file) == ".onnx", Strict: true})
}

// Defaults choose their registered release; explicit bindings and standalone
// companions preserve the caller's revision intent, including an omitted pin.
func (i *modelInventory) addDefaultDeployment(cfg *config.RouterConfig, spec config.ResolvedModelBinding) error {
	path := config.ResolveModelPath(spec.Deployment.Artifact)
	repo, err := i.registeredRepo(path)
	if err != nil {
		return err
	}
	spec.Deployment.Revision = modelRevision(path, repo)
	return i.addDeployment(cfg, spec)
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
		next.RerankerSelections = append(previous.RerankerSelections, next.RerankerSelections...)
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
	if model := config.GetModelByPath(path); model != nil && model.RepoID == repoID && model.Revision != "" {
		return model.Revision
	}
	return "main"
}

func modelDownloadExcludePatterns(path, repoID string, runtimePatterns []string) []string {
	patterns := slices.Clone(runtimePatterns)
	if model := config.GetModelByPath(path); model != nil && model.RepoID == repoID {
		for _, pattern := range model.DownloadExcludePatterns {
			if !slices.Contains(patterns, pattern) {
				patterns = append(patterns, pattern)
			}
		}
	}
	return patterns
}
