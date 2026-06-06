package modellifecycle

import (
	"path/filepath"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	DefaultBERTEmbeddingModelPath = "models/mom-embedding-light"
)

type AssetRole string

const (
	RoleQwen3Embedding      AssetRole = "qwen3_embedding"
	RoleGemmaEmbedding      AssetRole = "gemma_embedding"
	RoleMmBERTEmbedding     AssetRole = "mmbert_embedding"
	RoleMultiModalEmbedding AssetRole = "multimodal_embedding"
	RoleBERTEmbedding       AssetRole = "bert_embedding"
	RoleDomainClassifier    AssetRole = "domain_classifier"
	RolePIIClassifier       AssetRole = "pii_classifier"
	RolePromptGuard         AssetRole = "prompt_guard"
	RoleFactCheckClassifier AssetRole = "fact_check_classifier"
	RoleHallucinationModel  AssetRole = "hallucination_detector"
	RoleHallucinationNLI    AssetRole = "hallucination_explainer"
	RoleFeedbackDetector    AssetRole = "feedback_detector"
	RoleModalityClassifier  AssetRole = "modality_classifier"
)

type Asset struct {
	Role          AssetRole
	LocalPath     string
	RequiredFiles []string
	UseCPU        bool
}

type EmbeddingPaths struct {
	Qwen3      string
	Gemma      string
	MmBERT     string
	MultiModal string
	BERT       string
	UseCPU     bool
}

type Plan struct {
	configured []Asset
	required   []Asset
}

func BuildPlan(cfg *config.RouterConfig) Plan {
	if cfg == nil {
		return Plan{}
	}

	builder := planBuilder{}
	builder.addEmbeddingAssets(cfg)
	builder.addClassifierAssets(cfg)
	builder.addModalityAssets(cfg)
	return builder.plan()
}

func (p Plan) ConfiguredAssets() []Asset {
	return cloneAssets(p.configured)
}

func (p Plan) AssetForRole(role AssetRole) (Asset, bool) {
	for _, asset := range p.configured {
		if asset.Role == role {
			return asset, true
		}
	}
	return Asset{}, false
}

func (p Plan) DownloadAssets() []Asset {
	downloadable := make([]Asset, 0, len(p.required))
	for _, asset := range p.required {
		if isDownloadableModelDirectory(asset.LocalPath) {
			downloadable = append(downloadable, asset)
		}
	}
	return downloadable
}

func (p Plan) ConfiguredModelPaths() []string {
	return assetPaths(p.configured, true)
}

func (p Plan) RequiredFilesByModel() map[string][]string {
	files := make(map[string][]string)
	for _, asset := range p.configured {
		if !isDownloadableModelDirectory(asset.LocalPath) {
			continue
		}
		for _, required := range asset.RequiredFiles {
			if required == "" || slices.Contains(files[asset.LocalPath], required) {
				continue
			}
			files[asset.LocalPath] = append(files[asset.LocalPath], required)
		}
	}
	return files
}

func (p Plan) EmbeddingPaths() EmbeddingPaths {
	return EmbeddingPaths{
		Qwen3:      p.pathForRole(RoleQwen3Embedding),
		Gemma:      p.pathForRole(RoleGemmaEmbedding),
		MmBERT:     p.pathForRole(RoleMmBERTEmbedding),
		MultiModal: p.pathForRole(RoleMultiModalEmbedding),
		BERT:       p.pathForRole(RoleBERTEmbedding),
		UseCPU:     p.embeddingUseCPU(),
	}
}

func (p Plan) pathForRole(role AssetRole) string {
	for _, asset := range p.configured {
		if asset.Role == role {
			return asset.LocalPath
		}
	}
	return ""
}

func (p Plan) embeddingUseCPU() bool {
	for _, role := range []AssetRole{
		RoleQwen3Embedding,
		RoleGemmaEmbedding,
		RoleMmBERTEmbedding,
		RoleMultiModalEmbedding,
		RoleBERTEmbedding,
	} {
		if asset, ok := p.AssetForRole(role); ok {
			return asset.UseCPU
		}
	}
	return false
}

type planBuilder struct {
	configured []Asset
	required   []Asset
}

func (b *planBuilder) addEmbeddingAssets(cfg *config.RouterConfig) {
	embeddings := cfg.EmbeddingModels
	b.addRequiredAsset(RoleQwen3Embedding, embeddings.Qwen3ModelPath, nil, embeddings.UseCPU)
	b.addRequiredAsset(RoleGemmaEmbedding, embeddings.GemmaModelPath, nil, embeddings.UseCPU)
	b.addRequiredAsset(RoleMmBERTEmbedding, embeddings.MmBertModelPath, nil, embeddings.UseCPU)
	b.addRequiredAsset(RoleMultiModalEmbedding, embeddings.MultiModalModelPath, nil, embeddings.UseCPU)

	bertPath := embeddings.BertModelPath
	if bertPath == "" && needsBERTEmbeddingRuntime(cfg) {
		bertPath = DefaultBERTEmbeddingModelPath
	}
	b.addRequiredAsset(RoleBERTEmbedding, bertPath, nil, embeddings.UseCPU)
}

func (b *planBuilder) addClassifierAssets(cfg *config.RouterConfig) {
	b.addOptionalAsset(
		RoleDomainClassifier,
		cfg.CategoryModel.ModelID,
		mappingRequiredFile(cfg.CategoryModel.ModelID, cfg.CategoryModel.CategoryMappingPath),
		cfg.CategoryModel.UseCPU,
		cfg.NeedsCategoryMappingForRouting(),
	)
	b.addOptionalAsset(
		RolePIIClassifier,
		cfg.PIIModel.ModelID,
		mappingRequiredFile(cfg.PIIModel.ModelID, cfg.PIIModel.PIIMappingPath),
		cfg.PIIModel.UseCPU,
		cfg.NeedsPIIMappingForRouting(),
	)
	b.addOptionalAsset(
		RolePromptGuard,
		cfg.PromptGuard.ModelID,
		mappingRequiredFile(cfg.PromptGuard.ModelID, cfg.PromptGuard.JailbreakMappingPath),
		cfg.PromptGuard.UseCPU,
		cfg.NeedsJailbreakMappingForRouting(),
	)
	b.addOptionalAsset(
		RoleFactCheckClassifier,
		cfg.HallucinationMitigation.FactCheckModel.ModelID,
		nil,
		cfg.HallucinationMitigation.FactCheckModel.UseCPU,
		cfg.IsFactCheckClassifierEnabled(),
	)
	b.addOptionalAsset(
		RoleHallucinationModel,
		cfg.HallucinationMitigation.HallucinationModel.ModelID,
		nil,
		cfg.HallucinationMitigation.HallucinationModel.UseCPU,
		cfg.IsHallucinationModelEnabled(),
	)
	b.addOptionalAsset(
		RoleHallucinationNLI,
		cfg.HallucinationMitigation.NLIModel.ModelID,
		nil,
		cfg.HallucinationMitigation.NLIModel.UseCPU,
		cfg.IsHallucinationModelEnabled(),
	)
	b.addOptionalAsset(
		RoleFeedbackDetector,
		cfg.FeedbackDetector.ModelID,
		mappingRequiredFile(cfg.FeedbackDetector.ModelID, cfg.FeedbackDetector.FeedbackMappingPath),
		cfg.FeedbackDetector.UseCPU,
		cfg.IsFeedbackDetectorEnabled(),
	)
}

func (b *planBuilder) addModalityAssets(cfg *config.RouterConfig) {
	md := cfg.ModalityDetector
	if md.Classifier == nil {
		return
	}

	method := md.GetMethod()
	enabled := md.Enabled &&
		md.Classifier.ModelPath != "" &&
		(method == config.ModalityDetectionClassifier || method == config.ModalityDetectionHybrid)
	b.addOptionalAsset(RoleModalityClassifier, md.Classifier.ModelPath, nil, md.Classifier.UseCPU, enabled)
}

func (b *planBuilder) addRequiredAsset(role AssetRole, path string, requiredFiles []string, useCPU bool) {
	b.addOptionalAsset(role, path, requiredFiles, useCPU, true)
}

func (b *planBuilder) addOptionalAsset(role AssetRole, path string, requiredFiles []string, useCPU bool, required bool) {
	asset := newAsset(role, path, requiredFiles, useCPU)
	if asset.LocalPath == "" {
		return
	}
	b.configured = upsertConfiguredAsset(b.configured, asset)
	if required {
		b.required = upsertRequiredAsset(b.required, asset)
	}
}

func (b *planBuilder) plan() Plan {
	return Plan{
		configured: cloneAssets(b.configured),
		required:   cloneAssets(b.required),
	}
}

func newAsset(role AssetRole, path string, requiredFiles []string, useCPU bool) Asset {
	return Asset{
		Role:          role,
		LocalPath:     config.ResolveModelPath(strings.TrimSpace(path)),
		RequiredFiles: cloneStrings(requiredFiles),
		UseCPU:        useCPU,
	}
}

func upsertConfiguredAsset(assets []Asset, next Asset) []Asset {
	for i := range assets {
		if assets[i].Role != next.Role {
			continue
		}
		assets[i] = next
		return assets
	}
	return append(assets, next)
}

func upsertRequiredAsset(assets []Asset, next Asset) []Asset {
	for i := range assets {
		if assets[i].LocalPath != next.LocalPath {
			continue
		}
		assets[i].RequiredFiles = mergeStrings(assets[i].RequiredFiles, next.RequiredFiles)
		return assets
	}
	return append(assets, next)
}

func mappingRequiredFile(modelPath, mappingPath string) []string {
	modelPath = config.ResolveModelPath(strings.TrimSpace(modelPath))
	mappingPath = strings.TrimSpace(mappingPath)
	if modelPath == "" || mappingPath == "" {
		return nil
	}
	if filepath.Dir(mappingPath) != modelPath {
		return nil
	}
	fileName := filepath.Base(mappingPath)
	if fileName == "." || fileName == "" {
		return nil
	}
	return []string{fileName}
}

func needsBERTEmbeddingRuntime(cfg *config.RouterConfig) bool {
	return semanticCacheNeedsBERT(cfg) || vectorStoreNeedsBERT(cfg) || memoryNeedsBERT(cfg)
}

func semanticCacheNeedsBERT(cfg *config.RouterConfig) bool {
	return cfg.Enabled && ResolveSemanticCacheEmbeddingModel(cfg) == "bert"
}

func vectorStoreNeedsBERT(cfg *config.RouterConfig) bool {
	if cfg.VectorStore == nil || !cfg.VectorStore.Enabled {
		return false
	}
	model := strings.ToLower(strings.TrimSpace(cfg.VectorStore.EmbeddingModel))
	return model == "" || model == "bert"
}

func memoryNeedsBERT(cfg *config.RouterConfig) bool {
	if !memoryEnabled(cfg) {
		return false
	}
	return ResolveMemoryEmbeddingModel(cfg) == "bert"
}

func memoryEnabled(cfg *config.RouterConfig) bool {
	if cfg.Memory.Enabled {
		return true
	}
	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			return true
		}
	}
	return false
}

func isDownloadableModelDirectory(path string) bool {
	return strings.HasPrefix(path, "models/") && filepath.Ext(filepath.Base(path)) == ""
}

func assetPaths(assets []Asset, downloadableOnly bool) []string {
	paths := make([]string, 0, len(assets))
	for _, asset := range assets {
		if downloadableOnly && !isDownloadableModelDirectory(asset.LocalPath) {
			continue
		}
		if slices.Contains(paths, asset.LocalPath) {
			continue
		}
		paths = append(paths, asset.LocalPath)
	}
	return paths
}

func cloneAssets(assets []Asset) []Asset {
	if len(assets) == 0 {
		return nil
	}
	cloned := make([]Asset, len(assets))
	copy(cloned, assets)
	for i := range cloned {
		cloned[i].RequiredFiles = cloneStrings(cloned[i].RequiredFiles)
	}
	return cloned
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	cloned := make([]string, len(values))
	copy(cloned, values)
	return cloned
}

func mergeStrings(base, extra []string) []string {
	result := cloneStrings(base)
	for _, value := range extra {
		if value == "" || slices.Contains(result, value) {
			continue
		}
		result = append(result, value)
	}
	return result
}
