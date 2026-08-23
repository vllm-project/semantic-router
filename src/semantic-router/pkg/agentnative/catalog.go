package agentnative

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	jsonschema "github.com/invopop/jsonschema"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	defaultCatalogPageSize = 10
	maximumCatalogPageSize = 20
)

// ConfigCatalog projects the authoritative Router configuration types into a
// deterministic, model-safe catalog. It contains no connection, credential,
// compiled identity, or Entrypoint assignment fields.
type ConfigCatalog struct {
	revision    string
	descriptors []ComponentDescriptor
}

func NewConfigCatalog() (*ConfigCatalog, error) {
	builder := catalogBuilder{}
	builder.addSignals()
	builder.addProjections()
	builder.addDecision()
	builder.addAlgorithms()
	builder.addPlugins()
	if builder.err != nil {
		return nil, builder.err
	}
	sort.Slice(builder.descriptors, func(left, right int) bool {
		if builder.descriptors[left].Kind != builder.descriptors[right].Kind {
			return builder.descriptors[left].Kind < builder.descriptors[right].Kind
		}
		return builder.descriptors[left].Name < builder.descriptors[right].Name
	})
	canonical, err := json.Marshal(builder.descriptors)
	if err != nil {
		return nil, fmt.Errorf("encode Router component catalog: %w", err)
	}
	digest := sha256.Sum256(canonical)
	return &ConfigCatalog{
		revision:    "sha256:" + hex.EncodeToString(digest[:]),
		descriptors: builder.descriptors,
	}, nil
}

func (catalog *ConfigCatalog) Describe(query CatalogQuery) (CatalogPage, error) {
	if catalog == nil || catalog.revision == "" ||
		(query.Kind != "" && !query.Kind.valid()) || strings.TrimSpace(query.Name) != query.Name ||
		len(query.Name) > 128 || query.Name != "" && query.Kind == "" {
		return CatalogPage{}, agentmanagement.ErrInvalid
	}
	pageSize := query.PageSize
	if pageSize == 0 {
		pageSize = defaultCatalogPageSize
	}
	if pageSize < 1 || pageSize > maximumCatalogPageSize {
		return CatalogPage{}, agentmanagement.ErrInvalid
	}
	matching := make([]ComponentDescriptor, 0, len(catalog.descriptors))
	for _, descriptor := range catalog.descriptors {
		if query.Kind != "" && descriptor.Kind != query.Kind || query.Name != "" && descriptor.Name != query.Name {
			continue
		}
		copyDescriptor := descriptor
		// Browsing returns a compact directory. A focused lookup returns the
		// complete current schema and still remains bounded to one component.
		if query.Kind == "" || query.Name == "" {
			copyDescriptor.Schema = nil
		} else {
			copyDescriptor.Schema = append(json.RawMessage(nil), descriptor.Schema...)
		}
		matching = append(matching, copyDescriptor)
	}
	offset, err := catalog.decodeCursor(query)
	if err != nil || offset > len(matching) {
		return CatalogPage{}, agentmanagement.ErrInvalid
	}
	end := offset + pageSize
	if end > len(matching) {
		end = len(matching)
	}
	page := CatalogPage{
		Revision: catalog.revision,
		Data:     append([]ComponentDescriptor{}, matching[offset:end]...),
		HasMore:  end < len(matching), PageSize: pageSize,
	}
	if page.HasMore {
		page.NextCursor, err = catalog.encodeCursor(query, end)
		if err != nil {
			return CatalogPage{}, err
		}
	}
	return page, nil
}

type catalogCursor struct {
	Revision string        `json:"revision"`
	Kind     ComponentKind `json:"kind,omitempty"`
	Name     string        `json:"name,omitempty"`
	Offset   int           `json:"offset"`
}

func (catalog *ConfigCatalog) encodeCursor(query CatalogQuery, offset int) (string, error) {
	payload, err := json.Marshal(catalogCursor{
		Revision: catalog.revision, Kind: query.Kind, Name: query.Name, Offset: offset,
	})
	if err != nil {
		return "", fmt.Errorf("encode Router component cursor: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(payload), nil
}

func (catalog *ConfigCatalog) decodeCursor(query CatalogQuery) (int, error) {
	if query.Cursor == "" {
		return 0, nil
	}
	payload, err := base64.RawURLEncoding.DecodeString(query.Cursor)
	if err != nil || len(payload) > 2048 {
		return 0, agentmanagement.ErrInvalid
	}
	var cursor catalogCursor
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cursor); err != nil || cursor.Revision != catalog.revision ||
		cursor.Kind != query.Kind || cursor.Name != query.Name || cursor.Offset < 1 {
		return 0, agentmanagement.ErrInvalid
	}
	return cursor.Offset, nil
}

type catalogBuilder struct {
	descriptors []ComponentDescriptor
	err         error
}

func (builder *catalogBuilder) add(
	kind ComponentKind, name, description string, sample any, metadata ...string,
) {
	if builder.err != nil {
		return
	}
	schema, err := reflectAuthoringSchema(sample)
	if err != nil {
		builder.err = fmt.Errorf("reflect %s %q schema: %w", kind, name, err)
		return
	}
	descriptor := ComponentDescriptor{
		Kind: kind, Name: name, Description: description, Schema: schema,
	}
	if len(metadata) > 0 {
		descriptor.Tier = metadata[0]
	}
	if len(metadata) > 1 {
		descriptor.Execution = metadata[1]
	}
	builder.descriptors = append(builder.descriptors, descriptor)
}

func (builder *catalogBuilder) addSignals() {
	signals := map[string]any{
		config.SignalTypeAuthz:        config.RoleBinding{},
		config.SignalTypeComplexity:   config.ComplexityRule{},
		config.SignalTypeContext:      config.ContextRule{},
		config.SignalTypeConversation: config.ConversationRule{},
		config.SignalTypeDomain:       domainSignalAuthoring{},
		config.SignalTypeEmbedding:    config.EmbeddingRule{},
		config.SignalTypeFactCheck:    config.FactCheckRule{},
		config.SignalTypeJailbreak:    config.JailbreakRule{},
		config.SignalTypeKeyword:      config.KeywordRule{},
		config.SignalTypeLanguage:     config.LanguageRule{},
		config.SignalTypeModality:     config.ModalityRule{},
		config.SignalTypePII:          config.PIIRule{},
		config.SignalTypePreference:   config.PreferenceRule{},
		config.SignalTypeReask:        config.ReaskRule{},
		config.SignalTypeStructure:    config.StructureRule{},
		config.SignalTypeKB:           config.KBSignalRule{},
		config.SignalTypeUserFeedback: config.UserFeedbackRule{},
		config.SignalTypeEvent:        config.EventRule{},
		config.SignalTypeMetadata:     config.MetadataRule{},
		config.SignalTypeClassifier:   classifierSignalAuthoring{},
	}
	for _, name := range config.SupportedSignalTypes() {
		sample, found := signals[name]
		if !found {
			builder.err = fmt.Errorf("router signal %q has no authoring schema", name)
			return
		}
		builder.add(ComponentSignal, name, "Detects one request property for Recipe decisions.", sample)
	}
}

func (builder *catalogBuilder) addProjections() {
	builder.add(ComponentProjection, "partition", "Builds an exclusive partition from existing signals.", config.ProjectionPartition{})
	builder.add(ComponentProjection, "score", "Combines signal values into a continuous score.", config.ProjectionScore{})
	builder.add(ComponentProjection, "mapping", "Maps a score into named decision outputs.", config.ProjectionMapping{})
}

func (builder *catalogBuilder) addDecision() {
	builder.add(ComponentDecision, "decision", "Matches a signal expression and selects one execution algorithm.", decisionAuthoring{})
}

func (builder *catalogBuilder) addAlgorithms() {
	samples := map[string]any{
		config.DecisionAlgorithmAutoMix:      config.AutoMixSelectionConfig{},
		config.DecisionAlgorithmConfidence:   confidenceAlgorithmAuthoring{},
		config.DecisionAlgorithmFusion:       fusionAlgorithmAuthoring{},
		config.DecisionAlgorithmHybrid:       config.HybridSelectionConfig{},
		config.DecisionAlgorithmKMeans:       mlKMeansAlgorithmAuthoring{},
		config.DecisionAlgorithmKNN:          mlKNNAlgorithmAuthoring{},
		config.DecisionAlgorithmLatencyAware: config.LatencyAwareAlgorithmConfig{},
		config.DecisionAlgorithmMLP:          mlMLPAlgorithmAuthoring{},
		config.DecisionAlgorithmMultiFactor:  config.MultiFactorSelectionConfig{},
		config.DecisionAlgorithmRatings:      config.RatingsAlgorithmConfig{},
		config.DecisionAlgorithmReMoM:        reMoMAlgorithmAuthoring{},
		config.DecisionAlgorithmRouterDC:     config.RouterDCSelectionConfig{},
		config.DecisionAlgorithmStatic:       struct{}{},
		config.DecisionAlgorithmSVM:          mlSVMAlgorithmAuthoring{},
		config.DecisionAlgorithmWorkflows:    workflowsAlgorithmAuthoring{},
		config.DecisionAlgorithmPrompt:       promptAlgorithmAuthoring{},
	}
	for _, entry := range config.DecisionAlgorithmCatalog() {
		sample, found := samples[entry.Type]
		if !found {
			builder.err = fmt.Errorf("router algorithm %q has no authoring schema", entry.Type)
			return
		}
		builder.add(
			ComponentAlgorithm, entry.Type, "Controls how assigned Models execute after a Decision matches.",
			sample, entry.Tier, string(entry.Execution),
		)
	}
}

func (builder *catalogBuilder) addPlugins() {
	samples := map[string]any{
		config.DecisionPluginResponseCache:      responseCacheAuthoring{},
		config.DecisionPluginSystemPrompt:       config.SystemPromptPluginConfig{},
		config.DecisionPluginHeaderMutation:     config.HeaderMutationPluginConfig{},
		config.DecisionPluginHallucination:      config.HallucinationPluginConfig{},
		config.DecisionPluginResponseJailbreak:  config.ResponseJailbreakPluginConfig{},
		config.DecisionPluginRouterReplay:       config.RouterReplayPluginConfig{},
		config.DecisionPluginMemory:             config.MemoryPluginConfig{},
		config.DecisionPluginRAG:                ragPluginAuthoring{},
		config.DecisionPluginImageGen:           imageGenPluginAuthoring{},
		config.DecisionPluginFastResponse:       config.FastResponsePluginConfig{},
		config.DecisionPluginRequestParams:      config.RequestParamsPluginConfig{},
		config.DecisionPluginToolSelection:      toolSelectionPluginAuthoring{},
		config.DecisionPluginContextCompression: contextCompressionPluginAuthoring{},
		config.DecisionPluginTools:              config.ToolsPluginConfig{},
	}
	for _, name := range config.SupportedDecisionPluginTypes() {
		sample, found := samples[name]
		if !found {
			builder.err = fmt.Errorf("router plugin %q has no authoring schema", name)
			return
		}
		builder.add(ComponentPlugin, name, "Adds bounded decision-scoped request or response behavior.", sample)
	}
}

func reflectAuthoringSchema(sample any) (json.RawMessage, error) {
	reflector := jsonschema.Reflector{
		Anonymous: true, FieldNameTag: "yaml",
		AllowAdditionalProperties: false,
	}
	schema := reflector.Reflect(sample)
	encoded, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var root map[string]any
	if err := json.Unmarshal(encoded, &root); err != nil {
		return nil, err
	}
	delete(root, "$id")
	return json.Marshal(root)
}

// The following authoring views deliberately omit physical Model selection.
// Entrypoint assignments own those fields and materialize them at publication.
type domainSignalAuthoring struct {
	Name           string   `yaml:"name" json:"name"`
	Description    string   `yaml:"description,omitempty" json:"description,omitempty"`
	MMLUCategories []string `yaml:"mmlu_categories,omitempty" json:"mmlu_categories,omitempty"`
}

// Classifier assets and LLM bindings are deployment state. The Recipe may
// describe classifier behavior, but an Agent must never learn a local path or
// a physical Model name through the component catalog.
type classifierSignalAuthoring struct {
	Name         string   `yaml:"name" json:"name"`
	Description  string   `yaml:"description,omitempty" json:"description,omitempty"`
	Type         string   `yaml:"type" json:"type"`
	Labels       []string `yaml:"labels" json:"labels"`
	Instructions string   `yaml:"instructions,omitempty" json:"instructions,omitempty"`
	UseCPU       bool     `yaml:"use_cpu,omitempty" json:"use_cpu,omitempty"`
}

type decisionAuthoring struct {
	Name                string                           `yaml:"name" json:"name"`
	Description         string                           `yaml:"description,omitempty" json:"description,omitempty"`
	Priority            int                              `yaml:"priority,omitempty" json:"priority,omitempty"`
	Tier                int                              `yaml:"tier,omitempty" json:"tier,omitempty"`
	OutputContract      string                           `yaml:"output_contract,omitempty" json:"output_contract,omitempty"`
	OutputContractSpec  *config.OutputContractSpec       `yaml:"output_contract_spec,omitempty" json:"output_contract_spec,omitempty"`
	Rules               config.RuleCombination           `yaml:"rules" json:"rules"`
	Algorithm           *algorithmAuthoring              `yaml:"algorithm,omitempty" json:"algorithm,omitempty"`
	Adaptations         config.DecisionAdaptationsConfig `yaml:"adaptations,omitempty" json:"adaptations,omitempty"`
	Plugins             []pluginAuthoring                `yaml:"plugins,omitempty" json:"plugins,omitempty"`
	CandidateIterations []candidateIterationAuthoring    `yaml:"candidateIterations,omitempty" json:"candidateIterations,omitempty"`
	Emits               []config.EmitDirective           `yaml:"emits,omitempty" json:"emits,omitempty"`
	Annotations         map[string]any                   `yaml:"annotations,omitempty" json:"annotations,omitempty"`
}

type algorithmAuthoring struct {
	Type         string                              `yaml:"type" json:"type"`
	Confidence   *confidenceAlgorithmAuthoring       `yaml:"confidence,omitempty" json:"confidence,omitempty"`
	Ratings      *config.RatingsAlgorithmConfig      `yaml:"ratings,omitempty" json:"ratings,omitempty"`
	ReMoM        *reMoMAlgorithmAuthoring            `yaml:"remom,omitempty" json:"remom,omitempty"`
	Fusion       *fusionAlgorithmAuthoring           `yaml:"fusion,omitempty" json:"fusion,omitempty"`
	Workflows    *workflowsAlgorithmAuthoring        `yaml:"workflows,omitempty" json:"workflows,omitempty"`
	RouterDC     *config.RouterDCSelectionConfig     `yaml:"router_dc,omitempty" json:"router_dc,omitempty"`
	AutoMix      *config.AutoMixSelectionConfig      `yaml:"automix,omitempty" json:"automix,omitempty"`
	Hybrid       *config.HybridSelectionConfig       `yaml:"hybrid,omitempty" json:"hybrid,omitempty"`
	ML           *mlAlgorithmAuthoring               `yaml:"ml,omitempty" json:"ml,omitempty"`
	LatencyAware *config.LatencyAwareAlgorithmConfig `yaml:"latency_aware,omitempty" json:"latency_aware,omitempty"`
	MultiFactor  *config.MultiFactorSelectionConfig  `yaml:"multi_factor,omitempty" json:"multi_factor,omitempty"`
	Prompt       *promptAlgorithmAuthoring           `yaml:"prompt,omitempty" json:"prompt,omitempty"`
	OnError      string                              `yaml:"on_error,omitempty" json:"on_error,omitempty"`
}

// The verifier endpoint is process-level connectivity. It is deliberately
// absent even though the runtime config type contains one.
type confidenceAlgorithmAuthoring struct {
	ConfidenceMethod    string                      `yaml:"confidence_method,omitempty" json:"confidence_method,omitempty"`
	Threshold           float64                     `yaml:"threshold,omitempty" json:"threshold,omitempty"`
	HybridWeights       *config.HybridWeightsConfig `yaml:"hybrid_weights,omitempty" json:"hybrid_weights,omitempty"`
	OnError             string                      `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	EscalationOrder     string                      `yaml:"escalation_order,omitempty" json:"escalation_order,omitempty"`
	CostQualityTradeoff float64                     `yaml:"cost_quality_tradeoff,omitempty" json:"cost_quality_tradeoff,omitempty"`
	TokenFilter         string                      `yaml:"token_filter,omitempty" json:"token_filter,omitempty"`
}

// Artifact paths are operator-owned runtime state. These views retain only
// the portable tuning values that belong in a model-free Recipe.
type mlAlgorithmAuthoring struct {
	EmbeddingDim int                      `yaml:"embedding_dim,omitempty" json:"embedding_dim,omitempty"`
	KNN          *mlKNNConfigAuthoring    `yaml:"knn,omitempty" json:"knn,omitempty"`
	KMeans       *mlKMeansConfigAuthoring `yaml:"kmeans,omitempty" json:"kmeans,omitempty"`
	SVM          *mlSVMConfigAuthoring    `yaml:"svm,omitempty" json:"svm,omitempty"`
	MLP          *mlMLPConfigAuthoring    `yaml:"mlp,omitempty" json:"mlp,omitempty"`
}

type mlKNNAlgorithmAuthoring struct {
	EmbeddingDim int                   `yaml:"embedding_dim,omitempty" json:"embedding_dim,omitempty"`
	KNN          *mlKNNConfigAuthoring `yaml:"knn" json:"knn"`
}

type mlKMeansAlgorithmAuthoring struct {
	EmbeddingDim int                      `yaml:"embedding_dim,omitempty" json:"embedding_dim,omitempty"`
	KMeans       *mlKMeansConfigAuthoring `yaml:"kmeans" json:"kmeans"`
}

type mlSVMAlgorithmAuthoring struct {
	EmbeddingDim int                   `yaml:"embedding_dim,omitempty" json:"embedding_dim,omitempty"`
	SVM          *mlSVMConfigAuthoring `yaml:"svm" json:"svm"`
}

type mlMLPAlgorithmAuthoring struct {
	EmbeddingDim int                   `yaml:"embedding_dim,omitempty" json:"embedding_dim,omitempty"`
	MLP          *mlMLPConfigAuthoring `yaml:"mlp" json:"mlp"`
}

type mlKNNConfigAuthoring struct {
	K int `yaml:"k,omitempty" json:"k,omitempty"`
}

type mlKMeansConfigAuthoring struct {
	NumClusters      int     `yaml:"num_clusters,omitempty" json:"num_clusters,omitempty"`
	EfficiencyWeight float64 `yaml:"efficiency_weight,omitempty" json:"efficiency_weight,omitempty"`
}

type mlSVMConfigAuthoring struct {
	Kernel string  `yaml:"kernel,omitempty" json:"kernel,omitempty"`
	Gamma  float64 `yaml:"gamma,omitempty" json:"gamma,omitempty"`
}

type mlMLPConfigAuthoring struct {
	Device string `yaml:"device,omitempty" json:"device,omitempty"`
}

type pluginAuthoring struct {
	Type          string         `yaml:"type" json:"type"`
	Configuration map[string]any `yaml:"configuration" json:"configuration"`
}

type candidateIterationAuthoring struct {
	Variable string                                  `yaml:"variable" json:"variable"`
	Source   string                                  `yaml:"source" json:"source"`
	Outputs  []config.CandidateIterationOutputConfig `yaml:"outputs,omitempty" json:"outputs,omitempty"`
}

type fusionAlgorithmAuthoring struct {
	MaxConcurrent                int                           `yaml:"max_concurrent,omitempty" json:"max_concurrent,omitempty"`
	MaxCompletionTokens          int                           `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int                           `yaml:"round_timeout_seconds,omitempty" json:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int                           `yaml:"min_successful_responses,omitempty" json:"min_successful_responses,omitempty"`
	Temperature                  *float64                      `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeAnalysis              *bool                         `yaml:"include_analysis,omitempty" json:"include_analysis,omitempty"`
	OnError                      string                        `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	AnalysisTemplate             string                        `yaml:"analysis_template,omitempty" json:"analysis_template,omitempty"`
	SynthesisTemplate            string                        `yaml:"synthesis_template,omitempty" json:"synthesis_template,omitempty"`
	JudgePromptVersion           string                        `yaml:"judge_prompt_version,omitempty" json:"judge_prompt_version,omitempty"`
	IncludeIntermediateResponses *bool                         `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
	Grounding                    *config.FusionGroundingConfig `yaml:"grounding,omitempty" json:"grounding,omitempty"`
}

type reMoMAlgorithmAuthoring struct {
	BreadthSchedule              []int   `yaml:"breadth_schedule" json:"breadth_schedule"`
	ModelDistribution            string  `yaml:"model_distribution,omitempty" json:"model_distribution,omitempty"`
	Temperature                  float64 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeReasoning             bool    `yaml:"include_reasoning,omitempty" json:"include_reasoning,omitempty"`
	CompactionStrategy           string  `yaml:"compaction_strategy,omitempty" json:"compaction_strategy,omitempty"`
	CompactionTokens             int     `yaml:"compaction_tokens,omitempty" json:"compaction_tokens,omitempty"`
	SynthesisTemplate            string  `yaml:"synthesis_template,omitempty" json:"synthesis_template,omitempty"`
	MaxConcurrent                int     `yaml:"max_concurrent,omitempty" json:"max_concurrent,omitempty"`
	MaxCompletionTokens          *int    `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int     `yaml:"round_timeout_seconds,omitempty" json:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int     `yaml:"min_successful_responses,omitempty" json:"min_successful_responses,omitempty"`
	OnError                      string  `yaml:"on_error,omitempty" json:"on_error,omitempty"`
	ShuffleSeed                  int     `yaml:"shuffle_seed,omitempty" json:"shuffle_seed,omitempty"`
	IncludeIntermediateResponses bool    `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
	MaxResponsesPerRound         int     `yaml:"max_responses_per_round,omitempty" json:"max_responses_per_round,omitempty"`
}

type workflowsAlgorithmAuthoring struct {
	Mode                         string                   `yaml:"mode,omitempty" json:"mode,omitempty"`
	Template                     string                   `yaml:"template,omitempty" json:"template,omitempty"`
	Roles                        []workflowRoleAuthoring  `yaml:"roles,omitempty" json:"roles,omitempty"`
	Final                        workflowFinalAuthoring   `yaml:"final,omitempty" json:"final,omitempty"`
	Planner                      workflowPlannerAuthoring `yaml:"planner,omitempty" json:"planner,omitempty"`
	MaxSteps                     int                      `yaml:"max_steps,omitempty" json:"max_steps,omitempty"`
	MaxParallel                  int                      `yaml:"max_parallel,omitempty" json:"max_parallel,omitempty"`
	MaxCompletionTokens          int                      `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
	RoundTimeoutSeconds          int                      `yaml:"round_timeout_seconds,omitempty" json:"round_timeout_seconds,omitempty"`
	MinSuccessfulResponses       int                      `yaml:"min_successful_responses,omitempty" json:"min_successful_responses,omitempty"`
	Temperature                  *float64                 `yaml:"temperature,omitempty" json:"temperature,omitempty"`
	IncludeIntermediateResponses *bool                    `yaml:"include_intermediate_responses,omitempty" json:"include_intermediate_responses,omitempty"`
	OnError                      string                   `yaml:"on_error,omitempty" json:"on_error,omitempty"`
}

type workflowRoleAuthoring struct {
	Name       string   `yaml:"name,omitempty" json:"name,omitempty"`
	Prompt     string   `yaml:"prompt,omitempty" json:"prompt,omitempty"`
	AccessList []string `yaml:"access_list,omitempty" json:"access_list,omitempty"`
}

type workflowFinalAuthoring struct {
	Prompt string `yaml:"prompt,omitempty" json:"prompt,omitempty"`
}

type workflowPlannerAuthoring struct {
	MaxCompletionTokens int `yaml:"max_completion_tokens,omitempty" json:"max_completion_tokens,omitempty"`
}

type promptAlgorithmAuthoring struct {
	Instructions   string `yaml:"instructions" json:"instructions"`
	TimeoutSeconds int    `yaml:"timeout_seconds,omitempty" json:"timeout_seconds,omitempty"`
}

type responseCacheAuthoring struct {
	Enabled         bool                                       `yaml:"enabled" json:"enabled"`
	Mode            string                                     `yaml:"mode,omitempty" json:"mode,omitempty"`
	Scope           string                                     `yaml:"scope,omitempty" json:"scope,omitempty"`
	Semantic        *config.ResponseCacheSemanticConfig        `yaml:"semantic,omitempty" json:"semantic,omitempty"`
	RequestControls *config.ResponseCacheRequestControlsConfig `yaml:"request_controls,omitempty" json:"request_controls,omitempty"`
	Personalized    *config.ResponseCachePersonalizedConfig    `yaml:"personalized,omitempty" json:"personalized,omitempty"`
}

// Backend connection payloads are compiled by the control plane and are not
// portable Recipe authoring fields.
type ragPluginAuthoring struct {
	Enabled                bool     `yaml:"enabled" json:"enabled"`
	Backend                string   `yaml:"backend" json:"backend"`
	SimilarityThreshold    *float32 `yaml:"similarity_threshold,omitempty" json:"similarity_threshold,omitempty"`
	TopK                   *int     `yaml:"top_k,omitempty" json:"top_k,omitempty"`
	MaxContextLength       *int     `yaml:"max_context_length,omitempty" json:"max_context_length,omitempty"`
	InjectionMode          string   `yaml:"injection_mode,omitempty" json:"injection_mode,omitempty"`
	OnFailure              string   `yaml:"on_failure,omitempty" json:"on_failure,omitempty"`
	CacheResults           bool     `yaml:"cache_results,omitempty" json:"cache_results,omitempty"`
	CacheTTLSeconds        *int     `yaml:"cache_ttl_seconds,omitempty" json:"cache_ttl_seconds,omitempty"`
	MinConfidenceThreshold *float32 `yaml:"min_confidence_threshold,omitempty" json:"min_confidence_threshold,omitempty"`
}

type imageGenPluginAuthoring struct {
	Enabled           bool                        `yaml:"enabled" json:"enabled"`
	Backend           string                      `yaml:"backend" json:"backend"`
	ModalityDetection *modalityDetectionAuthoring `yaml:"modality_detection,omitempty" json:"modality_detection,omitempty"`
	DefaultWidth      int                         `yaml:"default_width,omitempty" json:"default_width,omitempty"`
	DefaultHeight     int                         `yaml:"default_height,omitempty" json:"default_height,omitempty"`
	MaxInferenceSteps int                         `yaml:"max_inference_steps,omitempty" json:"max_inference_steps,omitempty"`
	TimeoutSeconds    int                         `yaml:"timeout_seconds,omitempty" json:"timeout_seconds,omitempty"`
}

type modalityDetectionAuthoring struct {
	Method              string   `yaml:"method,omitempty" json:"method,omitempty"`
	Keywords            []string `yaml:"keywords,omitempty" json:"keywords,omitempty"`
	BothKeywords        []string `yaml:"both_keywords,omitempty" json:"both_keywords,omitempty"`
	ConfidenceThreshold float32  `yaml:"confidence_threshold,omitempty" json:"confidence_threshold,omitempty"`
	LowerThresholdRatio float32  `yaml:"lower_threshold_ratio,omitempty" json:"lower_threshold_ratio,omitempty"`
}

type toolSelectionPluginAuthoring struct {
	Enabled             bool                                `yaml:"enabled" json:"enabled"`
	Mode                string                              `yaml:"mode,omitempty" json:"mode,omitempty"`
	TopK                int                                 `yaml:"top_k,omitempty" json:"top_k,omitempty"`
	SimilarityThreshold *float32                            `yaml:"similarity_threshold,omitempty" json:"similarity_threshold,omitempty"`
	AdvancedFiltering   *config.AdvancedToolFilteringConfig `yaml:"advanced_filtering,omitempty" json:"advanced_filtering,omitempty"`
	Strategy            string                              `yaml:"strategy,omitempty" json:"strategy,omitempty"`
	FallbackToEmpty     *bool                               `yaml:"fallback_to_empty,omitempty" json:"fallback_to_empty,omitempty"`
	RelevanceThreshold  *float32                            `yaml:"relevance_threshold,omitempty" json:"relevance_threshold,omitempty"`
	PreserveCount       int                                 `yaml:"preserve_count,omitempty" json:"preserve_count,omitempty"`
}

type contextCompressionPluginAuthoring struct {
	Enabled         bool                                            `yaml:"enabled" json:"enabled"`
	Mode            string                                          `yaml:"mode,omitempty" json:"mode,omitempty"`
	Budget          *contextCompressionBudgetAuthoring              `yaml:"budget,omitempty" json:"budget,omitempty"`
	Targets         *config.ContextCompressionTargetsConfig         `yaml:"targets,omitempty" json:"targets,omitempty"`
	Scoring         *contextCompressionScoringAuthoring             `yaml:"scoring,omitempty" json:"scoring,omitempty"`
	Recovery        *config.ContextCompressionRecoveryConfig        `yaml:"recovery,omitempty" json:"recovery,omitempty"`
	RequestControls *config.ContextCompressionRequestControlsConfig `yaml:"request_controls,omitempty" json:"request_controls,omitempty"`
	FailureMode     string                                          `yaml:"failure_mode,omitempty" json:"failure_mode,omitempty"`
}

// Compression limits are scalar values in the public document. Using an
// interface here describes that scalar union without reflecting the private
// representation fields of config.CompressionTokenLimit.
type contextCompressionBudgetAuthoring struct {
	TriggerTokens       *compressionLimitAuthoring `yaml:"trigger_tokens,omitempty" json:"trigger_tokens,omitempty"`
	TargetTokens        *compressionLimitAuthoring `yaml:"target_tokens,omitempty" json:"target_tokens,omitempty"`
	ReserveOutputTokens *compressionLimitAuthoring `yaml:"reserve_output_tokens,omitempty" json:"reserve_output_tokens,omitempty"`
}

type compressionLimitAuthoring struct{}

func (compressionLimitAuthoring) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{OneOf: []*jsonschema.Schema{
		{Type: "string", Enum: []any{"auto"}},
		{Type: "integer", Minimum: json.Number("0")},
	}}
}

type contextCompressionScoringAuthoring struct {
	Method string `yaml:"method,omitempty" json:"method,omitempty"`
}

var _ CatalogSource = (*ConfigCatalog)(nil)
