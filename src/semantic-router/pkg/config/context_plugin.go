package config

import "strings"

// The context plugin's source types, RAG failure policies and injection
// modes.
const (
	DecisionPluginContext = "context"

	ContextSourceRAG    = "rag"
	ContextSourceMemory = "memory"

	RAGOnFailureSkip  = "skip"
	RAGOnFailureBlock = "block"
	RAGOnFailureWarn  = "warn"

	ContextInjectionToolRole     = "tool_role"
	ContextInjectionSystemPrompt = "system_prompt"
)

// ContextPluginConfig assembles a decision's request context. Its sources
// retrieve documents and memories, each kept by its match block; compress
// shrinks the request's own segments, judged by its match block, within the
// budget, and the text the routing signals read; injection places what the
// sources kept. A memory source also writes the turn to memory after the
// response.
type ContextPluginConfig struct {
	// Enabled turns the plugin on; false turns context off for the
	// decision, global memory included.
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Sources are retrieved in order; at most one of each type.
	Sources   []ContextSource         `json:"sources,omitempty" yaml:"sources,omitempty"`
	Compress  *ContextCompressConfig  `json:"compress,omitempty" yaml:"compress,omitempty"`
	Budget    *ContextBudgetConfig    `json:"budget,omitempty" yaml:"budget,omitempty"`
	Injection *ContextInjectionConfig `json:"injection,omitempty" yaml:"injection,omitempty"`
}

// ContextSource retrieves context from a RAG backend or from the user's
// memory. The backend fields belong to a rag source, the store fields to a
// memory source.
type ContextSource struct {
	Type string `json:"type" yaml:"type" jsonschema:"required,enum=rag,enum=memory"`
	// Match keeps the retrieved items: a rag source recalls and judges
	// relevance by default, a memory source recalls with the memory store's
	// default_retrieval_limit and default_similarity_threshold.
	Match *MatchConfig `json:"match,omitempty" yaml:"match,omitempty"`

	Backend string `json:"backend,omitempty" yaml:"backend,omitempty" jsonschema:"enum=vectorstore,enum=milvus,enum=qdrant,enum=external_api,enum=mcp,enum=openai,enum=hybrid"`
	// BackendConfig is the backend's own configuration (MilvusRAGConfig,
	// QdrantRAGConfig, ExternalAPIRAGConfig, MCPRAGConfig, OpenAIRAGConfig,
	// HybridRAGConfig or VectorStoreRAGConfig).
	BackendConfig *StructuredPayload `json:"backend_config,omitempty" yaml:"backend_config,omitempty"`
	// MaxContextLength truncates the injected documents to this many
	// characters (default 10000).
	MaxContextLength *int `json:"max_context_length,omitempty" yaml:"max_context_length,omitempty"`
	// OnFailure is what a failed retrieval does: skip (the default) goes on
	// without the documents, warn also logs a warning, block fails the
	// request.
	OnFailure       string `json:"on_failure,omitempty" yaml:"on_failure,omitempty" jsonschema:"enum=skip,enum=block,enum=warn"`
	CacheResults    bool   `json:"cache_results,omitempty" yaml:"cache_results,omitempty"`
	CacheTTLSeconds *int   `json:"cache_ttl_seconds,omitempty" yaml:"cache_ttl_seconds,omitempty"`

	Store *ContextMemoryStoreConfig `json:"store,omitempty" yaml:"store,omitempty"`
	// HybridSearch fuses the memory store's vector and lexical scores, by
	// HybridMode (weighted or rrf).
	HybridSearch bool                    `json:"hybrid_search,omitempty" yaml:"hybrid_search,omitempty"`
	HybridMode   string                  `json:"hybrid_mode,omitempty" yaml:"hybrid_mode,omitempty" jsonschema:"enum=weighted,enum=rrf"`
	Reflection   *MemoryReflectionConfig `json:"reflection,omitempty" yaml:"reflection,omitempty"`
}

// ContextMemoryStoreConfig says whether a memory source writes the turn to
// memory after the response.
type ContextMemoryStoreConfig struct {
	// Auto writes memories after the response; unset follows
	// global.stores.memory.auto_store. A request may turn the write off, and
	// on only where Auto is unset.
	Auto *bool `json:"auto,omitempty" yaml:"auto,omitempty"`
}

// ContextCompressConfig shrinks the request's own segments: tool outputs,
// history and the injected documents and memories. Its match block judges
// which segments matter, relevance by default; the least relevant shrink
// first, until the request fits the budget. Signals compresses the text the
// routing signals read; a block with only signals leaves the request whole.
type ContextCompressConfig struct {
	// Mode auto compresses a request near the budget or with segments over
	// their target's min_tokens; always compresses every request.
	Mode            string                                   `json:"mode,omitempty" yaml:"mode,omitempty" jsonschema:"enum=auto,enum=always"`
	Targets         *ContextCompressionTargetsConfig         `json:"targets,omitempty" yaml:"targets,omitempty"`
	Match           *MatchConfig                             `json:"match,omitempty" yaml:"match,omitempty"`
	Recovery        *ContextCompressionRecoveryConfig        `json:"recovery,omitempty" yaml:"recovery,omitempty"`
	RequestControls *ContextCompressionRequestControlsConfig `json:"request_controls,omitempty" yaml:"request_controls,omitempty"`
	// FailureMode fail_open (the default) sends the request uncompressed
	// when compression fails; fail_closed refuses it.
	FailureMode string `json:"failure_mode,omitempty" yaml:"failure_mode,omitempty" jsonschema:"enum=fail_open,enum=fail_closed"`
	// Signals runs before the decision is chosen, so it is read from
	// global.plugins and the recipe's plugins; a decision's own refuses it.
	Signals *PromptCompressionConfig `json:"signals,omitempty" yaml:"signals,omitempty"`
}

// ContextBudgetConfig bounds the provider-bound request that compress
// produces.
type ContextBudgetConfig struct {
	// MaxTokens is the most input tokens the request keeps: a request over
	// it is compressed down to it. auto (the default) is the model's context
	// window less the reserved output.
	MaxTokens *CompressionTokenLimit `json:"max_tokens,omitempty" yaml:"max_tokens,omitempty"`
	// ReserveOutputTokens is the output the window keeps free; auto (the
	// default) is the request's output limit.
	ReserveOutputTokens *CompressionTokenLimit `json:"reserve_output_tokens,omitempty" yaml:"reserve_output_tokens,omitempty"`
}

// ContextInjectionConfig places the retrieved context: tool_role (the
// default) adds one tool call and result per source after the last user
// message; system_prompt prepends it to the system prompt.
type ContextInjectionConfig struct {
	Mode string `json:"mode,omitempty" yaml:"mode,omitempty" jsonschema:"enum=tool_role,enum=system_prompt"`
}

// GetContextConfig returns the decision's context plugin configuration.
func (d *Decision) GetContextConfig() *ContextPluginConfig {
	return decodeDecisionPlugin(d, DecisionPluginContext, &ContextPluginConfig{})
}

// ContextPlugin is the context a request assembles: the context plugin it
// runs (RequestPlugins) when enabled; nil without one or when disabled.
func (c *RouterConfig) ContextPlugin(decision *Decision) *ContextPluginConfig {
	return ContextPluginOf(c.RequestPlugins(decision))
}

// ContextPluginOf is the enabled context plugin among a request's plugins.
func ContextPluginOf(plugins []DecisionPlugin) *ContextPluginConfig {
	plugin := decodePlugin(plugins, DecisionPluginContext, &ContextPluginConfig{})
	if plugin == nil || !plugin.Enabled {
		return nil
	}
	return plugin
}

// SignalsCompression is the compress.signals the requests of a recipe run
// before their decision: that of the context plugin of global.plugins layered
// with the recipe's plugins; nil when it compresses nothing.
func (r *RoutingRecipe) SignalsCompression() *PromptCompressionConfig {
	if r == nil {
		return nil
	}
	plugin := ContextPluginOf(r.Profile.DefaultPlugins)
	if plugin == nil || plugin.Compress == nil {
		return nil
	}
	if signals := plugin.Compress.Signals; signals != nil && signals.Enabled && signals.MaxTokens > 0 {
		return signals
	}
	return nil
}

// Source returns the plugin's source of a type, nil without one.
func (c *ContextPluginConfig) Source(sourceType string) *ContextSource {
	if c == nil {
		return nil
	}
	for index := range c.Sources {
		if c.Sources[index].Type == sourceType {
			return &c.Sources[index]
		}
	}
	return nil
}

// InjectionMode is where the sources' context goes.
func (c *ContextPluginConfig) InjectionMode() string {
	if c == nil || c.Injection == nil || c.Injection.Mode == "" {
		return ContextInjectionToolRole
	}
	return c.Injection.Mode
}

// IsRAGEnabledForDecision reports whether a decision retrieves documents.
func (c *RouterConfig) IsRAGEnabledForDecision(decisionName string) bool {
	return c.ContextPlugin(c.GetDecisionByName(decisionName)).Source(ContextSourceRAG) != nil
}

// IsMemoryEnabledForDecision reports whether a decision retrieves memories.
func (c *RouterConfig) IsMemoryEnabledForDecision(decisionName string) bool {
	return c.ContextPlugin(c.GetDecisionByName(decisionName)).Source(ContextSourceMemory) != nil
}

// HasPersonalizationPlugins reports whether a decision's context retrieves
// documents or memories, which a cached response would miss.
func (c *RouterConfig) HasPersonalizationPlugins(decisionName string) bool {
	plugin := c.ContextPlugin(c.GetDecisionByName(decisionName))
	return plugin != nil && len(plugin.Sources) > 0
}

// UsesMemory reports whether a request of any recipe retrieves memories, which
// the memory store is created for.
func (c *RouterConfig) UsesMemory() bool {
	if c == nil {
		return false
	}
	uses := false
	c.visitEveryPlugins(func(plugins []DecisionPlugin) {
		uses = uses || ContextPluginOf(plugins).Source(ContextSourceMemory) != nil
	})
	return uses
}

// usesReachableMemory reports whether a reachable request retrieves memories,
// which the shared embedding services are prepared for.
func (c *RouterConfig) usesReachableMemory() bool {
	uses := false
	c.visitReachablePlugins(func(plugins []DecisionPlugin) {
		uses = uses || ContextPluginOf(plugins).Source(ContextSourceMemory) != nil
	})
	return uses
}

// MemoryMatch is the match block a memory source runs: its own, else recall
// bounded by the memory store's default_retrieval_limit and
// default_similarity_threshold, whose own defaults are the memory row of the
// match defaults.
func (c *RouterConfig) MemoryMatch(source *ContextSource) MatchConfig {
	if source != nil && source.Match != nil {
		return MatchMemory.Effective(source.Match)
	}
	block := MatchMemory.Default()
	if c != nil && c.Memory.DefaultRetrievalLimit > 0 {
		block.Recall.TopK = c.Memory.DefaultRetrievalLimit
	}
	if c != nil && c.Memory.DefaultSimilarityThreshold > 0 {
		block.Recall.MinSimilarity = matchBound(float64(c.Memory.DefaultSimilarityThreshold))
	}
	return block
}

// MatchFeature is the row of the match defaults a source's block runs.
func (s *ContextSource) MatchFeature() MatchFeature {
	if s.Type == ContextSourceMemory {
		return MatchMemory
	}
	return MatchRAG
}

// AutoStore is the memory source's own write-after-response setting, nil
// when it follows the store's.
func (s *ContextSource) AutoStore() *bool {
	if s == nil || s.Store == nil {
		return nil
	}
	return s.Store.Auto
}

// EffectiveOnFailure is what a failed retrieval of a rag source does.
func (s *ContextSource) EffectiveOnFailure() string {
	if s == nil || s.OnFailure == "" {
		return RAGOnFailureSkip
	}
	return s.OnFailure
}

// CompressesRequest reports whether the block compresses the request the
// selected model reads: every block but one that holds only signals.
func (c *ContextCompressConfig) CompressesRequest() bool {
	if c == nil {
		return false
	}
	return c.Signals == nil || c.Mode != "" || c.Targets != nil || c.Match != nil ||
		c.Recovery != nil || c.RequestControls != nil || c.FailureMode != ""
}

// EffectiveMode returns the defaulted compression mode.
func (c *ContextCompressConfig) EffectiveMode() string {
	if c == nil || strings.TrimSpace(c.Mode) == "" {
		return ContextCompressionModeAuto
	}
	return strings.TrimSpace(c.Mode)
}

// EffectiveToolOutputTarget is the tool-output target with its defaults.
func (c *ContextCompressConfig) EffectiveToolOutputTarget() ContextCompressionTargetConfig {
	result := ContextCompressionTargetConfig{
		Mode:         ContextCompressionTargetExtractive,
		MinTokens:    2000,
		TargetTokens: 1000,
	}
	if c == nil || c.Targets == nil {
		return result
	}
	target := c.Targets.ToolOutputs
	if strings.TrimSpace(target.Mode) != "" {
		result.Mode = strings.TrimSpace(target.Mode)
	}
	if target.MinTokens != 0 {
		result.MinTokens = target.MinTokens
	}
	if target.TargetTokens != 0 {
		result.TargetTokens = target.TargetTokens
	}
	return result
}

// EffectiveTargetMode returns a target's defaulted mode.
func (c *ContextCompressConfig) EffectiveTargetMode(target ContextCompressionTargetConfig) string {
	if strings.TrimSpace(target.Mode) == "" {
		return ContextCompressionTargetPreserve
	}
	return strings.TrimSpace(target.Mode)
}

// EffectiveFailureMode returns the defaulted failure mode.
func (c *ContextCompressConfig) EffectiveFailureMode() string {
	if c == nil || strings.TrimSpace(c.FailureMode) == "" {
		return ContextCompressionFailureOpen
	}
	return strings.TrimSpace(c.FailureMode)
}
