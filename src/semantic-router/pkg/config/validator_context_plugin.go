package config

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// ragSourceFields and memorySourceFields are the fields that belong to one
// source type, which a source of the other type refuses.
var (
	ragSourceFields = map[string]func(*ContextSource) bool{
		"backend":            func(s *ContextSource) bool { return s.Backend != "" },
		"backend_config":     func(s *ContextSource) bool { return s.BackendConfig != nil },
		"max_context_length": func(s *ContextSource) bool { return s.MaxContextLength != nil },
		"on_failure":         func(s *ContextSource) bool { return s.OnFailure != "" },
		"cache_results":      func(s *ContextSource) bool { return s.CacheResults },
		"cache_ttl_seconds":  func(s *ContextSource) bool { return s.CacheTTLSeconds != nil },
	}
	memorySourceFields = map[string]func(*ContextSource) bool{
		"store":         func(s *ContextSource) bool { return s.Store != nil },
		"hybrid_search": func(s *ContextSource) bool { return s.HybridSearch },
		"hybrid_mode":   func(s *ContextSource) bool { return s.HybridMode != "" },
		"reflection":    func(s *ContextSource) bool { return s.Reflection != nil },
	}
)

// validateContextPlugin checks a decision's enabled context plugin, whose
// configuration is at path: its sources and their match blocks, compress
// and its match block, the budget compress keeps and the injection mode.
func validateContextPlugin(cfg *RouterConfig, path string, plugin *ContextPluginConfig) error {
	if !plugin.Enabled {
		return nil
	}
	if len(plugin.Sources) == 0 && plugin.Compress == nil {
		return fmt.Errorf("%s: an enabled context plugin retrieves sources, compresses the request or both; write sources or compress, or set enabled: false", path)
	}
	seen := make(map[string]bool, len(plugin.Sources))
	for index := range plugin.Sources {
		source := &plugin.Sources[index]
		at := fmt.Sprintf("%s.sources[%d]", path, index)
		if err := validateContextSource(cfg, at, source); err != nil {
			return err
		}
		if seen[source.Type] {
			return fmt.Errorf("%s: a context plugin has at most one %s source; a hybrid rag backend combines two backends", at, source.Type)
		}
		seen[source.Type] = true
	}
	if plugin.Compress != nil {
		if err := validateContextCompress(cfg, path+".compress", plugin.Compress); err != nil {
			return err
		}
	}
	if plugin.Budget != nil {
		if err := validateContextBudget(path+".budget", plugin); err != nil {
			return err
		}
	}
	if plugin.Injection != nil {
		switch {
		case plugin.Injection.Mode != "" && plugin.Injection.Mode != ContextInjectionToolRole && plugin.Injection.Mode != ContextInjectionSystemPrompt:
			return fmt.Errorf("%s.injection.mode must be %s or %s", path, ContextInjectionToolRole, ContextInjectionSystemPrompt)
		case len(plugin.Sources) == 0:
			return fmt.Errorf("%s.injection places what the sources retrieve; add a source or remove injection", path)
		}
	}
	return nil
}

func validateContextSource(cfg *RouterConfig, path string, source *ContextSource) error {
	var other map[string]func(*ContextSource) bool
	switch source.Type {
	case ContextSourceRAG:
		other = memorySourceFields
	case ContextSourceMemory:
		other = ragSourceFields
	default:
		return fmt.Errorf("%s.type must be %s or %s", path, ContextSourceRAG, ContextSourceMemory)
	}
	for _, field := range slices.Sorted(maps.Keys(other)) {
		if other[field](source) {
			return fmt.Errorf("%s.%s: a %s source takes no %s", path, field, source.Type, field)
		}
	}
	if source.Type == ContextSourceRAG {
		if err := validateRAGSource(source); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		if err := validateRAGQueryEmbeddings(cfg, path, source); err != nil {
			return err
		}
	} else if err := validateMemorySource(path, source); err != nil {
		return err
	}
	if err := validateMatch(cfg, path+".match", source.Match, source.MatchFeature()); err != nil {
		return err
	}
	if source.Type == ContextSourceRAG {
		if searchesVectorStore(source) && (cfg.VectorStore == nil || !cfg.VectorStore.Enabled) {
			return fmt.Errorf("%s: a vectorstore backend searches the Router's vector store; set global.stores.vector_store.enabled: true", path)
		}
		return validateRAGLexicalMatch(cfg, path, source)
	}
	return nil
}

// searchesVectorStore reports whether a rag source searches the Router's
// vector store: a vectorstore source, or a hybrid one with a vectorstore
// backend.
func searchesVectorStore(source *ContextSource) bool {
	if source.Backend != "hybrid" {
		return source.Backend == "vectorstore"
	}
	hybrid, err := source.HybridBackendConfig()
	return err == nil && (hybrid.Primary == "vectorstore" || hybrid.Fallback == "vectorstore")
}

// validateRAGLexicalMatch checks a rag source that recalls by BM25 or judges
// without recall. Both read the chunks of a local knowledge base, a
// vectorstore source over the in-memory vector store, which keeps their
// lexical index and lists them; every other backend recalls through its own
// search.
func validateRAGLexicalMatch(cfg *RouterConfig, path string, source *ContextSource) error {
	block := source.Match
	if block == nil || (block.Recall != nil && block.Recall.EffectiveMethod() != MatchRecallBM25) {
		return nil
	}
	at, what := path+".match.recall.method", "bm25 recall"
	if block.Recall == nil {
		at, what = path+".match", "a judge without recall"
	}
	switch {
	case source.Backend != "vectorstore":
		return fmt.Errorf("%s: %s reads a local knowledge base, a vectorstore source; a %s source recalls with embedding", at, what, source.Backend)
	case cfg.VectorStore == nil || cfg.VectorStore.BackendType != "memory":
		return fmt.Errorf("%s: %s reads the chunks the in-memory vector store keeps; set global.stores.vector_store.backend_type: memory, or recall with embedding", at, what)
	}
	return nil
}

func validateMemorySource(path string, source *ContextSource) error {
	if err := validateMemoryHybridMode(source.HybridMode, path+".hybrid_mode"); err != nil {
		return err
	}
	if source.HybridMode != "" && !source.HybridSearch {
		return fmt.Errorf("%s.hybrid_mode fuses the scores of hybrid_search; set hybrid_search: true or remove hybrid_mode", path)
	}
	if source.Reflection != nil {
		return validateMemoryReflectionContracts(*source.Reflection, path+".reflection")
	}
	return nil
}

// validateContextCompress checks the compress block: its mode and failure
// mode, targets, recovery store, request controls and match block, and its
// signals.
func validateContextCompress(cfg *RouterConfig, path string, compress *ContextCompressConfig) error {
	if compress.Signals != nil {
		if err := validateSignalsCompression(path+".signals", compress.Signals); err != nil {
			return err
		}
	}
	if !compress.CompressesRequest() {
		return nil
	}
	if err := validateContextCompressShape(path, compress); err != nil {
		return err
	}
	if err := validateContextCompressionRecovery(cfg, path+".recovery", compress.Recovery); err != nil {
		return err
	}
	return validateMatch(cfg, path+".match", compress.Match, MatchCompress)
}

// validateContextCompressShape checks what a compress block says on its
// own: its modes, targets and request controls.
func validateContextCompressShape(path string, compress *ContextCompressConfig) error {
	switch compress.EffectiveMode() {
	case ContextCompressionModeAuto, ContextCompressionModeAlways:
	default:
		return fmt.Errorf("%s.mode must be auto or always", path)
	}
	switch compress.EffectiveFailureMode() {
	case ContextCompressionFailureOpen, ContextCompressionFailureClosed:
	default:
		return fmt.Errorf("%s.failure_mode must be fail_open or fail_closed", path)
	}
	if err := validateContextCompressionTargets(path+".targets", compress); err != nil {
		return err
	}
	return validateContextCompressionControls(path+".request_controls", compress.RequestControls)
}

func validateContextBudget(path string, plugin *ContextPluginConfig) error {
	if plugin.Compress == nil {
		return fmt.Errorf("%s bounds the request compress produces; add compress or remove budget", path)
	}
	if !plugin.Compress.CompressesRequest() {
		return fmt.Errorf("%s bounds the request compress produces, and a compress block with only signals leaves the request whole; set compress.mode: auto or remove budget", path)
	}
	if limit := plugin.Budget.MaxTokens; limit != nil && !limit.Auto && limit.Value == 0 {
		return fmt.Errorf("%s.max_tokens must be at least 1, or auto", path)
	}
	return nil
}

func validateContextCompressionTargets(path string, compress *ContextCompressConfig) error {
	if compress.Targets != nil {
		mode := strings.TrimSpace(compress.Targets.CurrentUser.Mode)
		if mode != "" && mode != ContextCompressionTargetPreserve && mode != ContextCompressionTargetTruncate {
			return fmt.Errorf("%s.current_user.mode must be preserve or truncate", path)
		}
	}
	targets := map[string]ContextCompressionTargetConfig{"tool_outputs": compress.EffectiveToolOutputTarget()}
	if compress.Targets != nil {
		targets["history"], targets["rag"], targets["memory"] = compress.Targets.History, compress.Targets.RAG, compress.Targets.Memory
	}
	for _, name := range slices.Sorted(maps.Keys(targets)) {
		target := targets[name]
		switch compress.EffectiveTargetMode(target) {
		case ContextCompressionTargetPreserve, ContextCompressionTargetExtractive, ContextCompressionTargetRecoverable:
		default:
			return fmt.Errorf("%s.%s.mode must be preserve, extractive or recoverable", path, name)
		}
		if target.MinTokens < 0 || target.TargetTokens < 0 {
			return fmt.Errorf("%s.%s: token thresholds cannot be negative", path, name)
		}
		if target.MinTokens > 0 && target.TargetTokens > 0 && target.TargetTokens >= target.MinTokens {
			return fmt.Errorf("%s.%s.target_tokens must be less than min_tokens", path, name)
		}
	}
	return nil
}

// validateContextCompressionRecovery checks the store recoverable targets
// keep their originals in, which the response cache's connection serves.
func validateContextCompressionRecovery(cfg *RouterConfig, path string, recovery *ContextCompressionRecoveryConfig) error {
	if recovery == nil || !recovery.Enabled {
		return nil
	}
	if recovery.TTLSeconds < 0 || recovery.MaxBytesPerRequest < 0 || recovery.MaxTotalBytes < 0 || recovery.MaxRetrievals < 0 {
		return fmt.Errorf("%s: limits cannot be negative", path)
	}
	store := strings.TrimSpace(recovery.Store)
	switch store {
	case "redis", "valkey", "response_cache":
	default:
		return fmt.Errorf("%s.store must be redis, valkey or response_cache", path)
	}
	if store == "response_cache" {
		store = strings.TrimSpace(cfg.SemanticCache.BackendType)
	}
	switch {
	case store == "redis" && cfg.SemanticCache.Redis == nil:
		return fmt.Errorf("%s needs the response cache's redis connection (global.stores.response_cache.redis)", path)
	case store == "valkey" && cfg.SemanticCache.Valkey == nil:
		return fmt.Errorf("%s needs the response cache's valkey connection (global.stores.response_cache.valkey)", path)
	case store != "redis" && store != "valkey":
		return fmt.Errorf("%s needs a redis or valkey store; the response cache uses %q", path, store)
	}
	return nil
}

func validateContextCompressionControls(path string, controls *ContextCompressionRequestControlsConfig) error {
	if controls == nil {
		return nil
	}
	if controls.MaxTargetTokens < 0 {
		return fmt.Errorf("%s.max_target_tokens cannot be negative", path)
	}
	for _, directive := range controls.Allowed {
		switch strings.TrimSpace(directive) {
		case "bypass", "target":
		default:
			return fmt.Errorf("%s: unsupported request control %q", path, directive)
		}
	}
	return nil
}

// ValidateContextCompressPreview checks a context plugin's compress block and
// budget for a compression preview, which runs without a configuration: the
// recovery store and the match block's deployments are not checked.
func ValidateContextCompressPreview(plugin *ContextPluginConfig) error {
	if plugin == nil {
		return fmt.Errorf("context configuration is required")
	}
	if !plugin.Compress.CompressesRequest() {
		return nil
	}
	if err := validateContextCompressShape("compress", plugin.Compress); err != nil {
		return err
	}
	if plugin.Budget != nil {
		return validateContextBudget("budget", plugin)
	}
	return nil
}
