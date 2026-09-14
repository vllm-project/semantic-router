package tools

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// errTrailingJSON reports input whose top-level JSON value is followed by
// more bytes. json.Unmarshal rejects that on its own; a json.Decoder stops
// at the first value instead, so canonicalizeJSON re-asserts the check
// explicitly. Without it, a schema followed by arbitrary trailing bytes
// would canonicalize to the schema alone — a fingerprint collision between
// inputs that are not the same document.
var errTrailingJSON = errors.New("unexpected trailing data after top-level JSON value")

// canonicalizeJSON re-marshals raw JSON bytes so object keys are sorted and
// whitespace is normalized: encoding/json.Marshal already sorts
// map[string]interface{} keys, and the decode-then-Marshal round trip is
// what extends that guarantee to arbitrary input bytes (a tool's raw
// InputSchema) rather than only to values this package constructs itself.
// Two semantically identical JSON documents that differ only in key order
// or whitespace canonicalize to the same bytes.
//
// Numbers decode through Decoder.UseNumber, which keeps each numeric
// literal as a json.Number that Marshal writes back verbatim. Decoding into
// a plain interface{} instead would route every number through float64,
// whose 53-bit mantissa silently rounds larger integers: distinct valid
// schema integers past that precision (an authorization-shaped const or
// maximum, say) would then canonicalize to identical bytes, and a catalog
// fingerprint that cannot see the change lets sticky state survive a real
// tool-schema change.
//
// Preserving literals exactly makes canonicalization literal-sensitive in
// return: 100, 1e2, and 100.0 canonicalize distinctly rather than
// collapsing to 100. That is the safe direction — it can only invalidate
// sticky state that a byte-identical schema would have kept, never preserve
// state across a schema that actually changed — and normalizing the
// literals is precisely what would reintroduce the precision loss.
func canonicalizeJSON(raw json.RawMessage) (json.RawMessage, error) {
	if len(raw) == 0 {
		return json.RawMessage("null"), nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var generic interface{}
	if err := decoder.Decode(&generic); err != nil {
		return nil, err
	}
	// A second Decode must hit clean end-of-input: anything else is either
	// a further value or malformed bytes, neither of which the single-value
	// contract of an InputSchema allows.
	if err := decoder.Decode(new(interface{})); !errors.Is(err, io.EOF) {
		return nil, errTrailingJSON
	}
	canonical, err := json.Marshal(generic)
	if err != nil {
		return nil, err
	}
	return canonical, nil
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// marshalFingerprint hashes v's canonical JSON encoding. v must be built
// from fixed struct fields (deterministic marshal order) plus, where
// order is not itself meaningful, pre-sorted slices — never a bare
// map[string]T with semantically-insignificant key order left to chance
// wording, and never raw caller-supplied JSON bytes without first passing
// them through canonicalizeJSON.
func marshalFingerprint(v interface{}) string {
	encoded, err := json.Marshal(v)
	if err != nil {
		// The inputs built in this file are fixed structs of primitives,
		// pointers, and pre-canonicalized json.RawMessage — Marshal
		// cannot fail on them in practice. Fall back to a fixed sentinel
		// rather than panic so a fingerprinting failure never crashes a
		// request; a sentinel fingerprint just forces the next
		// comparison to treat this as changed, which is fail-safe.
		return sha256Hex([]byte("sessiontools:fingerprint_marshal_error"))
	}
	return sha256Hex(encoded)
}

type toolDefinitionFingerprintInput struct {
	Name         string          `json:"name"`
	Description  string          `json:"description"`
	Strict       *bool           `json:"strict"`
	CachePresent bool            `json:"cache_present"`
	CacheType    string          `json:"cache_type,omitempty"`
	CacheTTL     string          `json:"cache_ttl,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema"`
}

// ToolDefinitionFingerprint returns a canonical, deterministic fingerprint
// of tool's provider-visible definition: exact name and description,
// strictness, cache directive, and canonical input schema. Two calls for
// semantically identical tools — even if InputSchema's raw bytes differ in
// key order or whitespace — produce the same fingerprint; any actual
// definition change produces a different one. Strict and the cache
// directive's presence-vs-absence are preserved distinctly from their
// zero values (nil Strict is not the same configuration as Strict: false).
func ToolDefinitionFingerprint(tool llmprotocol.Tool) string {
	schema, err := canonicalizeJSON(tool.InputSchema)
	if err != nil {
		// A tool with an unparsable schema cannot be meaningfully compared
		// for equality against a well-formed one; fold the failure into
		// the hashed input itself (rather than silently substituting
		// "null") so it still fingerprints deterministically, and
		// distinctly from every valid schema.
		schema, _ = json.Marshal("invalid_input_schema:" + sha256Hex(tool.InputSchema))
	}
	input := toolDefinitionFingerprintInput{
		Name:        tool.Name,
		Description: tool.Description,
		Strict:      tool.Strict,
		InputSchema: schema,
	}
	if tool.Cache != nil {
		input.CachePresent = true
		input.CacheType = tool.Cache.Type
		input.CacheTTL = tool.Cache.TTL
	}
	return marshalFingerprint(input)
}

type toolCatalogEntry struct {
	Name        string `json:"name"`
	Fingerprint string `json:"fingerprint"`
}

// ToolCatalogFingerprint returns a canonical fingerprint of the whole
// catalog: (name, definition fingerprint) pairs sorted by name before
// hashing, so any permutation of the same tools produces the same
// fingerprint — catalog order is transport detail, not policy.
func ToolCatalogFingerprint(catalog []llmprotocol.Tool) string {
	entries := make([]toolCatalogEntry, len(catalog))
	for i, tool := range catalog {
		entries[i] = toolCatalogEntry{
			Name:        tool.Name,
			Fingerprint: ToolDefinitionFingerprint(tool),
		}
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Name == entries[j].Name {
			return entries[i].Fingerprint < entries[j].Fingerprint
		}
		return entries[i].Name < entries[j].Name
	})
	return marshalFingerprint(entries)
}

type toolPolicyFingerprintInput struct {
	Enabled              bool                                  `json:"enabled"`
	Mode                 string                                `json:"mode"`
	TopK                 int                                   `json:"top_k,omitempty"`
	SimilarityThreshold  *float32                              `json:"similarity_threshold,omitempty"`
	Strategy             string                                `json:"strategy"`
	RelevanceThreshold   *float32                              `json:"relevance_threshold,omitempty"`
	PreserveCount        int                                   `json:"preserve_count,omitempty"`
	AdvancedFiltering    *advancedToolFilteringFingerprintView `json:"advanced_filtering,omitempty"`
	FallbackToEmpty      *bool                                 `json:"fallback_to_empty,omitempty"`
	StickyEnabled        bool                                  `json:"sticky_enabled"`
	StickyMaxTools       int                                   `json:"sticky_max_tools"`
	StickyMaxNewPerTurn  int                                   `json:"sticky_max_new_tools_per_turn"`
	StickyPinCalledTools bool                                  `json:"sticky_pin_called_tools"`
}

// advancedToolFilteringFingerprintView mirrors config.AdvancedToolFilteringConfig
// for fingerprinting, except AllowTools/BlockTools are sorted copies: those
// are semantically unordered sets, and hashing the caller's raw slice order
// would make the fingerprint change on a config reformat that changes
// nothing about effective policy.
type advancedToolFilteringFingerprintView struct {
	Enabled                     bool                                     `json:"enabled"`
	RetrievalStrategy           string                                   `json:"retrieval_strategy,omitempty"`
	CandidatePoolSize           *int                                     `json:"candidate_pool_size,omitempty"`
	MinLexicalOverlap           *int                                     `json:"min_lexical_overlap,omitempty"`
	MinCombinedScore            *float32                                 `json:"min_combined_score,omitempty"`
	Weights                     config.ToolFilteringWeights              `json:"weights,omitempty"`
	UseCategoryFilter           *bool                                    `json:"use_category_filter,omitempty"`
	CategoryConfidenceThreshold *float32                                 `json:"category_confidence_threshold,omitempty"`
	AllowTools                  []string                                 `json:"allow_tools,omitempty"`
	BlockTools                  []string                                 `json:"block_tools,omitempty"`
	HybridHistory               *config.HybridHistoryToolRetrievalConfig `json:"hybrid_history,omitempty"`
}

// legacyToolsPolicyFingerprintView is the canonical, bounded projection of
// the decision-level legacy tools plugin. The request path merges this plugin
// with tool_selection before it authorizes a candidate; omitting it would let
// sticky state survive an allow/block, mode, or retrieval-policy change.
type legacyToolsPolicyFingerprintView struct {
	Enabled          bool                             `json:"enabled"`
	Mode             string                           `json:"mode"`
	SelectionEnabled bool                             `json:"selection_enabled"`
	AllowTools       []string                         `json:"allow_tools,omitempty"`
	BlockTools       []string                         `json:"block_tools,omitempty"`
	StripToolHistory bool                             `json:"strip_tool_history"`
	Strategy         string                           `json:"strategy"`
	DynamicRetrieval *dynamicRetrievalFingerprintView `json:"dynamic_retrieval,omitempty"`
}

type dynamicRetrievalFingerprintView struct {
	Enabled                 bool                            `json:"enabled"`
	Strategy                string                          `json:"strategy"`
	HistoryWindow           int                             `json:"history_window,omitempty"`
	Weights                 *config.DynamicRetrievalWeights `json:"weights,omitempty"`
	MinHistoryConfidence    float64                         `json:"min_history_confidence,omitempty"`
	FallbackOnLowConfidence bool                            `json:"fallback_on_low_confidence,omitempty"`
}

func newAdvancedToolFilteringFingerprintView(c *config.AdvancedToolFilteringConfig) *advancedToolFilteringFingerprintView {
	if c == nil {
		return nil
	}
	return &advancedToolFilteringFingerprintView{
		Enabled:                     c.Enabled,
		RetrievalStrategy:           c.RetrievalStrategy,
		CandidatePoolSize:           c.CandidatePoolSize,
		MinLexicalOverlap:           c.MinLexicalOverlap,
		MinCombinedScore:            c.MinCombinedScore,
		Weights:                     c.Weights,
		UseCategoryFilter:           c.UseCategoryFilter,
		CategoryConfidenceThreshold: c.CategoryConfidenceThreshold,
		AllowTools:                  sortedFoldedStrings(c.AllowTools),
		BlockTools:                  sortedFoldedStrings(c.BlockTools),
		HybridHistory:               c.HybridHistory,
	}
}

func newDynamicRetrievalFingerprintView(c *config.DynamicRetrievalConfig) *dynamicRetrievalFingerprintView {
	if c == nil || !c.Enabled {
		return nil
	}
	return &dynamicRetrievalFingerprintView{
		Enabled:                 true,
		Strategy:                c.EffectiveStrategy(),
		HistoryWindow:           c.HistoryWindow,
		Weights:                 c.Weights,
		MinHistoryConfidence:    c.MinHistoryConfidence,
		FallbackOnLowConfidence: c.FallbackOnLowConfidence,
	}
}

func newLegacyToolsPolicyFingerprintView(c *config.ToolsPluginConfig) *legacyToolsPolicyFingerprintView {
	if c == nil {
		return nil
	}
	return &legacyToolsPolicyFingerprintView{
		Enabled:          c.Enabled,
		Mode:             c.EffectiveMode(),
		SelectionEnabled: c.SelectionEnabled(),
		AllowTools:       sortedFoldedStrings(c.AllowTools),
		BlockTools:       sortedFoldedStrings(c.BlockTools),
		StripToolHistory: c.StripToolHistory,
		Strategy:         c.EffectiveStrategy(),
		DynamicRetrieval: newDynamicRetrievalFingerprintView(c.DynamicRetrieval),
	}
}

func sortedStrings(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	unique := make(map[string]struct{}, len(in))
	for _, value := range in {
		value = strings.TrimSpace(value)
		if value != "" {
			unique[value] = struct{}{}
		}
	}
	sorted := make([]string, 0, len(unique))
	for value := range unique {
		sorted = append(sorted, value)
	}
	sort.Strings(sorted)
	return sorted
}

func sortedFoldedStrings(in []string) []string {
	folded := make([]string, len(in))
	for i, value := range in {
		folded[i] = strings.ToLower(value)
	}
	return sortedStrings(folded)
}

// ToolPolicyFingerprint returns a canonical fingerprint of the effective
// selection policy: mode, thresholds, strategy, advanced filtering, and
// sticky bounds. Deliberately excludes catalog contents (see
// ToolCatalogFingerprint) and source identity such as decision name — two
// decisions with the same effective policy share continuity (see PL-0042
// section 2.3). Sticky bounds are read through their Effective* helpers
// (nil-safe: StickyToolSelectionConfig's Effective* methods handle a nil
// receiver), not the raw configured pointers, so an explicit value equal
// to the default and an omitted field produce the same fingerprint —
// matching the policy fingerprint's purpose (did the *effective* policy
// change), not a raw-YAML diff.
func ToolPolicyFingerprint(pluginCfg *config.ToolSelectionPluginConfig) string {
	if pluginCfg == nil {
		return marshalFingerprint(toolPolicyFingerprintInput{})
	}
	mode := strings.TrimSpace(pluginCfg.Mode)
	if mode == "" {
		mode = config.ToolSelectionModeAdd
	}
	input := toolPolicyFingerprintInput{
		Enabled:              pluginCfg.Enabled,
		Mode:                 mode,
		TopK:                 pluginCfg.TopK,
		SimilarityThreshold:  pluginCfg.SimilarityThreshold,
		Strategy:             pluginCfg.EffectiveStrategy(),
		RelevanceThreshold:   pluginCfg.RelevanceThreshold,
		PreserveCount:        pluginCfg.PreserveCount,
		AdvancedFiltering:    newAdvancedToolFilteringFingerprintView(pluginCfg.AdvancedFiltering),
		FallbackToEmpty:      pluginCfg.FallbackToEmpty,
		StickyMaxTools:       pluginCfg.Sticky.EffectiveMaxTools(),
		StickyMaxNewPerTurn:  pluginCfg.Sticky.EffectiveMaxNewToolsPerTurn(),
		StickyPinCalledTools: pluginCfg.Sticky.EffectivePinCalledTools(),
	}
	if pluginCfg.Sticky != nil {
		input.StickyEnabled = pluginCfg.Sticky.Enabled
	}
	return marshalFingerprint(input)
}

type effectiveToolPolicyFingerprintInput struct {
	Selection string                            `json:"selection"`
	Legacy    *legacyToolsPolicyFingerprintView `json:"legacy,omitempty"`
}

// EffectiveToolPolicyFingerprint fingerprints the effective tool-selection
// policy after the decision-level legacy tools plugin is taken into account.
// The legacy view is identity-only and canonicalizes unordered allow/block
// lists, so equivalent configuration order does not invalidate a session.
// Callers that also apply global tool defaults should fold those defaults into
// the supplied legacy view before calling this helper.
//
// A nil legacy config preserves ToolPolicyFingerprint's historical output;
// this keeps callers that only have the tool_selection plugin source
// compatible while making the combined form explicit at the runtime seam.
func EffectiveToolPolicyFingerprint(
	pluginCfg *config.ToolSelectionPluginConfig,
	legacyCfg *config.ToolsPluginConfig,
) string {
	if legacyCfg == nil {
		return ToolPolicyFingerprint(pluginCfg)
	}
	return marshalFingerprint(effectiveToolPolicyFingerprintInput{
		Selection: ToolPolicyFingerprint(pluginCfg),
		Legacy:    newLegacyToolsPolicyFingerprintView(legacyCfg),
	})
}

type toolCapabilityFingerprintInput struct {
	ModelCapabilities []string `json:"model_capabilities"`
	WireFormat        string   `json:"wire_format"`
}

// ToolCapabilityFingerprint returns a canonical fingerprint of the current
// selected logical model's declared capabilities and the target wire
// format, sorted so capability declaration order is not semantically
// meaningful.
func ToolCapabilityFingerprint(modelCapabilities []string, wireFormat string) string {
	return marshalFingerprint(toolCapabilityFingerprintInput{
		ModelCapabilities: sortedStrings(modelCapabilities),
		WireFormat:        wireFormat,
	})
}
