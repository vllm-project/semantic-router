package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"unicode"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Empty deployments still have an immutable configuration identity: SHA256 of
// the empty byte sequence. This is not a claim that a Router config exists.
const unavailableConfigDigest = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

// fixturePolicySnapshotDigest is the content identity of the immutable
// builtin replay policy executed by the Python fixture adapter.
const fixturePolicySnapshotDigest = "sha256:34063b31576749e60610d650ba7a045988db38b7de9d27b69b71e3f1e426a9f3"

// ModelArmSnapshot is the public, connectivity-free view of the models that a
// live evaluation can address through the server-owned Envoy target. The
// digest and arms are derived from the same immutable byte slice.
type ModelArmSnapshot struct {
	ModelArms             []ModelArm
	ConfigDigest          string
	PolicySnapshotDigest  string
	BackendTopologyDigest string
}

// LoadModelArmSnapshot reads and freezes the current Router config. An empty
// path represents a deployment without a configured Router snapshot.
func LoadModelArmSnapshot(configPath, runtimeRevision string) (ModelArmSnapshot, error) {
	configPath = strings.TrimSpace(configPath)
	if configPath == "" {
		return ModelArmSnapshot{
			ConfigDigest:         unavailableConfigDigest,
			PolicySnapshotDigest: policySnapshotDigest(routerconfig.CanonicalConfig{}),
		}, nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return ModelArmSnapshot{}, fmt.Errorf("read evaluated Router config: %w", err)
	}
	return ModelArmSnapshotFromYAML(data, runtimeRevision)
}

// ModelArmSnapshotFromYAML parses the canonical Router contract and exports
// only logical model identity, one-way provider identity, pricing, and public
// capability metadata. Backend connectivity and credentials never cross this
// boundary.
func ModelArmSnapshotFromYAML(data []byte, runtimeRevision string) (ModelArmSnapshot, error) {
	digest := digestBytes(data)
	cfg, err := routerconfig.ParseYAMLBytes(data)
	if err != nil {
		return ModelArmSnapshot{}, fmt.Errorf("parse evaluated Router config: %w", err)
	}
	canonical := routerconfig.CanonicalConfigFromRouterConfig(cfg)
	return ModelArmSnapshot{
		ModelArms:             modelArmsFromCanonical(canonical, runtimeRevision),
		ConfigDigest:          digest,
		PolicySnapshotDigest:  policySnapshotDigest(canonical),
		BackendTopologyDigest: backendTopologyDigest(canonical),
	}, nil
}

// policySnapshotFingerprint deliberately excludes model cards, providers,
// listeners, and global runtime state. Those inputs belong to the pool,
// backend-topology, or full-config lineage factors instead of the routing
// policy treatment.
type policySnapshotFingerprint struct {
	Entrypoints []routerconfig.CanonicalEntrypoint `json:"entrypoints,omitempty"`
	Routing     policyRoutingFingerprint           `json:"routing"`
	Recipes     []policyRecipeFingerprint          `json:"recipes,omitempty"`
}

type policyRoutingFingerprint struct {
	Signals     routerconfig.CanonicalSignals     `json:"signals"`
	Projections routerconfig.CanonicalProjections `json:"projections"`
	Decisions   []routerconfig.Decision           `json:"decisions,omitempty"`
	Strategy    routerconfig.RoutingStrategy      `json:"strategy,omitempty"`
}

type policyRecipeFingerprint struct {
	Name        string                   `json:"name"`
	Description string                   `json:"description,omitempty"`
	Routing     policyRoutingFingerprint `json:"routing"`
}

func policySnapshotDigest(canonical routerconfig.CanonicalConfig) string {
	recipes := make([]policyRecipeFingerprint, 0, len(canonical.Recipes))
	for _, recipe := range canonical.Recipes {
		recipes = append(recipes, policyRecipeFingerprint{
			Name:        recipe.Name,
			Description: recipe.Description,
			Routing:     policyRoutingFromCanonical(recipe.Routing),
		})
	}
	return digestJSON(policySnapshotFingerprint{
		Entrypoints: canonical.Entrypoints,
		Routing:     policyRoutingFromCanonical(canonical.Routing),
		Recipes:     recipes,
	})
}

func policyRoutingFromCanonical(routing routerconfig.CanonicalRouting) policyRoutingFingerprint {
	return policyRoutingFingerprint{
		Signals:     routing.Signals,
		Projections: routing.Projections,
		Decisions:   routing.Decisions,
		Strategy:    routing.Strategy,
	}
}

func modelArmsFromCanonical(
	canonical routerconfig.CanonicalConfig,
	runtimeRevision string,
) []ModelArm {
	cards := make(map[string]routerconfig.RoutingModel, len(canonical.Routing.ModelCards))
	for _, card := range canonical.Routing.ModelCards {
		cards[card.Name] = card
	}

	arms := make([]ModelArm, 0, len(canonical.Providers.Models))
	seenModels := make(map[string]struct{}, len(canonical.Providers.Models))
	for _, provider := range canonical.Providers.Models {
		model := strings.TrimSpace(provider.Name)
		if !validProviderArm(provider, model) {
			continue
		}
		if _, duplicate := seenModels[model]; duplicate {
			continue
		}
		seenModels[model] = struct{}{}

		providerIdentity := strings.TrimSpace(provider.ProviderModelID)
		if providerIdentity == "" {
			providerIdentity = model
		}
		card := cards[provider.Name]
		arm := ModelArm{
			ID:                            portableModelArmID(model),
			Model:                         model,
			ProviderModelIDDigest:         digestString(providerIdentity),
			InputCostPerMillionTokensUSD:  provider.Pricing.PromptPer1M,
			OutputCostPerMillionTokensUSD: provider.Pricing.CompletionPer1M,
			Capabilities:                  normalizedCapabilities(card.Capabilities),
			Modalities:                    normalizedModalities(card),
			ContextWindowTokens:           positiveInt(card.ContextWindowSize),
			ParameterSize:                 boundedOptionalString(card.ParamSize, 64),
			RuntimeRevision:               runtimeRevisionPointer(runtimeRevision),
		}
		arm.ConfigDigest = stringPointer(modelArmConfigDigest(provider, arm))
		arms = append(arms, arm)
	}

	sort.Slice(arms, func(i, j int) bool {
		return arms[i].Model < arms[j].Model
	})
	if len(arms) == 0 {
		return nil
	}
	return arms
}

type armConfigFingerprint struct {
	Model                         string   `json:"model"`
	ProviderModelIDDigest         string   `json:"provider_model_id_digest"`
	ReasoningFamily               string   `json:"reasoning_family,omitempty"`
	APIFormat                     string   `json:"api_format,omitempty"`
	InputCostPerMillionTokensUSD  float64  `json:"input_cost_per_million_tokens_usd"`
	OutputCostPerMillionTokensUSD float64  `json:"output_cost_per_million_tokens_usd"`
	Capabilities                  []string `json:"capabilities,omitempty"`
	Modalities                    []string `json:"modalities,omitempty"`
	ContextWindowTokens           *int     `json:"context_window_tokens,omitempty"`
	ParameterSize                 *string  `json:"parameter_size,omitempty"`
	ExternalModelIDDigest         string   `json:"external_model_ids_digest,omitempty"`
}

func modelArmConfigDigest(provider routerconfig.CanonicalProviderModel, arm ModelArm) string {
	externalDigest := ""
	if len(provider.ExternalModelIDs) > 0 {
		externalDigest = digestJSON(provider.ExternalModelIDs)
	}
	return digestJSON(armConfigFingerprint{
		Model: arm.Model, ProviderModelIDDigest: arm.ProviderModelIDDigest,
		ReasoningFamily: provider.ReasoningFamily, APIFormat: provider.APIFormat,
		InputCostPerMillionTokensUSD:  arm.InputCostPerMillionTokensUSD,
		OutputCostPerMillionTokensUSD: arm.OutputCostPerMillionTokensUSD,
		Capabilities:                  arm.Capabilities, Modalities: arm.Modalities,
		ContextWindowTokens: arm.ContextWindowTokens, ParameterSize: arm.ParameterSize,
		ExternalModelIDDigest: externalDigest,
	})
}

type topologyBackendFingerprint struct {
	Name             string   `json:"name,omitempty"`
	EndpointDigest   string   `json:"endpoint_digest,omitempty"`
	BaseURLDigest    string   `json:"base_url_digest,omitempty"`
	Protocol         string   `json:"protocol,omitempty"`
	Weight           int      `json:"weight,omitempty"`
	Type             string   `json:"type,omitempty"`
	Provider         string   `json:"provider,omitempty"`
	APIVersion       string   `json:"api_version,omitempty"`
	ChatPath         string   `json:"chat_path,omitempty"`
	ExtraHeaderNames []string `json:"extra_header_names,omitempty"`
}

type topologyModelFingerprint struct {
	Model                 string                       `json:"model"`
	ProviderModelIDDigest string                       `json:"provider_model_id_digest"`
	Backends              []topologyBackendFingerprint `json:"backends"`
}

func backendTopologyDigest(canonical routerconfig.CanonicalConfig) string {
	models := make([]topologyModelFingerprint, 0, len(canonical.Providers.Models))
	for _, provider := range canonical.Providers.Models {
		if len(provider.BackendRefs) == 0 {
			continue
		}
		identity := strings.TrimSpace(provider.ProviderModelID)
		if identity == "" {
			identity = strings.TrimSpace(provider.Name)
		}
		model := topologyModelFingerprint{
			Model: strings.TrimSpace(provider.Name), ProviderModelIDDigest: digestString(identity),
			Backends: make([]topologyBackendFingerprint, 0, len(provider.BackendRefs)),
		}
		for _, backend := range provider.BackendRefs {
			headers := make([]string, 0, len(backend.ExtraHeaders))
			for name := range backend.ExtraHeaders {
				headers = append(headers, strings.ToLower(strings.TrimSpace(name)))
			}
			sort.Strings(headers)
			model.Backends = append(model.Backends, topologyBackendFingerprint{
				Name: strings.TrimSpace(backend.Name), EndpointDigest: optionalValueDigest(backend.Endpoint),
				BaseURLDigest: optionalValueDigest(backend.BaseURL), Protocol: strings.TrimSpace(backend.Protocol),
				Weight: backend.Weight, Type: strings.TrimSpace(backend.Type), Provider: strings.TrimSpace(backend.Provider),
				APIVersion: strings.TrimSpace(backend.APIVersion), ChatPath: strings.TrimSpace(backend.ChatPath),
				ExtraHeaderNames: headers,
			})
		}
		sort.Slice(model.Backends, func(i, j int) bool {
			return digestJSON(model.Backends[i]) < digestJSON(model.Backends[j])
		})
		models = append(models, model)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Model < models[j].Model })
	return digestJSON(models)
}

func optionalValueDigest(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}
	return digestString(value)
}

func digestJSON(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(fmt.Sprintf("canonical evaluation digest: %v", err))
	}
	return digestBytes(encoded)
}

func validProviderArm(provider routerconfig.CanonicalProviderModel, model string) bool {
	if model == "" || len(model) > 512 || len(provider.BackendRefs) == 0 {
		return false
	}
	pricing := provider.Pricing
	if pricing.PromptPer1M < 0 || pricing.CompletionPer1M < 0 ||
		math.IsNaN(pricing.PromptPer1M) || math.IsNaN(pricing.CompletionPer1M) ||
		math.IsInf(pricing.PromptPer1M, 0) || math.IsInf(pricing.CompletionPer1M, 0) {
		return false
	}
	currency := strings.TrimSpace(pricing.Currency)
	if strings.EqualFold(currency, "USD") {
		return true
	}
	return currency == "" && pricing.PromptPer1M == 0 && pricing.CompletionPer1M == 0
}

func portableModelArmID(model string) string {
	var base strings.Builder
	lastSeparator := false
	for _, value := range strings.ToLower(model) {
		if (value >= 'a' && value <= 'z') || (value >= '0' && value <= '9') || value == '.' || value == '_' {
			base.WriteRune(value)
			lastSeparator = false
			continue
		}
		if !lastSeparator && base.Len() > 0 {
			base.WriteByte('-')
			lastSeparator = true
		}
	}
	readable := strings.Trim(base.String(), ".-_")
	if readable == "" || !startsWithASCIILetterOrDigit(readable) {
		readable = "model"
	}
	// A full digest makes IDs collision-resistant even when different model
	// names normalize to the same portable prefix. 63+1+64 is the Python
	// contract's 128-character maximum.
	if len(readable) > 63 {
		readable = strings.TrimRight(readable[:63], ".-_")
	}
	return readable + "-" + strings.TrimPrefix(digestString(model), "sha256:")
}

func normalizedCapabilities(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, duplicate := seen[value]; duplicate {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Strings(result)
	if len(result) == 0 {
		return nil
	}
	return result
}

func normalizedModalities(card routerconfig.RoutingModel) []string {
	seen := make(map[string]bool, 5)
	add := func(modality string) { seen[modality] = true }
	switch strings.ToLower(strings.TrimSpace(card.Modality)) {
	case "text", "ar":
		add("text")
	case "diffusion", "image":
		add("image")
	case "omni":
		add("text")
		add("image")
	case "document", "audio", "video":
		add(strings.ToLower(strings.TrimSpace(card.Modality)))
	}
	for _, capability := range card.Capabilities {
		normalized := normalizeCapabilityForModality(capability)
		switch normalized {
		case "text", "chat", "reasoning", "code", "text_generation":
			add("text")
		case "image", "vision", "image_understanding", "image_generation", "multimodal", "omni":
			add("image")
		case "document", "document_understanding", "ocr":
			add("document")
		case "audio", "speech", "speech_to_text", "text_to_speech":
			add("audio")
		case "video", "video_understanding":
			add("video")
		}
	}

	order := []string{"text", "image", "document", "audio", "video"}
	result := make([]string, 0, len(seen))
	for _, modality := range order {
		if seen[modality] {
			result = append(result, modality)
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func normalizeCapabilityForModality(value string) string {
	var normalized strings.Builder
	lastSeparator := false
	for _, char := range strings.ToLower(strings.TrimSpace(value)) {
		if unicode.IsLetter(char) || unicode.IsDigit(char) {
			normalized.WriteRune(char)
			lastSeparator = false
			continue
		}
		if !lastSeparator && normalized.Len() > 0 {
			normalized.WriteByte('_')
			lastSeparator = true
		}
	}
	return strings.Trim(normalized.String(), "_")
}

func digestBytes(data []byte) string {
	digest := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%x", digest[:])
}

func digestString(value string) string {
	return digestBytes([]byte(value))
}

func startsWithASCIILetterOrDigit(value string) bool {
	first := value[0]
	return (first >= 'a' && first <= 'z') || (first >= '0' && first <= '9')
}

func positiveInt(value int) *int {
	if value <= 0 {
		return nil
	}
	return &value
}

func boundedOptionalString(value string, limit int) *string {
	value = strings.TrimSpace(value)
	if value == "" || len(value) > limit {
		return nil
	}
	return &value
}

func runtimeRevisionPointer(value string) *string {
	if strings.EqualFold(strings.TrimSpace(value), "unavailable") {
		return nil
	}
	return boundedOptionalString(value, 160)
}

func stringPointer(value string) *string {
	return &value
}
