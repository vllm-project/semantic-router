package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"unicode"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Empty deployments still have an immutable configuration identity: SHA256 of
// the empty byte sequence. This is not a claim that a Router config exists.
const emptyConfigDigest = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

// fixturePolicySnapshotDigest is the content identity of the immutable
// builtin replay policy executed by the Python fixture adapter.
const fixturePolicySnapshotDigest = "sha256:34063b31576749e60610d650ba7a045988db38b7de9d27b69b71e3f1e426a9f3"

// ModelArmSnapshot freezes every request-reachable Mixture-of-Models from one
// immutable Router configuration byte slice.
type ModelArmSnapshot struct {
	Mixtures     []MixtureTargetSnapshot
	ConfigDigest string
}

// MixtureTargetSnapshot carries the server-owned execution identity for one
// recipe-scoped target. Ready is deliberately not serialized; it controls
// whether the registry exposes executable tracks for an otherwise inspectable
// catalog subject.
type MixtureTargetSnapshot struct {
	Mixture               ManifestMixture
	BackendTopologyDigest string
	ConfigDigest          string
	Ready                 bool
}

// LoadModelArmSnapshot reads and freezes the current Router config. An empty
// path represents a deployment without a configured Router snapshot.
func LoadModelArmSnapshot(configPath, runtimeRevision string) (ModelArmSnapshot, error) {
	configPath = strings.TrimSpace(configPath)
	if configPath == "" {
		return ModelArmSnapshot{ConfigDigest: emptyConfigDigest}, nil
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
	mixtures, err := mixtureSnapshotsFromConfig(cfg, canonical, runtimeRevision)
	if err != nil {
		return ModelArmSnapshot{}, err
	}
	for index := range mixtures {
		mixtures[index].ConfigDigest = digest
	}
	return ModelArmSnapshot{
		Mixtures:     mixtures,
		ConfigDigest: digest,
	}, nil
}

type policyRoutingFingerprint struct {
	Signals   routerconfig.CanonicalSignals `json:"signals"`
	Decisions []routerconfig.Decision       `json:"decisions,omitempty"`
	Strategy  routerconfig.RoutingStrategy  `json:"strategy,omitempty"`
}

type selectorDecisionFingerprint struct {
	Algorithm *routerconfig.AlgorithmConfig `json:"algorithm,omitempty"`
}

type selectorPolicyFingerprint struct {
	Classifiers []routerconfig.ClassifierSignalRule `json:"classifiers,omitempty"`
	Projections routerconfig.CanonicalProjections   `json:"projections"`
	Decisions   []selectorDecisionFingerprint       `json:"decisions,omitempty"`
}

type adaptationDecisionFingerprint struct {
	Adaptations routerconfig.DecisionAdaptationsConfig `json:"adaptations"`
}

type policyRecipeFingerprint struct {
	Name    string                   `json:"name"`
	Routing policyRoutingFingerprint `json:"routing"`
}

func policyRoutingFromCanonical(routing routerconfig.CanonicalRouting) policyRoutingFingerprint {
	signals := routing.Signals
	// Classifier configuration is a selector factor. Keeping it out of the
	// Recipe factor prevents one executable delta from being declared as either
	// a Recipe or selector treatment.
	signals.Classifiers = nil
	decisions := make([]routerconfig.Decision, len(routing.Decisions))
	for index, decision := range routing.Decisions {
		// Candidate identity, selector algorithms, and online adaptations each
		// have their own factor. The Recipe factor retains the decision/rule/plugin
		// structure while removing those independently testable treatments.
		decision.ModelRefs = nil
		decision.Algorithm = nil
		decision.Adaptations = routerconfig.DecisionAdaptationsConfig{}
		// Human-facing descriptions and transport/replay annotations are
		// explicitly non-executable config metadata. They remain in the raw config
		// lineage, but cannot define a routing-policy treatment.
		decision.Description = ""
		decision.Annotations = nil
		if len(decision.CandidateIterations) > 0 {
			iterations := append([]routerconfig.CandidateIterationConfig(nil), decision.CandidateIterations...)
			for iterationIndex := range iterations {
				iterations[iterationIndex].Models = nil
			}
			decision.CandidateIterations = iterations
		}
		decisions[index] = decision
	}
	return policyRoutingFingerprint{
		Signals: signals, Decisions: decisions, Strategy: routing.Strategy,
	}
}

func selectorPolicySnapshotDigest(routing routerconfig.CanonicalRouting) string {
	decisions := make([]selectorDecisionFingerprint, 0, len(routing.Decisions))
	for _, decision := range routing.Decisions {
		decisions = append(decisions, selectorDecisionFingerprint{
			Algorithm: decision.Algorithm,
		})
	}
	return digestJSON(selectorPolicyFingerprint{
		Classifiers: append([]routerconfig.ClassifierSignalRule(nil), routing.Signals.Classifiers...),
		Projections: routing.Projections,
		Decisions:   decisions,
	})
}

func adaptationSnapshotDigest(routing routerconfig.CanonicalRouting) string {
	decisions := make([]adaptationDecisionFingerprint, 0, len(routing.Decisions))
	for _, decision := range routing.Decisions {
		decisions = append(decisions, adaptationDecisionFingerprint{
			Adaptations: decision.Adaptations,
		})
	}
	return digestJSON(struct {
		Decisions []adaptationDecisionFingerprint `json:"decisions"`
	}{Decisions: decisions})
}

func appendUniqueStrings(existing []string, values ...string) []string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" && !containsString(existing, value) {
			existing = append(existing, value)
		}
	}
	return existing
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
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

func backendTopologyDigestForModels(
	canonical routerconfig.CanonicalConfig,
	allowedModels map[string]struct{},
) string {
	if len(allowedModels) == 0 {
		return ""
	}
	models := make([]topologyModelFingerprint, 0, len(canonical.Providers.Models))
	for _, provider := range canonical.Providers.Models {
		modelName := strings.TrimSpace(provider.Name)
		if _, allowed := allowedModels[modelName]; !allowed || len(provider.BackendRefs) == 0 {
			continue
		}
		identity := strings.TrimSpace(provider.ProviderModelID)
		if identity == "" {
			identity = strings.TrimSpace(provider.Name)
		}
		model := topologyModelFingerprint{
			Model: modelName, ProviderModelIDDigest: digestString(identity),
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
	if len(models) != len(allowedModels) {
		return ""
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
	return boundedOptionalString(value, 160)
}

func copyStringPointer(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func stringPointer(value string) *string {
	return &value
}
