package routerprojection

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"sort"
	"strings"
	"time"

	routerauthoring "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerauthoring"
	routercontract "github.com/vllm-project/semantic-router/src/semantic-router/pkg/routercontract"
)

const projectionSchemaVersion = "v1"

type ActiveConfigProjection struct {
	SchemaVersion string                `json:"schema_version"`
	ConfigVersion string                `json:"config_version"`
	ConfigHash    string                `json:"config_hash"`
	GeneratedAt   string                `json:"generated_at"`
	Models        []ProjectedModel      `json:"models"`
	Signals       []ProjectedSignal     `json:"signals"`
	Decisions     []ProjectedDecision   `json:"decisions"`
	Plugins       []ProjectedPlugin     `json:"plugins"`
	Validation    ProjectedValidation   `json:"validation"`
	DSLSnapshot   *ProjectedDSLSnapshot `json:"dsl_snapshot,omitempty"`
}

type ProjectedModel struct {
	Name             string   `json:"name"`
	ReasoningFamily  string   `json:"reasoning_family,omitempty"`
	ProviderModelID  string   `json:"provider_model_id,omitempty"`
	Capabilities     []string `json:"capabilities,omitempty"`
	PreferredBackend []string `json:"preferred_backends,omitempty"`
}

type ProjectedSignal struct {
	Type string `json:"type"`
	Name string `json:"name"`
}

type ProjectedDecision struct {
	Name     string   `json:"name"`
	Priority int      `json:"priority"`
	Models   []string `json:"models,omitempty"`
	Plugins  []string `json:"plugins,omitempty"`
	HasRules bool     `json:"has_rules"`
	Fallback bool     `json:"fallback,omitempty"`
}

type ProjectedPlugin struct {
	Type         string `json:"type"`
	DecisionName string `json:"decision_name"`
}

type ProjectedValidation struct {
	Valid  bool     `json:"valid"`
	Errors []string `json:"errors,omitempty"`
}

type ProjectedDSLSnapshot struct {
	Source string `json:"source"`
	Text   string `json:"text"`
}

func BuildActiveConfigProjectionFromFile(configPath string, dslSnapshotPath string) (*ActiveConfigProjection, error) {
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}

	var dslData []byte
	if strings.TrimSpace(dslSnapshotPath) != "" {
		if raw, readErr := os.ReadFile(dslSnapshotPath); readErr == nil {
			dslData = raw
		}
	}

	return BuildActiveConfigProjection(configData, dslData)
}

func BuildActiveConfigProjection(configData []byte, dslData []byte) (*ActiveConfigProjection, error) {
	canonical, err := routercontract.ParseYAMLBytes(configData)
	if err != nil {
		return nil, err
	}

	projection := &ActiveConfigProjection{
		SchemaVersion: projectionSchemaVersion,
		ConfigVersion: canonical.Version,
		ConfigHash:    HashConfigBytes(configData),
		GeneratedAt:   time.Now().UTC().Format(time.RFC3339),
		Models:        projectModels(canonical),
		Signals:       projectSignals(canonical),
		Decisions:     projectDecisions(canonical),
		Plugins:       projectPlugins(canonical),
		Validation: ProjectedValidation{
			Valid: true,
		},
	}

	if snapshot := buildDSLSnapshot(canonical, dslData); snapshot != nil {
		projection.DSLSnapshot = snapshot
	}

	return projection, nil
}

func HashConfigBytes(configData []byte) string {
	sum := sha256.Sum256(configData)
	return hex.EncodeToString(sum[:])
}

func buildDSLSnapshot(canonical *routercontract.CanonicalConfig, dslData []byte) *ProjectedDSLSnapshot {
	if trimmed := strings.TrimSpace(string(dslData)); trimmed != "" {
		return &ProjectedDSLSnapshot{
			Source: "archived_source",
			Text:   trimmed,
		}
	}

	text, err := routerauthoring.DecompileRouting(canonical)
	if err != nil {
		return nil
	}
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return nil
	}
	return &ProjectedDSLSnapshot{
		Source: "decompiled_routing",
		Text:   trimmed,
	}
}

func projectModels(canonical *routercontract.CanonicalConfig) []ProjectedModel {
	modelCards := map[string]routercontract.RoutingModel{}
	for _, card := range canonical.Routing.ModelCards {
		modelCards[card.Name] = card
	}

	models := make([]ProjectedModel, 0, len(canonical.Providers.Models))
	for _, model := range canonical.Providers.Models {
		card := modelCards[model.Name]
		projected := ProjectedModel{
			Name:            model.Name,
			ReasoningFamily: model.ReasoningFamily,
			ProviderModelID: model.ProviderModelID,
			Capabilities:    append([]string(nil), card.Capabilities...),
		}
		for _, backend := range model.BackendRefs {
			if strings.TrimSpace(backend.Name) != "" {
				projected.PreferredBackend = append(projected.PreferredBackend, backend.Name)
			}
		}
		models = append(models, projected)
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Name < models[j].Name })
	return models
}

func projectSignals(canonical *routercontract.CanonicalConfig) []ProjectedSignal {
	signals := []ProjectedSignal{}
	appendNamedSignals := func(signalType string, names []string) {
		for _, name := range names {
			signals = append(signals, ProjectedSignal{Type: signalType, Name: name})
		}
	}

	appendNamedSignals("keyword", keywordSignalNames(canonical.Routing.Signals.Keywords))
	appendNamedSignals("embedding", embeddingSignalNames(canonical.Routing.Signals.Embeddings))
	appendNamedSignals("domain", domainSignalNames(canonical.Routing.Signals.Domains))
	appendNamedSignals("fact_check", factCheckSignalNames(canonical.Routing.Signals.FactCheck))
	appendNamedSignals("user_feedback", userFeedbackSignalNames(canonical.Routing.Signals.UserFeedbacks))
	appendNamedSignals("reask", reaskSignalNames(canonical.Routing.Signals.Reasks))
	appendNamedSignals("preference", preferenceSignalNames(canonical.Routing.Signals.Preferences))
	appendNamedSignals("language", languageSignalNames(canonical.Routing.Signals.Language))
	appendNamedSignals("context", contextSignalNames(canonical.Routing.Signals.Context))
	appendNamedSignals("structure", structureSignalNames(canonical.Routing.Signals.Structure))
	appendNamedSignals("complexity", complexitySignalNames(canonical.Routing.Signals.Complexity))
	appendNamedSignals("modality", modalitySignalNames(canonical.Routing.Signals.Modality))
	appendNamedSignals("role_binding", roleBindingSignalNames(canonical.Routing.Signals.RoleBindings))
	appendNamedSignals("jailbreak", jailbreakSignalNames(canonical.Routing.Signals.Jailbreak))
	appendNamedSignals("pii", piiSignalNames(canonical.Routing.Signals.PII))
	appendNamedSignals("kb", kbSignalNames(canonical.Routing.Signals.KB))

	sort.Slice(signals, func(i, j int) bool {
		if signals[i].Type == signals[j].Type {
			return signals[i].Name < signals[j].Name
		}
		return signals[i].Type < signals[j].Type
	})
	return signals
}

func projectDecisions(canonical *routercontract.CanonicalConfig) []ProjectedDecision {
	decisions := make([]ProjectedDecision, 0, len(canonical.Routing.Decisions))
	for _, decision := range canonical.Routing.Decisions {
		projected := ProjectedDecision{
			Name:     decision.Name,
			Priority: decision.Priority,
			HasRules: decision.Rules.Operator != "",
			Fallback: len(decision.ModelRefs) == 0,
		}
		for _, modelRef := range decision.ModelRefs {
			if strings.TrimSpace(modelRef.Model) != "" {
				projected.Models = append(projected.Models, modelRef.Model)
			}
		}
		for _, plugin := range decision.Plugins {
			if strings.TrimSpace(plugin.Type) != "" {
				projected.Plugins = append(projected.Plugins, plugin.Type)
			}
		}
		sort.Strings(projected.Models)
		sort.Strings(projected.Plugins)
		decisions = append(decisions, projected)
	}
	sort.Slice(decisions, func(i, j int) bool {
		if decisions[i].Priority == decisions[j].Priority {
			return decisions[i].Name < decisions[j].Name
		}
		return decisions[i].Priority < decisions[j].Priority
	})
	return decisions
}

func projectPlugins(canonical *routercontract.CanonicalConfig) []ProjectedPlugin {
	plugins := []ProjectedPlugin{}
	for _, decision := range canonical.Routing.Decisions {
		for _, plugin := range decision.Plugins {
			if strings.TrimSpace(plugin.Type) == "" {
				continue
			}
			plugins = append(plugins, ProjectedPlugin{
				Type:         plugin.Type,
				DecisionName: decision.Name,
			})
		}
	}
	sort.Slice(plugins, func(i, j int) bool {
		if plugins[i].Type == plugins[j].Type {
			return plugins[i].DecisionName < plugins[j].DecisionName
		}
		return plugins[i].Type < plugins[j].Type
	})
	return plugins
}

func keywordSignalNames(rules []routercontract.KeywordRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func embeddingSignalNames(rules []routercontract.EmbeddingRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func domainSignalNames(rules []routercontract.Category) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func factCheckSignalNames(rules []routercontract.FactCheckRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func userFeedbackSignalNames(rules []routercontract.UserFeedbackRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func reaskSignalNames(rules []routercontract.ReaskRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func preferenceSignalNames(rules []routercontract.PreferenceRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func languageSignalNames(rules []routercontract.LanguageRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func contextSignalNames(rules []routercontract.ContextRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func structureSignalNames(rules []routercontract.StructureRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func complexitySignalNames(rules []routercontract.ComplexityRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func modalitySignalNames(rules []routercontract.ModalityRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func roleBindingSignalNames(rules []routercontract.RoleBinding) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func jailbreakSignalNames(rules []routercontract.JailbreakRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func piiSignalNames(rules []routercontract.PIIRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func kbSignalNames(rules []routercontract.KBSignalRule) []string {
	return signalNames(len(rules), func(i int) string { return rules[i].Name })
}

func signalNames(count int, getName func(int) string) []string {
	names := make([]string, 0, count)
	for i := 0; i < count; i++ {
		name := strings.TrimSpace(getName(i))
		if name != "" {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
