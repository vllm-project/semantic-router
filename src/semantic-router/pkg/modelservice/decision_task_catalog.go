package modelservice

import (
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type TaskCatalogDefinition struct {
	TaskDefinition
	Template TaskTemplate `json:"template"`
}

type TaskCatalogModel struct {
	Model               string           `json:"model"`
	NativeQuestionTypes []string         `json:"native_question_types"`
	Tasks               []TaskCapability `json:"tasks"`
}

type TaskCatalogDeployment struct {
	TaskCatalogModel
	Deployment string `json:"deployment"`
	Ready      bool   `json:"ready"`
}

type TaskCatalogBinding struct {
	TaskID   string `json:"task_id"`
	Consumer string `json:"consumer"`
	// Key is the binding key a configuration writes for the consumer: its
	// signal type, or the rule consumer of a rule that names its deployment.
	Key        string              `json:"key"`
	Recipe     string              `json:"recipe"`
	Deployment string              `json:"deployment"`
	Model      string              `json:"model"`
	Source     string              `json:"source"`
	Ready      bool                `json:"ready"`
	Editable   bool                `json:"editable"`
	Path       []string            `json:"path"`
	Binding    config.ModelBinding `json:"binding"`
}

type TaskCatalogResponse struct {
	DefaultDeployment string                         `json:"default_deployment"`
	GlobalBindings    map[string]config.ModelBinding `json:"global_bindings"`
	DefaultBindings   map[string]config.ModelBinding `json:"default_bindings"`
	Tasks             []TaskCatalogDefinition        `json:"tasks"`
	Models            []TaskCatalogModel             `json:"models"`
	Deployments       []TaskCatalogDeployment        `json:"deployments"`
	Bindings          []TaskCatalogBinding           `json:"bindings"`
}

// ProjectTaskCatalog is shared by Router and the persistent Dashboard control
// plane in Engine mode. Observations are supplied by the caller; this function
// never loads models, probes endpoints, or reads configuration from disk.
func ProjectTaskCatalog(cfg *config.RouterConfig, statuses []DeploymentStatus) TaskCatalogResponse {
	response := TaskCatalogResponse{GlobalBindings: map[string]config.ModelBinding{}, DefaultBindings: map[string]config.ModelBinding{}, Tasks: []TaskCatalogDefinition{}, Models: []TaskCatalogModel{}, Deployments: []TaskCatalogDeployment{}, Bindings: []TaskCatalogBinding{}}
	definitions := BuiltinTasks()
	catalogCards := make(map[string]ModelCard)
	for _, card := range CatalogDecisionCards() {
		catalogCards[card.ID] = card
		response.Models = append(response.Models, projectTaskModel(card, definitions))
	}
	for _, definition := range definitions {
		response.Tasks = append(response.Tasks, TaskCatalogDefinition{TaskDefinition: definition, Template: definition.Template()})
	}
	observed := make(map[string]DeploymentStatus, len(statuses))
	for _, status := range statuses {
		observed[status.Name] = status
		card := ModelCard{ID: status.Model}
		if status.Card != nil {
			card = *status.Card
		} else if known, ok := catalogCards[status.Artifact]; ok {
			card = known
		}
		if !card.Serves("decisions") && (cfg == nil || status.Name != cfg.DecisionModel) {
			continue
		}
		model := projectTaskModel(card, definitions)
		artifact := ""
		if cfg != nil {
			artifact = cfg.ModelDeployments[status.Name].Artifact
		}
		model.Model = taskCatalogModelName(status, artifact)
		response.Deployments = append(response.Deployments, TaskCatalogDeployment{TaskCatalogModel: model, Deployment: status.Name, Ready: status.Ready})
	}
	if cfg == nil {
		sort.Slice(response.Deployments, func(i, j int) bool { return response.Deployments[i].Deployment < response.Deployments[j].Deployment })
		return response
	}
	response.DefaultDeployment, _, _, _ = cfg.DecisionModelResource()
	for name, binding := range cfg.GlobalModelBindings {
		response.GlobalBindings[name] = binding
	}
	for _, signalType := range config.SharedAnswerSignalTypes {
		name, _ := cfg.ApplicationTypeDeployment(signalType)
		response.DefaultBindings[signalType] = config.ModelBinding{Deployment: name, Contract: config.DecisionTaskContract}
	}
	for name, resource := range cfg.ModelDeployments {
		if _, exists := observed[name]; exists {
			continue
		}
		card, known := catalogCards[resource.Artifact]
		if !known && name != response.DefaultDeployment {
			continue
		}
		if !known {
			card.ID = resource.Artifact
		}
		response.Deployments = append(response.Deployments, TaskCatalogDeployment{TaskCatalogModel: projectTaskModel(card, definitions), Deployment: name, Ready: false})
	}
	sort.Slice(response.Deployments, func(i, j int) bool { return response.Deployments[i].Deployment < response.Deployments[j].Deployment })
	profiles := cfg.Recipes
	if len(profiles) == 0 {
		profiles = []config.RoutingRecipe{{Name: config.DefaultRecipeName, Profile: config.RoutingProfile{Signals: cfg.Signals, ModelBindings: cfg.ModelBindings}}}
	}
	namedIndex := 0
	for index := range profiles {
		recipe := &profiles[index]
		scoped := cfg
		if len(cfg.Recipes) > 0 {
			scoped = cfg.ConfigForRecipe(recipe)
		}
		path := []string{"routing", "model_bindings"}
		if recipe.Name != config.DefaultRecipeName {
			path = []string{"recipes", strconv.Itoa(namedIndex), "routing", "model_bindings"}
			namedIndex++
		}
		// An application signal type the recipe asks is bound once, for
		// the rules that name no deployment of their own.
		for _, signalType := range applicationTypesInUse(scoped) {
			authored, source := scoped.ApplicationTypeBinding(signalType)
			name := authored.Deployment
			if authored.Contract == "" {
				authored.Contract = config.DecisionTaskContract
			}
			status := observed[name]
			response.Bindings = append(response.Bindings, TaskCatalogBinding{
				TaskID: PresetTask(applicationTypePreset(scoped, signalType)), Consumer: signalType, Key: signalType, Recipe: string(recipe.Name),
				Deployment: name, Model: taskCatalogModelName(status, cfg.ModelDeployments[name].Artifact), Source: source, Ready: status.Ready, Editable: true,
				Path: append(append([]string(nil), path...), signalType), Binding: authored,
			})
		}
		// A decision selector names its deployment in its algorithm block,
		// which the decision's own editor writes.
		for _, decision := range scoped.Decisions {
			if decision.Algorithm == nil || decision.Algorithm.SystemOne == nil || !strings.EqualFold(strings.TrimSpace(decision.Algorithm.Type), config.DecisionAlgorithmSystemOne) {
				continue
			}
			name, source := scoped.SystemOneSelectorDeployment(*decision.Algorithm.SystemOne), "rule"
			if strings.TrimSpace(decision.Algorithm.SystemOne.Deployment) == "" {
				source = "default"
			}
			status := observed[name]
			response.Bindings = append(response.Bindings, TaskCatalogBinding{
				TaskID: "selector", Consumer: "selector." + decision.Name, Key: "selector." + decision.Name, Recipe: string(recipe.Name),
				Deployment: name, Model: taskCatalogModelName(status, cfg.ModelDeployments[name].Artifact), Source: source, Ready: status.Ready,
				Binding: config.ModelBinding{Deployment: name, Contract: config.DecisionTaskContract},
			})
		}
	}
	return response
}

// applicationTypesInUse are the application signal types a recipe asks.
func applicationTypesInUse(scoped *config.RouterConfig) []string {
	var types []string
	for _, rule := range scoped.ApplicationRules() {
		if slices.Contains(types, rule.Type) {
			continue
		}
		if scoped.AsksApplicationRule(rule) {
			types = append(types, rule.Type)
		}
	}
	return types
}

// applicationTypePreset is the preset a recipe's first rule of a type asks.
func applicationTypePreset(scoped *config.RouterConfig, signalType string) string {
	for _, rule := range scoped.ApplicationRules() {
		if rule.Type == signalType {
			return rule.Preset
		}
	}
	return ""
}

// Runtime card IDs address logical deployments or served aliases. Display the
// observed artifact when available without changing those execution identities.
func taskCatalogModelName(status DeploymentStatus, declaredArtifact string) string {
	if status.Card != nil && status.Card.Repo != "" {
		return status.Card.Repo
	}
	if status.Artifact != "" {
		return status.Artifact
	}
	if declaredArtifact != "" {
		return declaredArtifact
	}
	if status.Card != nil && status.Card.ID != "" {
		return status.Card.ID
	}
	return status.Model
}

func projectTaskModel(card ModelCard, definitions []TaskDefinition) TaskCatalogModel {
	model := TaskCatalogModel{Model: card.ID, NativeQuestionTypes: []string{}, Tasks: []TaskCapability{}}
	for _, kind := range []string{"choice", "noul", "score", "set", "span"} {
		if card.Answers(kind) {
			model.NativeQuestionTypes = append(model.NativeQuestionTypes, kind)
		}
	}
	for _, definition := range definitions {
		model.Tasks = append(model.Tasks, CapabilityForTask(definition, card))
	}
	return model
}
