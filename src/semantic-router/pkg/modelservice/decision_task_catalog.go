package modelservice

import (
	"sort"
	"strconv"

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
	TaskID     string              `json:"task_id"`
	Consumer   string              `json:"consumer"`
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
	response.DefaultDeployment, _, _, _ = cfg.DecisionModelDeployment()
	for name, binding := range cfg.GlobalModelBindings {
		response.GlobalBindings[name] = binding
	}
	for _, consumer := range taskConsumers(cfg) {
		if consumer.name == "reask" {
			// There is no implicit native binding: unbound reask uses cosine
			// similarity, whose thresholds are not Noul probabilities.
			continue
		}
		name, _, ok, _ := cfg.ImplicitTaskDeployment(consumer.name)
		if !ok {
			name = response.DefaultDeployment
		}
		contract := consumer.contract
		if name == response.DefaultDeployment && (consumer.task == "pii_presence" || consumer.task == "hallucination" || consumer.task == "preference" || consumer.task == "reask" || consumer.task == "complexity") {
			contract = config.DecisionTaskContract
		}
		response.DefaultBindings[consumer.name] = config.ModelBinding{Deployment: name, Contract: contract}
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
		for _, consumer := range taskConsumers(scoped) {
			if !config.TaskConsumerInUse(scoped, recipe.Name, consumer.name) {
				continue
			}
			binding, source := recipe.Profile.ModelBindings[consumer.name], "recipe"
			if binding.Deployment == "" {
				binding, source = cfg.GlobalModelBindings[consumer.name], "global"
			}
			if binding.Deployment == "" {
				name, _, ok, _ := scoped.ImplicitTaskDeployment(consumer.name)
				if !ok {
					name = response.DefaultDeployment
				}
				binding = config.ModelBinding{Deployment: name, Contract: consumer.contract}
				if name == response.DefaultDeployment && (consumer.task == "pii_presence" || consumer.task == "hallucination" || consumer.task == "preference" || consumer.task == "reask" || consumer.task == "complexity") {
					binding.Contract = config.DecisionTaskContract
				}
				source = "default"
				if name != response.DefaultDeployment {
					source = "module"
				}
			}
			status := observed[binding.Deployment]
			model := taskCatalogModelName(status, cfg.ModelDeployments[binding.Deployment].Artifact)
			response.Bindings = append(response.Bindings, TaskCatalogBinding{
				TaskID: consumer.task, Consumer: consumer.name, Recipe: string(recipe.Name),
				Deployment: binding.Deployment, Model: model, Source: source, Ready: status.Ready, Editable: true,
				Path: append(append([]string(nil), path...), consumer.name), Binding: binding,
			})
		}
	}
	return response
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

type taskConsumer struct{ task, name, contract string }

func taskConsumers(cfg *config.RouterConfig) []taskConsumer {
	consumers := []taskConsumer{
		{"domain", "domain_classifier", config.RemoteClassifierContractLabelDistribution},
		{"fact_check", "fact_check_classifier", config.RemoteClassifierContractLabelDistribution},
		{"user_feedback", "feedback_detector", config.RemoteClassifierContractLabelDistribution},
		{"modality", "modality_detector", config.RemoteClassifierContractLabelDistribution},
		{"jailbreak", "prompt_guard", config.RemoteClassifierContractLabelDistribution},
		{"pii_presence", "pii_classifier", config.RemoteClassifierContractTokenSpans},
		{"hallucination", "hallucination_detector", config.RemoteClassifierContractTokenSpans},
		{"preference", "preference", config.DecisionTaskContract},
		{"reask", "reask", config.DecisionTaskContract},
		{"complexity", "complexity", config.DecisionTaskContract},
	}
	for _, rule := range cfg.SafetyRules {
		consumers = append(consumers, taskConsumer{"safety", "safety." + rule.Name, config.RemoteClassifierContractLabelDistribution})
		if rule.Hazard != nil {
			consumers = append(consumers, taskConsumer{"safety_categories", "safety." + rule.Name + ".hazard", config.RemoteClassifierContractLabelScores})
		}
	}
	for _, rule := range cfg.ClassifierRules {
		consumers = append(consumers, taskConsumer{"classifier", "classifier." + rule.Name, config.RemoteClassifierContractLabelDistribution})
	}
	return consumers
}
