//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/url"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

const apiPluginsPath = apiRootPath + "/plugins"

type pluginOperation struct {
	Method string `json:"method"`
	Path   string `json:"path"`
	Mode   string `json:"mode"`
}

type pluginDescriptor struct {
	config.DecisionPluginCatalogEntry
	ConfigurationScope string            `json:"configuration_scope"`
	Schema             string            `json:"schema"`
	Bindings           string            `json:"bindings"`
	Operations         []pluginOperation `json:"operations"`
}

type pluginCatalogResponse struct {
	Plugins []pluginDescriptor `json:"plugins"`
}

type pluginDependency = pluginruntime.Dependency

type pluginBinding struct {
	Recipe           string             `json:"recipe"`
	Decision         string             `json:"decision"`
	Enabled          bool               `json:"enabled"`
	RuntimeInspected bool               `json:"runtime_inspected"`
	Reachable        bool               `json:"reachable"`
	Configuration    string             `json:"configuration"`
	Dependencies     []pluginDependency `json:"dependencies"`
	Status           string             `json:"status"`
}

type pluginBindingsResponse struct {
	Type     string          `json:"type"`
	Source   string          `json:"source"`
	Bindings []pluginBinding `json:"bindings"`
}

func apiPluginRoutes() []apiRoute {
	return append([]apiRoute{
		managedRoute(EndpointMetadata{Path: apiPluginsPath, Method: "GET", Description: "Discover every registered recipe-scoped plugin, its schema, bindings, and supported operations"}, routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic}, (*ClassificationAPIServer).handlePlugins, jsonResponse[pluginCatalogResponse](http.StatusOK, "Registered plugin catalog")),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/{type}", Method: "GET", Description: "Describe one canonical plugin type and its supported management operations"}, routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic}, (*ClassificationAPIServer).handlePlugin, jsonResponse[pluginDescriptor](http.StatusOK, "Plugin descriptor"), errorResponses(http.StatusNotFound)),
		managedRoute(EndpointMetadata{Path: apiPluginsPath + "/{type}/bindings", Method: "GET", Description: "Inspect active recipe and decision plugin bindings and published dependency availability; does not probe network health", Parameters: []OpenAPIParameter{queryParameter("recipe", "Filter by exact recipe name.", "string"), queryParameter("decision", "Filter by exact decision name.", "string")}}, routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig}, (*ClassificationAPIServer).handlePluginBindings, jsonResponse[pluginBindingsResponse](http.StatusOK, "Active plugin bindings"), errorResponses(http.StatusNotFound, http.StatusServiceUnavailable)),
	}, append(append(apiPluginPreviewRoutes(), apiPluginGuardRoutes()...), apiPluginRetrievalRoutes()...)...)
}

func pluginDescriptorFor(entry config.DecisionPluginCatalogEntry, routes []apiRoute) pluginDescriptor {
	return pluginDescriptor{DecisionPluginCatalogEntry: entry, ConfigurationScope: "recipe/decision", Schema: apiConfigSchemaPath + "?view=surface&kind=plugin&name=" + url.QueryEscape(entry.Type), Bindings: apiPluginsPath + "/" + entry.Type + "/bindings", Operations: pluginOperations(entry.Type, routes)}
}

func pluginOperations(pluginType string, routes []apiRoute) []pluginOperation {
	operations := []pluginOperation{}
	for _, route := range routes {
		for _, operation := range route.PluginOperations {
			if operation.Plugin == pluginType {
				operations = append(operations, pluginOperation{Method: route.Method, Path: route.Path, Mode: operation.Mode})
			}
		}
	}
	sort.Slice(operations, func(i, j int) bool {
		if operations[i].Path == operations[j].Path {
			return operations[i].Method < operations[j].Method
		}
		return operations[i].Path < operations[j].Path
	})
	return operations
}

// pluginOperationFor links a supported operation to its canonical plugin.
// Shared resources may attach more than one owner without duplicating routes.
func pluginOperationFor(pluginType, mode string) apiRouteOption {
	return pluginOperationOption{Plugin: pluginType, Mode: mode}
}

type pluginOperationOption PluginOperationContract

func (option pluginOperationOption) applyRoute(route *apiRoute) {
	route.PluginOperations = append(route.PluginOperations, PluginOperationContract(option))
}

func canonicalPluginEntry(name string) (config.DecisionPluginCatalogEntry, bool) {
	for _, entry := range config.DecisionPluginCatalog() {
		if entry.Type == name {
			return entry, true
		}
	}
	return config.DecisionPluginCatalogEntry{}, false
}

func (s *ClassificationAPIServer) handlePlugins(w http.ResponseWriter, _ *http.Request) {
	response := pluginCatalogResponse{Plugins: []pluginDescriptor{}}
	routes := apiRoutes()
	for _, entry := range config.DecisionPluginCatalog() {
		response.Plugins = append(response.Plugins, pluginDescriptorFor(entry, routes))
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}

func (s *ClassificationAPIServer) handlePlugin(w http.ResponseWriter, r *http.Request) {
	entry, ok := canonicalPluginEntry(r.PathValue("type"))
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "PLUGIN_NOT_FOUND", "Unknown canonical plugin type")
		return
	}
	s.writeJSONResponse(w, http.StatusOK, pluginDescriptorFor(entry, apiRoutes()))
}

func (s *ClassificationAPIServer) pluginInventory() (routerruntime.PluginInventory, func(), bool) {
	if s.runtimeRegistry != nil {
		return s.runtimeRegistry.AcquirePluginInventory()
	}
	return routerruntime.PluginInventory{Config: s.currentConfig()}, func() {}, true
}

func (s *ClassificationAPIServer) handlePluginBindings(w http.ResponseWriter, r *http.Request) {
	entry, ok := canonicalPluginEntry(r.PathValue("type"))
	if !ok {
		s.writeErrorResponse(w, http.StatusNotFound, "PLUGIN_NOT_FOUND", "Unknown canonical plugin type")
		return
	}
	inventory, release, ok := s.pluginInventory()
	defer release()
	if !ok || inventory.Config == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "RUNTIME_UNAVAILABLE", "Active router configuration is unavailable")
		return
	}
	reachable := map[config.RecipeName]bool{}
	for _, recipe := range inventory.Config.ReachableRoutingRecipes() {
		reachable[recipe.Name] = true
	}
	response := pluginBindingsResponse{Type: entry.Type, Source: "active_runtime", Bindings: []pluginBinding{}}
	for _, ref := range inventory.Config.RoutingDecisionRefs() {
		if recipe := r.URL.Query().Get("recipe"); recipe != "" && recipe != string(ref.Recipe) {
			continue
		}
		if decision := r.URL.Query().Get("decision"); decision != "" && decision != ref.Decision.Name {
			continue
		}
		plugin := ref.Decision.GetPlugin(entry.Type)
		if plugin == nil {
			continue
		}
		binding := pluginBinding{Recipe: string(ref.Recipe), Decision: ref.Decision.Name, Reachable: reachable[ref.Recipe], Configuration: apiRecipesPath + "/" + url.PathEscape(string(ref.Recipe)), Dependencies: []pluginDependency{}, Status: "configured"}
		payload, err := config.DecodeDecisionPlugin(*plugin)
		if err != nil {
			binding.Status = "invalid_configuration"
		} else {
			binding.Enabled = config.DecisionPluginEnabled(payload)
			if inventory.Inspector != nil {
				if dependencies, inspectErr := inventory.Inspector.InspectPluginBinding(pluginruntime.Binding{Recipe: ref.Recipe, Decision: ref.Decision.Name}, entry.Type); inspectErr == nil {
					binding.Dependencies = dependencies
					binding.RuntimeInspected = true
				}
			}
			if !binding.Enabled {
				binding.Status = "disabled"
			} else if !binding.Reachable {
				binding.Status = "unreachable"
			} else if !binding.RuntimeInspected {
				binding.Status = "not_inspected"
			} else {
				for _, dependency := range binding.Dependencies {
					if dependency.Availability == "unavailable" {
						binding.Status = "dependency_unavailable"
						break
					}
				}
			}
		}
		response.Bindings = append(response.Bindings, binding)
	}
	s.writeJSONResponse(w, http.StatusOK, response)
}
