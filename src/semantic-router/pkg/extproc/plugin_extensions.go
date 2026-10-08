package extproc

import (
	"context"
	"sync"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// extensionTypes caches, per registered plugin type, whether its payload
// implements a runtime plugin interface.
var extensionTypes sync.Map

func runsAsExtension(pluginType string) bool {
	normalized := config.NormalizeDecisionPluginType(pluginType)
	if runs, ok := extensionTypes.Load(normalized); ok {
		return runs.(bool)
	}
	spec, ok := config.DecisionPlugins.Lookup(normalized)
	if !ok {
		return false
	}
	payload := spec.NewPayload()
	_, request := payload.(pluginruntime.RequestPlugin)
	_, response := payload.(pluginruntime.ResponsePlugin)
	extensionTypes.Store(normalized, request || response)
	return request || response
}

// extensionPayload returns the decoded payload of a decision's extension
// plugin, or nil for a built-in one. A generation decodes each payload once.
func (r *OpenAIRouter) extensionPayload(decision *config.Decision, index int) interface{} {
	plugin := decision.Plugins[index]
	if plugin.Configuration == nil || !runsAsExtension(plugin.Type) {
		return nil
	}
	if payload, ok := r.extensions.Load(plugin.Configuration); ok {
		return payload
	}
	payload, err := config.DecodeDecisionPluginAt(
		config.PluginAt{Decision: decision.Name, Index: index, Type: plugin.Type}, plugin)
	if err != nil {
		logExtensionFailure(decision, index, "decode", err)
		payload = nil
	}
	r.extensions.Store(plugin.Configuration, payload)
	return payload
}

// runRequestExtensions runs the selected decision's extension plugins on the
// provider-bound request. A plugin that fails is logged and its changes are
// dropped; the request continues.
func (r *OpenAIRouter) runRequestExtensions(state *routeHeaderState, ctx *RequestContext) {
	decision := ctx.VSRSelectedDecision
	for index := range decision.Plugins {
		plugin, ok := r.extensionPayload(decision, index).(pluginruntime.RequestPlugin)
		if !ok {
			continue
		}
		request := pluginruntime.NewPluginRequest(decision.Name, ctx.VSRSelectedModel,
			func(name string) string { return ctx.Headers[name] })
		if err := plugin.OnRequest(extensionContext(ctx), request); err != nil {
			logExtensionFailure(decision, index, "request", err)
			continue
		}
		for _, mutation := range request.Mutations() {
			if mutation.Operation == "remove" {
				state.removeHeaders = append(state.removeHeaders, mutation.Name)
				continue
			}
			state.setHeaders = append(state.setHeaders, &core.HeaderValueOption{
				Header:       &core.HeaderValue{Key: mutation.Name, RawValue: []byte(mutation.Value)},
				AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
			})
		}
	}
}

// runResponseExtensions runs the selected decision's extension plugins on the
// upstream response's headers.
func (r *OpenAIRouter) runResponseExtensions(ctx *RequestContext, status int) *ext_proc.HeaderMutation {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	decision := ctx.VSRSelectedDecision
	var set []*core.HeaderValueOption
	for index := range decision.Plugins {
		plugin, ok := r.extensionPayload(decision, index).(pluginruntime.ResponsePlugin)
		if !ok {
			continue
		}
		response := pluginruntime.NewPluginResponse(decision.Name, ctx.VSRSelectedModel, status)
		if err := plugin.OnResponse(extensionContext(ctx), response); err != nil {
			logExtensionFailure(decision, index, "response", err)
			continue
		}
		for _, mutation := range response.Mutations() {
			set = append(set, &core.HeaderValueOption{
				Header:       &core.HeaderValue{Key: mutation.Name, RawValue: []byte(mutation.Value)},
				AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
			})
		}
	}
	if len(set) == 0 {
		return nil
	}
	return &ext_proc.HeaderMutation{SetHeaders: set}
}

// extensionAlgorithmSelector is the selector of a decision whose algorithm
// type is registered outside the Router, or nil for a built-in one. A
// generation decodes each such block once.
func (r *OpenAIRouter) extensionAlgorithmSelector(method selection.SelectionMethod, algorithm *config.AlgorithmConfig, ctx *RequestContext) selection.Selector {
	if algorithm == nil {
		return nil
	}
	if cached, ok := r.extensions.Load(algorithm); ok {
		selector, _ := cached.(selection.Selector)
		return selector
	}
	decision := ""
	if ctx != nil && ctx.VSRSelectedDecision != nil {
		decision = ctx.VSRSelectedDecision.Name
	}
	var selector selection.Selector
	payload, registered, err := config.DecodeDecisionAlgorithm(decision, algorithm)
	switch implementation, ok := payload.(selection.ExtensionAlgorithm); {
	case !registered:
	case err != nil:
		logging.ComponentErrorEvent("extproc", "decision_algorithm_failed", map[string]interface{}{
			"decision": decision, "algorithm": algorithm.Type, "error": err.Error(),
		})
	case !ok:
		logging.ComponentErrorEvent("extproc", "decision_algorithm_failed", map[string]interface{}{
			"decision": decision, "algorithm": algorithm.Type, "error": "the algorithm's payload does not implement selection.ExtensionAlgorithm",
		})
	default:
		selector = selection.NewExtensionSelector(method, implementation)
	}
	r.extensions.Store(algorithm, selector)
	return selector
}

func extensionContext(ctx *RequestContext) context.Context {
	if ctx.TraceContext != nil {
		return ctx.TraceContext
	}
	return context.Background()
}

func logExtensionFailure(decision *config.Decision, index int, phase string, err error) {
	logging.ComponentErrorEvent("extproc", "decision_plugin_failed", map[string]interface{}{
		"decision": decision.Name,
		"plugin":   decision.Plugins[index].Type,
		"phase":    phase,
		"error":    err.Error(),
	})
}
