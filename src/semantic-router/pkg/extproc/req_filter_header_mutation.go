package extproc

import (
	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	"google.golang.org/protobuf/types/known/wrapperspb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// buildHeaderMutations builds header mutations based on the header_mutation plugin the request runs
// Returns (setHeaders, removeHeaders) to be applied to the request
func (r *OpenAIRouter) buildHeaderMutations(ctx *RequestContext) ([]*corev3.HeaderValueOption, []string) {
	// Get header mutation configuration
	headerConfig := config.HeaderMutationPluginOf(requestPlugins(r.Config, ctx))
	if headerConfig == nil {
		return nil, nil
	}

	logging.ComponentDebugEvent("extproc", "header_mutations_prepared", map[string]interface{}{
		"request_id":   ctx.RequestID,
		"add_count":    len(headerConfig.Add),
		"update_count": len(headerConfig.Update),
		"delete_count": len(headerConfig.Delete),
	})

	var setHeaders []*corev3.HeaderValueOption
	var removeHeaders []string

	for _, mutation := range pluginruntime.HeaderMutations(headerConfig) {
		if mutation.Operation == "remove" {
			removeHeaders = append(removeHeaders, mutation.Name)
			continue
		}
		header := &corev3.HeaderValueOption{Header: &corev3.HeaderValue{Key: mutation.Name, RawValue: []byte(mutation.Value)}}
		if mutation.Operation == "set" {
			header.AppendAction = corev3.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD
		} else {
			// "add" keeps an existing value. Envoy's ext_proc filter reads the
			// deprecated append flag and ignores append_action, so the enum
			// alone would still overwrite: set both.
			header.Append = &wrapperspb.BoolValue{Value: true}
			header.AppendAction = corev3.HeaderValueOption_APPEND_IF_EXISTS_OR_ADD
		}
		setHeaders = append(setHeaders, header)
	}

	return setHeaders, removeHeaders
}
