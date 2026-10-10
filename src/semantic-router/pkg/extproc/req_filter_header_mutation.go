package extproc

import (
	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
)

// buildHeaderMutations builds header mutations based on the decision's header_mutation plugin configuration
// Returns (setHeaders, removeHeaders) to be applied to the request
func (r *OpenAIRouter) buildHeaderMutations(decision *config.Decision) ([]*corev3.HeaderValueOption, []string) {
	if decision == nil {
		return nil, nil
	}

	// Get header mutation configuration
	headerConfig := decision.GetHeaderMutationConfig()
	if headerConfig == nil {
		return nil, nil
	}

	logging.ComponentDebugEvent("extproc", "header_mutations_prepared", map[string]interface{}{
		"decision":     decision.Name,
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
		}
		setHeaders = append(setHeaders, header)
	}

	return setHeaders, removeHeaders
}
