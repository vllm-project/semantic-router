package extproc

import (
	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

	logging.Debugf("Building header mutations for decision %s: add=%d, update=%d, delete=%d",
		decision.Name, len(headerConfig.Add), len(headerConfig.Update), len(headerConfig.Delete))

	var setHeaders []*corev3.HeaderValueOption
	var removeHeaders []string

	// Apply additions (add new headers)
	for _, headerPair := range headerConfig.Add {
		setHeaders = append(setHeaders, &corev3.HeaderValueOption{
			Header: &corev3.HeaderValue{
				Key:      headerPair.Name,
				RawValue: []byte(headerPair.Value),
			},
		})
		logging.Debugf("Adding header: %s=%s", headerPair.Name, headerPair.Value)
	}

	// Apply updates (modify existing headers - in Envoy this is the same as set)
	for _, headerPair := range headerConfig.Update {
		setHeaders = append(setHeaders, &corev3.HeaderValueOption{
			Header: &corev3.HeaderValue{
				Key:      headerPair.Name,
				RawValue: []byte(headerPair.Value),
			},
		})
		logging.Debugf("Updating header: %s=%s", headerPair.Name, headerPair.Value)
	}

	// Apply deletions
	for _, headerName := range headerConfig.Delete {
		removeHeaders = append(removeHeaders, headerName)
		logging.Debugf("Deleting header: %s", headerName)
	}

	return setHeaders, removeHeaders
}
