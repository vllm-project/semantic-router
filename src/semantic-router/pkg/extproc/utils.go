package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// sendResponse sends a response with proper error handling and logging
func sendResponse(stream ext_proc.ExternalProcessor_ProcessServer, response *ext_proc.ProcessingResponse, msgType string) error {
	observability.Debugf("Sending at Stage [%s]: %+v", msgType, response)

	// Debug: dump response structure if needed
	if err := stream.Send(response); err != nil {
		observability.Errorf("Error sending %s response: %v", msgType, err)
		return err
	}
	observability.Debugf("Successfully sent %s response", msgType)
	return nil
}
