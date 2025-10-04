// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
