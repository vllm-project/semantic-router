/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"fmt"

	"github.com/openai/openai-go"
)

// ModelTarget identifies one concrete model deployment for a Looper call.
// AccessKey is request-scoped because different candidate models may use
// different credentials.
type ModelTarget struct {
	Name      string
	AccessKey string
}

// ResponseMode describes the wire response expected from the upstream model.
type ResponseMode uint8

const (
	ResponseJSON ResponseMode = iota
	ResponseSSE
)

// CallOptions carries request-scoped Looper execution metadata. Keeping this
// data off Client allows one Client to be reused safely by concurrent calls.
type CallOptions struct {
	DecisionName string
	Iteration    uint32
	FusionDepth  uint32
	Mode         ResponseMode
	Logprobs     *LogprobsConfig
}

func (options CallOptions) validate(target ModelTarget) error {
	if target.Name == "" {
		return fmt.Errorf("model target name is required")
	}
	if options.Iteration == 0 {
		return fmt.Errorf("looper iteration must be positive")
	}
	if options.Mode != ResponseJSON && options.Mode != ResponseSSE {
		return fmt.Errorf("unsupported looper response mode %d", options.Mode)
	}
	return nil
}

// CallModelWithOptions invokes a model without storing request-scoped metadata
// on Client. The input request is passed by value so model and sampling fields
// can be adjusted without mutating the caller's top-level request value.
func (c *Client) CallModelWithOptions(
	ctx context.Context,
	request openai.ChatCompletionNewParams,
	target ModelTarget,
	options CallOptions,
) (*ModelResponse, error) {
	if err := options.validate(target); err != nil {
		return nil, err
	}
	return c.callModel(
		ctx,
		&request,
		target.Name,
		options.Mode == ResponseSSE,
		int(options.Iteration),
		options.Logprobs,
		target.AccessKey,
		options.DecisionName,
		int(options.FusionDepth),
	)
}
