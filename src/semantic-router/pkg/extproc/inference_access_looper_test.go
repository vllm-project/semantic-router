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

package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/usageaccounting"
)

func TestLooperDispatchCompletionConsumesNeutralTerminal(t *testing.T) {
	const requestID = "request-1"
	const dispatchID = "dispatch-1"
	store := backendinvoker.NewLocalResponseTerminalStore()
	router := &OpenAIRouter{ResponseTerminals: store}
	reference := testResponseTerminalReference(requestID, dispatchID, "model-1")
	reference.DispatchType = "looper"
	request := &RequestContext{
		RequestID: requestID,
		ManagedDispatch: &managedRequestDispatch{
			requestID: requestID,
			dispatches: []*inferenceDispatch{{
				id: dispatchID, settlementEligible: true, terminalReference: reference,
			}},
		},
	}
	if err := store.Finalize(
		context.Background(),
		testResponseTerminalPlan(reference),
		backendinvoker.AttemptResult{
			Attempt: backendinvoker.Attempt{ID: "attempt-1", Number: 1, BackendID: "backend-1"},
			State:   backendinvoker.AttemptResponseStarted,
		},
		backendinvoker.ResponseTerminal{
			Usage: testAuthoritativeUsage(11, 7), StopReason: llmprotocol.StopEndTurn,
		},
	); err != nil {
		t.Fatalf("store terminal: %v", err)
	}

	observer := &inferenceDispatchObserver{router: router, request: request}
	observer.Completed(context.Background(), looper.DispatchCompletion{
		DispatchID: dispatchID, HTTPStarted: true,
	})

	dispatch := request.ManagedDispatch.dispatches[0]
	if dispatch.state != usageaccounting.EvidenceKnownActual ||
		dispatch.usage.InputTotal.String() != "11" || dispatch.usage.Output.String() != "7" {
		t.Fatalf("dispatch evidence = state %q usage %+v", dispatch.state, dispatch.usage)
	}
	if _, found, err := store.Take(context.Background(), reference); err != nil || found {
		t.Fatal("neutral response terminal was not consumed exactly once")
	}
}

func TestLooperDispatchCompletionFencesMissingTerminal(t *testing.T) {
	request := &RequestContext{
		RequestID: "request-1",
		ManagedDispatch: &managedRequestDispatch{
			requestID: "request-1",
			dispatches: []*inferenceDispatch{{
				id: "dispatch-1", settlementEligible: true,
				terminalReference: func() backendinvoker.ResponseTerminalReference {
					reference := testResponseTerminalReference("request-1", "dispatch-1", "model-1")
					reference.DispatchType = "looper"
					return reference
				}(),
			}},
		},
	}
	observer := &inferenceDispatchObserver{
		router:  &OpenAIRouter{ResponseTerminals: backendinvoker.NewLocalResponseTerminalStore()},
		request: request,
	}
	observer.Completed(context.Background(), looper.DispatchCompletion{
		DispatchID: "dispatch-1", HTTPStarted: true,
	})

	dispatch := request.ManagedDispatch.dispatches[0]
	if dispatch.state != usageaccounting.EvidenceUnknown || dispatch.reason != "response_terminal_missing" {
		t.Fatalf("dispatch evidence = state %q reason %q", dispatch.state, dispatch.reason)
	}
}

func TestLooperDispatchCompletionBeforeHTTPIsKnownZero(t *testing.T) {
	request := &RequestContext{
		RequestID: "request-1",
		ManagedDispatch: &managedRequestDispatch{
			requestID: "request-1",
			dispatches: []*inferenceDispatch{{
				id: "dispatch-1", settlementEligible: true,
			}},
		},
	}
	observer := &inferenceDispatchObserver{router: &OpenAIRouter{}, request: request}
	observer.Completed(context.Background(), looper.DispatchCompletion{DispatchID: "dispatch-1"})

	if state := request.ManagedDispatch.dispatches[0].state; state != usageaccounting.EvidenceKnownZero {
		t.Fatalf("dispatch evidence = %q, want known_zero", state)
	}
}
