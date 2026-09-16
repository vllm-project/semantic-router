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
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

type closeTrackingModelConnector struct {
	closeCalls int
}

func (*closeTrackingModelConnector) DoRequest(
	context.Context,
	connector.Operation,
	connector.Request,
) (connector.Result, error) {
	return connector.Result{}, nil
}

func (c *closeTrackingModelConnector) Close() error {
	c.closeCalls++
	return nil
}

func TestFactoryConstructsAllSupportedAlgorithms(t *testing.T) {
	tests := []struct {
		algorithmType string
		wantType      reflect.Type
	}{
		{config.DecisionAlgorithmConfidence, reflect.TypeOf((*ConfidenceLooper)(nil))},
		{config.DecisionAlgorithmFusion, reflect.TypeOf((*FusionLooper)(nil))},
		{config.DecisionAlgorithmRatings, reflect.TypeOf((*RatingsLooper)(nil))},
		{config.DecisionAlgorithmReMoM, reflect.TypeOf((*ReMoMLooper)(nil))},
		{config.DecisionAlgorithmWorkflows, reflect.TypeOf((*WorkflowsLooper)(nil))},
	}

	for _, tt := range tests {
		t.Run(tt.algorithmType, func(t *testing.T) {
			got, err := Factory(&config.LooperConfig{Endpoint: "http://unused.invalid"}, tt.algorithmType)
			if err != nil {
				t.Fatalf("Factory(%q) returned error: %v", tt.algorithmType, err)
			}
			t.Cleanup(func() { _ = got.Close() })
			if gotType := reflect.TypeOf(got); gotType != tt.wantType {
				t.Fatalf("Factory(%q) returned %v, want %v", tt.algorithmType, gotType, tt.wantType)
			}
		})
	}
}

func TestFactoryRegistryMatchesConfigCatalog(t *testing.T) {
	got := make([]string, 0, len(algorithmConstructors))
	for algorithmType := range algorithmConstructors {
		got = append(got, algorithmType)
	}
	sort.Strings(got)
	want := config.SupportedLooperAlgorithmTypes()

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Looper constructors = %v, config catalog = %v", got, want)
	}
}

func TestFactoryWithClientSharesClientAcrossAlgorithms(t *testing.T) {
	connector := &closeTrackingModelConnector{}
	client := &Client{connector: connector}
	for _, algorithmType := range config.SupportedLooperAlgorithmTypes() {
		constructed, err := FactoryWithClient(&config.LooperConfig{}, algorithmType, client)
		if err != nil {
			t.Fatalf("FactoryWithClient(%q) returned error: %v", algorithmType, err)
		}
		if got := baseLooperClient(constructed); got != client {
			t.Fatalf("FactoryWithClient(%q) client = %p, want %p", algorithmType, got, client)
		}
		managed, ok := constructed.(ManagedLooper)
		if !ok {
			t.Fatalf("FactoryWithClient(%q) returned non-closeable %T", algorithmType, constructed)
		}
		if err := managed.Close(); err != nil {
			t.Fatalf("FactoryWithClient(%q) Close() error = %v", algorithmType, err)
		}
	}
	if connector.closeCalls != 0 {
		t.Fatalf("borrowed connector closed %d times, want 0", connector.closeCalls)
	}
}

func TestBaseLooperClosesOnlyOwnedClient(t *testing.T) {
	ownedConnector := &closeTrackingModelConnector{}
	owned := newBaseLooper(
		&config.LooperConfig{},
		ownClient(&Client{connector: ownedConnector}),
	)
	if err := owned.Close(); err != nil {
		t.Fatalf("owned Close() error = %v", err)
	}
	if ownedConnector.closeCalls != 1 {
		t.Fatalf("owned connector closed %d times, want 1", ownedConnector.closeCalls)
	}

	borrowedConnector := &closeTrackingModelConnector{}
	borrowed := newBaseLooper(
		&config.LooperConfig{},
		borrowClient(&Client{connector: borrowedConnector}),
	)
	if err := borrowed.Close(); err != nil {
		t.Fatalf("borrowed Close() error = %v", err)
	}
	if borrowedConnector.closeCalls != 0 {
		t.Fatalf("borrowed connector closed %d times, want 0", borrowedConnector.closeCalls)
	}
}

func TestFactoryRejectsInvalidConnectorConfig(t *testing.T) {
	got, err := Factory(&config.LooperConfig{}, config.DecisionAlgorithmFusion)
	if got != nil {
		t.Fatalf("Factory returned %T for invalid config; want nil", got)
	}
	if err == nil {
		t.Fatal("Factory returned nil error for invalid connector config")
	}
}

func baseLooperClient(constructed Looper) *Client {
	switch typed := constructed.(type) {
	case *ConfidenceLooper:
		return typed.client
	case *FusionLooper:
		return typed.client
	case *RatingsLooper:
		return typed.client
	case *ReMoMLooper:
		return typed.client
	case *WorkflowsLooper:
		return typed.client
	default:
		return nil
	}
}

func TestFactoryRejectsUnknownAlgorithmWithoutBaseFallback(t *testing.T) {
	got, err := Factory(&config.LooperConfig{}, "unregistered")
	if got != nil {
		t.Fatalf("Factory returned %T for an unknown algorithm; want nil", got)
	}
	if err == nil {
		t.Fatal("Factory returned nil error for an unknown algorithm")
	}

	var unsupported *UnsupportedAlgorithmError
	if !errors.As(err, &unsupported) {
		t.Fatalf("Factory error = %T, want *UnsupportedAlgorithmError", err)
	}
	if unsupported.AlgorithmType != "unregistered" {
		t.Fatalf("UnsupportedAlgorithmError.AlgorithmType = %q, want %q", unsupported.AlgorithmType, "unregistered")
	}
}
