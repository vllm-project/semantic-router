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
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
			got, err := Factory(&config.LooperConfig{}, tt.algorithmType)
			if err != nil {
				t.Fatalf("Factory(%q) returned error: %v", tt.algorithmType, err)
			}
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
