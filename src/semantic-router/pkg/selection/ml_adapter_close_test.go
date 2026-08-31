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

package selection

import (
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
)

type countingMLSelector struct {
	closes   int
	closeErr error
}

func (s *countingMLSelector) Select(_ *modelselection.SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	if len(refs) == 0 {
		return nil, errors.New("no refs")
	}
	return &refs[0], nil
}

func (s *countingMLSelector) Name() string { return "counting" }

func (s *countingMLSelector) Train(_ []modelselection.TrainingRecord) error { return nil }

func (s *countingMLSelector) Close() error {
	s.closes++
	return s.closeErr
}

type nonClosingMLSelector struct{}

func (nonClosingMLSelector) Select(_ *modelselection.SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	return nil, nil
}

func (nonClosingMLSelector) Name() string { return "non-closing" }

func (nonClosingMLSelector) Train(_ []modelselection.TrainingRecord) error { return nil }

func TestMLSelectorAdapter_CloseForwardsToMLSelector(t *testing.T) {
	ml := &countingMLSelector{}
	adapter := NewMLSelectorAdapter(ml, MethodKNN)

	if err := adapter.Close(); err != nil {
		t.Fatalf("Close returned %v, want nil", err)
	}
	if got := ml.closes; got != 1 {
		t.Fatalf("wrapped selector closed %d times, want 1", got)
	}
}

func TestMLSelectorAdapter_CloseSurfacesError(t *testing.T) {
	wantErr := errors.New("handle release failed")
	adapter := NewMLSelectorAdapter(&countingMLSelector{closeErr: wantErr}, MethodSVM)

	if err := adapter.Close(); !errors.Is(err, wantErr) {
		t.Fatalf("Close returned %v, want %v", err, wantErr)
	}
}

func TestMLSelectorAdapter_CloseWithoutCloseableSelector(t *testing.T) {
	if err := NewMLSelectorAdapter(nonClosingMLSelector{}, MethodKMeans).Close(); err != nil {
		t.Fatalf("Close returned %v, want nil for a selector holding no native state", err)
	}
	if err := NewMLSelectorAdapter(nil, MethodMLP).Close(); err != nil {
		t.Fatalf("Close returned %v, want nil for an adapter with no selector", err)
	}
}

func TestRegistryClose_ReachesMLAdapters(t *testing.T) {
	mls := map[SelectionMethod]*countingMLSelector{
		MethodKNN:    {},
		MethodKMeans: {},
		MethodSVM:    {},
		MethodMLP:    {},
	}

	registry := NewRegistry()
	for method, ml := range mls {
		registry.Register(method, NewMLSelectorAdapter(ml, method))
	}
	registry.Register(MethodStatic, NewStaticSelector(DefaultStaticConfig()))

	if err := registry.Close(); err != nil {
		t.Fatalf("Registry.Close returned %v, want nil", err)
	}

	for method, ml := range mls {
		if got := ml.closes; got != 1 {
			t.Errorf("%s selector closed %d times, want 1", method, got)
		}
	}
}

func TestRegistryClose_JoinsMLAdapterErrors(t *testing.T) {
	knnErr := errors.New("knn handle release failed")
	svmErr := errors.New("svm handle release failed")

	registry := NewRegistry()
	registry.Register(MethodKNN, NewMLSelectorAdapter(&countingMLSelector{closeErr: knnErr}, MethodKNN))
	registry.Register(MethodSVM, NewMLSelectorAdapter(&countingMLSelector{closeErr: svmErr}, MethodSVM))

	err := registry.Close()
	if !errors.Is(err, knnErr) || !errors.Is(err, svmErr) {
		t.Fatalf("Registry.Close returned %v, want both %v and %v joined", err, knnErr, svmErr)
	}
}
