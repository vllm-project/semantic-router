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
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestValidateSelectionContextRejectsInvalidContracts(t *testing.T) {
	for _, tc := range []struct {
		name    string
		selCtx  *SelectionContext
		wantErr error
	}{
		{
			name:    "nil context",
			selCtx:  nil,
			wantErr: ErrSelectionContextRequired,
		},
		{
			name:    "empty candidates",
			selCtx:  &SelectionContext{},
			wantErr: ErrCandidateModelsRequired,
		},
		{
			name: "blank candidate model",
			selCtx: &SelectionContext{
				CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "  "}},
			},
			wantErr: ErrCandidateModelNameMissing,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateSelectionContext(tc.selCtx)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestSelectorsRejectInvalidSelectionContextWithoutPanic(t *testing.T) {
	selectors := []struct {
		name     string
		selector Selector
	}{
		{name: "static", selector: NewStaticSelector(nil)},
		{name: "elo", selector: NewEloSelector(nil)},
		{name: "router_dc", selector: NewRouterDCSelector(nil)},
		{name: "automix", selector: NewAutoMixSelector(nil)},
		{name: "hybrid", selector: NewHybridSelector(nil)},
		{name: "gmt_router", selector: NewGMTRouterSelector(nil)},
		{name: "rl_driven", selector: NewRLDrivenSelector(nil)},
		{name: "latency_aware", selector: NewLatencyAwareSelector(nil)},
		{name: "ml_adapter", selector: NewMLSelectorAdapter(nil, MethodKNN)},
		{name: "session_aware", selector: NewSessionAwareSelector(nil)},
	}

	for _, tc := range selectors {
		t.Run(tc.name+"/nil_context", func(t *testing.T) {
			_, err := tc.selector.Select(context.Background(), nil)
			if !errors.Is(err, ErrSelectionContextRequired) {
				t.Fatalf("expected %v, got %v", ErrSelectionContextRequired, err)
			}
		})

		t.Run(tc.name+"/empty_candidates", func(t *testing.T) {
			_, err := tc.selector.Select(context.Background(), &SelectionContext{})
			if !errors.Is(err, ErrCandidateModelsRequired) {
				t.Fatalf("expected %v, got %v", ErrCandidateModelsRequired, err)
			}
		})

		t.Run(tc.name+"/blank_candidate_model", func(t *testing.T) {
			_, err := tc.selector.Select(context.Background(), &SelectionContext{
				CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "\t"}},
			})
			if !errors.Is(err, ErrCandidateModelNameMissing) {
				t.Fatalf("expected %v, got %v", ErrCandidateModelNameMissing, err)
			}
		})
	}
}

func TestGlobalSelectRejectsInvalidSelectionContextBeforeFallback(t *testing.T) {
	_, err := Select(context.Background(), MethodStatic, nil)
	if !errors.Is(err, ErrSelectionContextRequired) {
		t.Fatalf("expected %v, got %v", ErrSelectionContextRequired, err)
	}

	_, err = Select(context.Background(), MethodStatic, &SelectionContext{})
	if !errors.Is(err, ErrCandidateModelsRequired) {
		t.Fatalf("expected %v, got %v", ErrCandidateModelsRequired, err)
	}

	_, err = Select(context.Background(), MethodStatic, &SelectionContext{
		CandidateModels: []config.ModelRef{{Model: ""}},
	})
	if !errors.Is(err, ErrCandidateModelNameMissing) {
		t.Fatalf("expected %v, got %v", ErrCandidateModelNameMissing, err)
	}
}

func TestSelectorCascadeEntrypointsRejectInvalidSelectionContext(t *testing.T) {
	autoMix := NewAutoMixSelector(nil)
	if _, _, err := autoMix.InitializeCascade(context.Background(), nil); !errors.Is(err, ErrSelectionContextRequired) {
		t.Fatalf("expected %v, got %v", ErrSelectionContextRequired, err)
	}

	rlCfg := DefaultRLDrivenConfig()
	rlCfg.EnableMultiRoundAggregation = true
	rlDriven := NewRLDrivenSelector(rlCfg)
	if _, err := rlDriven.SelectMultiRound(context.Background(), nil); !errors.Is(err, ErrSelectionContextRequired) {
		t.Fatalf("expected %v, got %v", ErrSelectionContextRequired, err)
	}
}
