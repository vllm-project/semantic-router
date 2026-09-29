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
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
)

func TestRouterDCSelectorPropagatesEmbeddingCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})

	selector := NewRouterDCSelector(DefaultRouterDCConfig())
	selector.setContextEmbeddingFunc(func(got context.Context, _ string) ([]float32, error) {
		if got != ctx {
			t.Errorf("embedding context = %p, want request context %p", got, ctx)
		}
		close(started)
		<-got.Done()
		return nil, got.Err()
	})

	errCh := make(chan error, 1)
	go func() {
		_, err := selector.Select(ctx, &SelectionContext{
			Query:           "cancel me",
			CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		})
		errCh <- err
	}()

	waitForEmbeddingStart(t, started)
	cancel()
	assertEmbeddingCancellation(t, errCh)
}

func TestMLSelectorAdapterPropagatesEmbeddingCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	ml := &cancellationTestMLSelector{}
	adapter := NewMLSelectorAdapter(ml, MethodKNN)
	adapter.setContextEmbeddingFunc(func(got context.Context, _ string) ([]float32, error) {
		if got != ctx {
			t.Errorf("embedding context = %p, want request context %p", got, ctx)
		}
		close(started)
		<-got.Done()
		return nil, got.Err()
	})

	errCh := make(chan error, 1)
	go func() {
		_, err := adapter.Select(ctx, &SelectionContext{
			Query:           "cancel me",
			CandidateModels: []config.ModelRef{{Model: "model-a"}, {Model: "model-b"}},
		})
		errCh <- err
	}()

	waitForEmbeddingStart(t, started)
	cancel()
	assertEmbeddingCancellation(t, errCh)
	if calls := ml.calls.Load(); calls != 0 {
		t.Fatalf("ML selector calls = %d, want 0 after embedding cancellation", calls)
	}
}

func waitForEmbeddingStart(t *testing.T, started <-chan struct{}) {
	t.Helper()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("embedding callback was not entered")
	}
}

func assertEmbeddingCancellation(t *testing.T, errCh <-chan error) {
	t.Helper()
	select {
	case err := <-errCh:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("selection error = %v, want context cancellation", err)
		}
	case <-time.After(time.Second):
		t.Fatal("selection did not stop after embedding cancellation")
	}
}

type cancellationTestMLSelector struct {
	calls atomic.Int32
}

func (s *cancellationTestMLSelector) Select(_ *modelselection.SelectionContext, refs []config.ModelRef) (*config.ModelRef, error) {
	s.calls.Add(1)
	return &refs[0], nil
}

func (s *cancellationTestMLSelector) Name() string { return "cancellation-test" }

func (s *cancellationTestMLSelector) Train(_ []modelselection.TrainingRecord) error { return nil }
