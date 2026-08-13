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

package modelselection

import (
	"io"
	"testing"
)

// The Rust bindings report a freed handle by returning this exact error from
// Select — the Go wrapper nils its handle under a write lock in Close, and
// Select checks for nil before crossing the FFI boundary. That makes it the
// only black-box evidence available that an allocation was actually released,
// since a live-but-untrained handle fails with a different message.
const freedHandleErr = "selector not initialized"

// knnModelJSON is a two-sample KNN model small enough to inline. It has to
// satisfy both the Go metadata parse (SavedModelData) and the Rust
// KNNModelData deserializer, which is why the embeddings/labels/qualities
// arrays are all present and equal length. Inlining it keeps this test off any
// pretrained model file on disk.
const knnModelJSON = `{
  "version": "1.0",
  "algorithm": "knn",
  "trained": true,
  "k": 1,
  "embeddings": [[1.0, 0.0], [0.0, 1.0]],
  "labels": ["model-a", "model-b"],
  "qualities": [0.9, 0.5],
  "latencies": [1000, 2000]
}`

// TestKNNSelector_LoadFromJSONFreesDisplacedHandle covers the leak where a
// load installed a fresh Rust allocation over the one the constructor had
// already made, stranding it for the process lifetime.
func TestKNNSelector_LoadFromJSONFreesDisplacedHandle(t *testing.T) {
	s := NewKNNSelector(1)
	constructed := s.mlKNN
	if constructed == nil {
		t.Fatal("constructor did not allocate a native handle")
	}

	if err := s.LoadFromJSON([]byte(knnModelJSON)); err != nil {
		t.Fatalf("LoadFromJSON failed: %v", err)
	}

	if s.mlKNN == constructed {
		t.Fatal("LoadFromJSON reused the constructed handle; test can no longer detect the leak")
	}
	if _, err := constructed.Select([]float64{1, 0}); err == nil || err.Error() != freedHandleErr {
		t.Fatalf("displaced handle was not freed: Select returned %v, want %q", err, freedHandleErr)
	}

	// The replacement must survive the swap that freed its predecessor.
	if !s.mlKNN.IsTrained() {
		t.Fatal("loaded handle is not usable after the swap")
	}
	if _, err := s.mlKNN.Select([]float64{1, 0}); err != nil {
		t.Fatalf("loaded handle failed to select: %v", err)
	}
}

// TestKNNSelector_SetMLKNNIdempotentSwap guards the "exactly once" half of the
// fix: installing the handle that is already installed must not free it, or
// the swap would hand Rust a pointer it has already dropped.
func TestKNNSelector_SetMLKNNIdempotentSwap(t *testing.T) {
	s := NewKNNSelector(1)
	handle := s.mlKNN

	s.setMLKNN(handle)

	if s.mlKNN != handle {
		t.Fatal("self-swap replaced the handle")
	}
	if _, err := handle.Select([]float64{1, 0}); err != nil && err.Error() == freedHandleErr {
		t.Fatal("self-swap freed the handle it was installing")
	}
}

// TestSelectorsCloseNativeHandles asserts each Rust-backed selector releases
// its handle on Close and tolerates a second Close — Generation registers one
// closer per construction step and a retried or duplicated shutdown must not
// double-free.
func TestSelectorsCloseNativeHandles(t *testing.T) {
	tests := []struct {
		name string
		// newSelector returns the selector plus a probe reporting whether its
		// native handle field has been cleared. The field types differ per
		// selector, so the probe closes over the concrete one.
		newSelector func() (io.Closer, func() bool)
	}{
		{
			name: "knn",
			newSelector: func() (io.Closer, func() bool) {
				s := NewKNNSelector(3)
				return s, func() bool { return s.mlKNN == nil }
			},
		},
		{
			name: "kmeans",
			newSelector: func() (io.Closer, func() bool) {
				s := NewKMeansSelector(4)
				return s, func() bool { return s.mlKMeans == nil }
			},
		},
		{
			name: "svm",
			newSelector: func() (io.Closer, func() bool) {
				s := NewSVMSelector("rbf")
				return s, func() bool { return s.mlSVM == nil }
			},
		},
		{
			name: "mlp",
			newSelector: func() (io.Closer, func() bool) {
				s := NewMLPSelector()
				return s, func() bool { return s.mlMLP == nil }
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector, handleCleared := tt.newSelector()
			if handleCleared() {
				t.Fatal("constructor did not allocate a native handle")
			}

			if err := selector.Close(); err != nil {
				t.Fatalf("Close failed: %v", err)
			}
			if !handleCleared() {
				t.Fatal("Close left the native handle in place")
			}

			if err := selector.Close(); err != nil {
				t.Fatalf("second Close failed: %v", err)
			}
		})
	}
}

// TestSelectorsImplementCloser pins the interface the selection layer depends
// on: Registry.Close discovers work through an io.Closer type assertion, which
// fails silently, so a selector that stops satisfying it stops being freed
// without any signal at runtime.
func TestSelectorsImplementCloser(t *testing.T) {
	selectors := map[string]Selector{
		"knn":    NewKNNSelector(3),
		"kmeans": NewKMeansSelector(4),
		"svm":    NewSVMSelector("rbf"),
		"mlp":    NewMLPSelector(),
	}

	for name, selector := range selectors {
		closer, ok := selector.(io.Closer)
		if !ok {
			t.Errorf("%s selector does not implement io.Closer; Registry.Close will skip it", name)
			continue
		}
		if err := closer.Close(); err != nil {
			t.Errorf("%s selector Close failed: %v", name, err)
		}
	}
}
