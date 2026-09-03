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

const freedHandleErr = "selector not initialized"

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

	if !s.mlKNN.IsTrained() {
		t.Fatal("loaded handle is not usable after the swap")
	}
	if _, err := s.mlKNN.Select([]float64{1, 0}); err != nil {
		t.Fatalf("loaded handle failed to select: %v", err)
	}
}

func TestKNNSelector_ReplaceMLKNNIdempotentSwap(t *testing.T) {
	s := NewKNNSelector(1)
	handle := s.mlKNN

	s.replaceMLKNN(handle)

	if s.mlKNN != handle {
		t.Fatal("self-swap replaced the handle")
	}
	if _, err := handle.Select([]float64{1, 0}); err != nil && err.Error() == freedHandleErr {
		t.Fatal("self-swap freed the handle it was installing")
	}
}

func TestSelectorsCloseNativeHandles(t *testing.T) {
	tests := []struct {
		name        string
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
