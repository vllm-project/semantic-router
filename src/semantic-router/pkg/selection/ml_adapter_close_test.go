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
	"io"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
)

// countingMLSelector stands in for a Rust-backed modelselection.Selector. The
// real ones need the native library and give no way to observe how often their
// handle was freed, so the counter lives here instead.
type countingMLSelector struct {
	closes   atomic.Int32
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
	s.closes.Add(1)
	return s.closeErr
}

// nonClosingMLSelector is a modelselection.Selector with no native state, to
// confirm the adapter's type assertion degrades to a no-op instead of panicking.
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
	if got := ml.closes.Load(); got != 1 {
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

// TestRegistryClose_ReachesMLAdapters is the end of the chain a config reload
// depends on: Generation defers Registry.Close, which has to arrive at the
// native handle two type assertions deep.
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
	// A stateless selector alongside them: Close must not trip over it.
	registry.Register(MethodStatic, NewStaticSelector(DefaultStaticConfig()))

	if err := registry.Close(); err != nil {
		t.Fatalf("Registry.Close returned %v, want nil", err)
	}

	for method, ml := range mls {
		if got := ml.closes.Load(); got != 1 {
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

// =============================================================================
// Structural guard: everything CreateAll registers that owns a native handle
// must be reachable by Registry.Close.
// =============================================================================

var closerType = reflect.TypeOf((*io.Closer)(nil)).Elem()

// maxHandleWalkDepth bounds the value walk. Selectors compose each other
// (HybridSelector holds Elo/RouterDC/AutoMix), so the graph is deeper than one
// level but nowhere near this.
const maxHandleWalkDepth = 16

// isNativeHandleType reports whether t is a handle type owned by one of the
// Rust FFI bindings. Matching on the module path suffix rather than a list of
// type names means a binding type added later is classified automatically.
func isNativeHandleType(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return strings.HasSuffix(t.PkgPath(), "-binding")
}

// typeCanReachNativeHandle decides statically whether t could contain a binding
// handle at all. Interfaces always could, since their dynamic type is unknown,
// so the value walk still has to descend into those; everything else is settled
// here, which keeps the walk off the large float64 slices these selectors carry.
func typeCanReachNativeHandle(t reflect.Type, seen map[reflect.Type]bool) bool {
	if t == nil || seen[t] {
		return false
	}
	seen[t] = true

	if isNativeHandleType(t) {
		return true
	}

	switch t.Kind() {
	case reflect.Interface:
		return true
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return typeCanReachNativeHandle(t.Elem(), seen)
	case reflect.Map:
		return typeCanReachNativeHandle(t.Key(), seen) || typeCanReachNativeHandle(t.Elem(), seen)
	case reflect.Struct:
		for i := 0; i < t.NumField(); i++ {
			if typeCanReachNativeHandle(t.Field(i).Type, seen) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// nativeHandleWalk collects, for every binding handle reachable from v, the
// chain of pointer types that own it: outermost first, the handle type last.
//
// The walk reads unexported fields, which reflect permits as long as the values
// are never converted back with Interface() — that restriction is why the chain
// is expressed as types and the io.Closer check runs against the type rather
// than a live value.
type nativeHandleWalk struct {
	chains [][]reflect.Type
	seen   map[uintptr]bool
}

func (w *nativeHandleWalk) visit(v reflect.Value, owners []reflect.Type, depth int) {
	if depth > maxHandleWalkDepth || !v.IsValid() {
		return
	}
	if !typeCanReachNativeHandle(v.Type(), map[reflect.Type]bool{}) {
		return
	}

	switch v.Kind() {
	case reflect.Interface:
		w.visitInterface(v, owners, depth)
	case reflect.Pointer:
		w.visitPointer(v, owners, depth)
	case reflect.Struct:
		w.visitStruct(v, owners, depth)
	case reflect.Slice, reflect.Array:
		w.visitSequence(v, owners, depth)
	case reflect.Map:
		w.visitMap(v, owners, depth)
	}
}

func (w *nativeHandleWalk) visitInterface(v reflect.Value, owners []reflect.Type, depth int) {
	if !v.IsNil() {
		w.visit(v.Elem(), owners, depth+1)
	}
}

func (w *nativeHandleWalk) visitPointer(v reflect.Value, owners []reflect.Type, depth int) {
	if v.IsNil() {
		return
	}
	if isNativeHandleType(v.Type()) {
		w.chains = append(w.chains, append(append([]reflect.Type{}, owners...), v.Type()))
		return
	}
	if w.seen[v.Pointer()] {
		return
	}
	w.seen[v.Pointer()] = true
	// A pointer hop is an ownership boundary: this is the type whose Close
	// the level above has to call for anything below to be released.
	w.visit(v.Elem(), append(owners, v.Type()), depth+1)
}

func (w *nativeHandleWalk) visitStruct(v reflect.Value, owners []reflect.Type, depth int) {
	// Fields of an inlined struct belong to the enclosing pointer, so the
	// owner chain does not grow here.
	for i := 0; i < v.NumField(); i++ {
		w.visit(v.Field(i), owners, depth+1)
	}
}

func (w *nativeHandleWalk) visitSequence(v reflect.Value, owners []reflect.Type, depth int) {
	for i := 0; i < v.Len(); i++ {
		w.visit(v.Index(i), owners, depth+1)
	}
}

func (w *nativeHandleWalk) visitMap(v reflect.Value, owners []reflect.Type, depth int) {
	iter := v.MapRange()
	for iter.Next() {
		w.visit(iter.Key(), owners, depth+1)
		w.visit(iter.Value(), owners, depth+1)
	}
}

// uncloseableOwner reproduces the shape of the bug this file guards against:
// something registered as a selector that holds a Rust-backed ML selector but
// offers no Close of its own.
type uncloseableOwner struct {
	inner modelselection.Selector
}

// TestNativeHandleWalk_FlagsOwnerWithoutClose tests the walk itself rather than
// the production wiring, so the guard below cannot pass merely because the walk
// is blind. It confirms the walk sees through an interface-typed field to the
// handle underneath, and that the io.Closer predicate rejects an owner that
// would swallow the handle.
func TestNativeHandleWalk_FlagsOwnerWithoutClose(t *testing.T) {
	knn := modelselection.NewKNNSelector(3)
	t.Cleanup(func() {
		if err := knn.Close(); err != nil {
			t.Errorf("closing test KNN selector: %v", err)
		}
	})

	walk := &nativeHandleWalk{seen: map[uintptr]bool{}}
	walk.visit(reflect.ValueOf(&uncloseableOwner{inner: knn}), nil, 0)

	if len(walk.chains) != 1 {
		t.Fatalf("walk found %d native handles, want 1: %v", len(walk.chains), walk.chains)
	}

	chain := walk.chains[0]
	if got, want := chain[0], reflect.TypeOf(&uncloseableOwner{}); got != want {
		t.Fatalf("outermost owner is %s, want %s", got, want)
	}
	if chain[0].Implements(closerType) {
		t.Error("uncloseableOwner unexpectedly implements io.Closer; it cannot demonstrate the failure mode")
	}
	// The hop below it is the one the fix added; without it, nothing in the
	// chain could release the handle even if the outer type gained a Close.
	if !chain[1].Implements(closerType) {
		t.Errorf("%s does not implement io.Closer, so its native handle cannot be released", chain[1])
	}
}

// TestCreateAll_NativeHandleOwnersAreCloseable is the recurrence guard for the
// leak this file exists for. Registry.Close finds its work through an io.Closer
// type assertion that fails silently, so a fifth ML selector wired in without a
// Close would leak with no error and no log line. Instead of naming the
// selectors that hold native state, this walks the value graph of everything
// CreateAll registered, finds the binding handles, and requires every pointer
// hop between the registry and each handle to implement io.Closer — which is
// exactly the condition under which Close actually reaches it.
func TestCreateAll_NativeHandleOwnersAreCloseable(t *testing.T) {
	registry := NewFactory(&ModelSelectionConfig{
		Method: string(MethodStatic),
		ML:     DefaultMLSelectorConfig(),
	}).CreateAll()
	t.Cleanup(func() {
		if err := registry.Close(); err != nil {
			t.Errorf("Registry.Close returned %v, want nil", err)
		}
	})

	registry.mu.RLock()
	snapshot := make(map[SelectionMethod]Selector, len(registry.selectors))
	for method, selector := range registry.selectors {
		snapshot[method] = selector
	}
	registry.mu.RUnlock()

	methodsWithHandles := make(map[SelectionMethod]int)
	for method, selector := range snapshot {
		walk := &nativeHandleWalk{seen: map[uintptr]bool{}}
		walk.visit(reflect.ValueOf(selector), nil, 0)

		for _, chain := range walk.chains {
			methodsWithHandles[method]++
			handle := chain[len(chain)-1]
			for _, owner := range chain[:len(chain)-1] {
				if !owner.Implements(closerType) {
					t.Errorf("%s selector owns native handle %s through %s, which does not implement io.Closer:"+
						" Registry.Close will skip it and the handle leaks on every reload (chain: %v)",
						method, handle, owner, chain)
				}
			}
		}
	}

	// Without this the test would pass vacuously the moment the walk stops
	// finding handles — a refactor of MLSelectorAdapter, a binding moved to a
	// differently named module, or CreateAll no longer building ML selectors.
	if len(methodsWithHandles) == 0 {
		t.Fatal("walk found no native handles among the registered selectors;" +
			" the guard is no longer testing anything and needs updating")
	}
	for _, method := range []SelectionMethod{MethodKNN, MethodKMeans, MethodSVM, MethodMLP} {
		if methodsWithHandles[method] == 0 {
			t.Errorf("walk found no native handle behind the %s selector, but that selector is Rust-backed:"+
				" the walk can no longer see through to the handles it is meant to police", method)
		}
	}
}
