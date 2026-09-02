package extproc

import (
	"errors"
	"slices"
	"testing"
)

func TestResourceScopeClosesInReverseOrderOnceAndJoinsErrors(t *testing.T) {
	errA := errors.New("a")
	errB := errors.New("b")
	var order []int
	scope := newResourceScope()
	scope.add(func() error {
		order = append(order, 1)
		return errA
	})
	scope.add(func() error {
		order = append(order, 2)
		return errB
	})

	err := scope.close()
	if !errors.Is(err, errA) || !errors.Is(err, errB) {
		t.Fatalf("close() error = %v, want both errors", err)
	}
	if !slices.Equal(order, []int{2, 1}) {
		t.Fatalf("close order = %v, want [2 1]", order)
	}
	if err := scope.close(); err != nil {
		t.Fatalf("second close() error = %v", err)
	}
	if !slices.Equal(order, []int{2, 1}) {
		t.Fatalf("second close changed order to %v", order)
	}
}
