package onnx_binding

import "fmt"

// dimensionStateFromNative checks availability and dimension invariants without
// making model-specific decisions.
func dimensionStateFromNative(state uint32, native int, dimensions []int) (DimensionState, error) {
	switch state {
	case 0:
		if native != 0 || len(dimensions) != 0 {
			return "", fmt.Errorf("%w: unloaded model has dimension data", ErrMalformedCapabilities)
		}
		return DimensionStateNotLoaded, nil
	case 1:
		if err := validateObservedDimensions(native, dimensions); err != nil {
			return "", err
		}
		return DimensionStateAvailable, nil
	default:
		return "", fmt.Errorf("%w: dimension state %d", ErrMalformedCapabilities, state)
	}
}

func validateObservedDimensions(native int, dimensions []int) error {
	if native <= 0 || len(dimensions) == 0 {
		return fmt.Errorf("%w: missing observed dimensions", ErrMalformedCapabilities)
	}
	seen := make(map[int]bool, len(dimensions))
	for _, dimension := range dimensions {
		if dimension <= 0 || seen[dimension] {
			return fmt.Errorf("%w: invalid or repeated dimension %d", ErrMalformedCapabilities, dimension)
		}
		seen[dimension] = true
	}
	if !seen[native] {
		return fmt.Errorf("%w: native dimension %d is not declared", ErrMalformedCapabilities, native)
	}
	return nil
}
