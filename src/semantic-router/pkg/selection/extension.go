package selection

import (
	"context"
	"errors"
)

// ExtensionAlgorithm is how a decision algorithm type registered outside the
// Router chooses among a decision's candidates: the decoded payload of its
// block implements it.
type ExtensionAlgorithm interface {
	Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error)
}

// NewExtensionSelector runs algorithm as the selector of method. Such a
// selector is experimental, learns nothing from feedback and declares no
// dependencies.
func NewExtensionSelector(method SelectionMethod, algorithm ExtensionAlgorithm) Selector {
	return &extensionSelector{method: method, algorithm: algorithm}
}

type extensionSelector struct {
	method    SelectionMethod
	algorithm ExtensionAlgorithm
}

func (s *extensionSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}
	result, err := s.algorithm.Select(ctx, selCtx)
	if err != nil {
		return nil, err
	}
	if result == nil || result.SelectedModel == "" {
		return nil, errors.New("selection: the algorithm selected no candidate")
	}
	result.Method, result.Tier = s.method, TierExperimental
	return result, nil
}

func (s *extensionSelector) Method() SelectionMethod { return s.method }

func (s *extensionSelector) UpdateFeedback(context.Context, *Feedback) error { return nil }

func (s *extensionSelector) Tier() AlgorithmTier { return TierExperimental }

func (s *extensionSelector) ExternalDependencies() []Dependency { return nil }
