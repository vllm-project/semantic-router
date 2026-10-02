//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func diagnosticRecipeName(recipe string) string {
	if recipe == "" {
		return string(config.DefaultRecipeName)
	}
	return recipe
}

// A borrowed recipe view keeps combined/batch operations on one classifier.
// No explicit recipe is interpreted by a legacy service that cannot select it.
func recipeDiagnosticService(service classificationService, recipe string) (classificationService, func(), error) {
	if recipe == "" {
		return service, func() {}, nil
	}
	source, ok := service.(interface {
		AcquireRecipeService(string) (*services.ClassificationService, func(), error)
	})
	if !ok {
		return nil, func() {}, services.ErrClassifierUnavailable
	}
	return source.AcquireRecipeService(recipe)
}
