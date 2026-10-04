package main

import (
	"context"
	"errors"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// routerGrounding returns the grounding backends the router serves for the
// default recipe of a router config: its model catalog through the managed
// runtime, then that recipe's classifier. Panel and context grounding both read
// the hallucination detector.
func routerGrounding(path string) (*looper.GroundingBackends, io.Closer, error) {
	cfg, err := config.Load(path)
	if err != nil {
		return nil, nil, fmt.Errorf("load router config: %w", err)
	}
	owner := &groundingOwner{manager: modelservice.NewManager()}
	if owner.lease, err = owner.manager.Acquire(cfg); err != nil {
		return nil, nil, errors.Join(err, owner.Close())
	}
	owner.classifiers, err = classification.BuildRecipeClassifiers(cfg, nil, nil, nil,
		classification.RecipeRuntimeOptions{Runtime: serving.New(owner.lease, nil)})
	if err == nil {
		err = owner.classifiers.InitializeRuntime()
	}
	if err != nil {
		return nil, nil, errors.Join(err, owner.Close())
	}
	classifier := owner.classifiers.Default()
	if classifier == nil {
		return nil, nil, errors.Join(errors.New("router config has no default recipe classifier"), owner.Close())
	}
	return classifier.GroundingBackends(), owner, nil
}

// groundingOwner keeps the classifiers, their lease and the runtime processes
// alive for the whole evaluation.
type groundingOwner struct {
	manager     *modelservice.Manager
	lease       *modelservice.Lease
	classifiers *classification.RecipeClassifiers
}

func (o *groundingOwner) Close() error {
	var errs []error
	if o.classifiers != nil {
		errs = append(errs, o.classifiers.Close())
	}
	if o.lease != nil {
		errs = append(errs, o.lease.Close())
	}
	return errors.Join(append(errs, o.manager.Shutdown(context.Background()))...)
}

// needsShippedGrounding reports whether any requested arm grounds with the
// router's backends rather than none or the placebo.
func needsShippedGrounding(arms []string, reference string) bool {
	for _, arm := range arms {
		if cfg, placebo := armGrounding(arm, reference); cfg != nil && !placebo {
			return true
		}
	}
	return false
}
