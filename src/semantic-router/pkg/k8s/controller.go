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

package k8s

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Controller watches IntelligentPool and IntelligentRoute CRDs and updates configuration
// This is now a wrapper around the Reconciler for backward compatibility
type Controller struct {
	namespace  string
	reconciler *Reconciler
	stopCh     chan struct{}
}

// ControllerConfig holds configuration for the controller
type ControllerConfig struct {
	Namespace      string
	Kubeconfig     string
	StaticConfig   *config.RouterConfig
	OnConfigUpdate func(*config.RouterConfig) error
}

// NewController creates a new Kubernetes controller using controller-runtime
// This is now a wrapper around the new Reconciler implementation
func NewController(cfg ControllerConfig) (*Controller, error) {
	// Convert ControllerConfig to ReconcilerConfig
	reconcilerCfg := ReconcilerConfig(cfg)

	reconciler, err := NewReconciler(reconcilerCfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create reconciler: %w", err)
	}

	return &Controller{
		namespace:  cfg.Namespace,
		reconciler: reconciler,
		stopCh:     make(chan struct{}),
	}, nil
}

// Start starts the controller
func (c *Controller) Start(ctx context.Context) error {
	if err := c.reconciler.Start(ctx); err != nil {
		return err
	}

	<-c.stopCh
	logging.Infof("Kubernetes controller stopped")
	return nil
}

// Stop stops the controller
func (c *Controller) Stop() {
	c.reconciler.Stop()
	close(c.stopCh)
}
