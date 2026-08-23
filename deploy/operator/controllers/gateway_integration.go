/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// resolveGatewayMode selects either the Operator-owned Envoy sidecar or an
// existing Gateway. HTTPRoute ownership stays with the Gateway deployment;
// the Operator exposes only the Router ExtProc Service in external mode.
func resolveGatewayMode(ctx context.Context, c client.Client, sr *vllmv1alpha1.SemanticRouter) (string, error) {
	logger := log.FromContext(ctx)

	if sr.Spec.Gateway == nil || sr.Spec.Gateway.ExistingRef == nil {
		logger.Info("No Gateway configured; using the inference-only Envoy sidecar")
		return gatewayModeSidecar, nil
	}

	gateway := &gatewayv1.Gateway{}
	err := c.Get(ctx, types.NamespacedName{
		Name:      sr.Spec.Gateway.ExistingRef.Name,
		Namespace: sr.Spec.Gateway.ExistingRef.Namespace,
	}, gateway)

	if err != nil {
		logger.Error(err, "Gateway not found", "name", sr.Spec.Gateway.ExistingRef.Name, "namespace", sr.Spec.Gateway.ExistingRef.Namespace)
		return "", fmt.Errorf("gateway %s/%s not found: %w", sr.Spec.Gateway.ExistingRef.Namespace, sr.Spec.Gateway.ExistingRef.Name, err)
	}

	logger.Info("Using existing Gateway", "gateway", gateway.Name)
	return gatewayModeExternal, nil
}
