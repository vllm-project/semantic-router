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

	routev1 "github.com/openshift/api/route/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// reconcileRoute creates or deletes OpenShift Route based on config
func reconcileRoute(ctx context.Context, c client.Client, scheme *runtime.Scheme, sr *vllmv1alpha1.SemanticRouter, isOpenShift bool) error {
	logger := log.FromContext(ctx)

	// Check if Route creation enabled
	if sr.Spec.OpenShift == nil || sr.Spec.OpenShift.Routes == nil || !sr.Spec.OpenShift.Routes.Enabled {
		// Delete Route if exists
		return deleteRouteIfExists(ctx, c, sr)
	}

	// Only create on OpenShift
	if !isOpenShift {
		logger.Info("Route creation requested but not on OpenShift platform")
		return nil
	}

	// Create or update Route
	return createOrUpdateRoute(ctx, c, scheme, sr)
}

// createOrUpdateRoute builds and applies Route
func createOrUpdateRoute(ctx context.Context, c client.Client, scheme *runtime.Scheme, sr *vllmv1alpha1.SemanticRouter) error {
	logger := log.FromContext(ctx)

	termination := routev1.TLSTerminationEdge
	if sr.Spec.OpenShift.Routes.TLS != nil && sr.Spec.OpenShift.Routes.TLS.Termination != "" {
		termination = routev1.TLSTerminationType(sr.Spec.OpenShift.Routes.TLS.Termination)
	}

	insecurePolicy := routev1.InsecureEdgeTerminationPolicyRedirect
	if sr.Spec.OpenShift.Routes.TLS != nil && sr.Spec.OpenShift.Routes.TLS.InsecureEdgeTerminationPolicy != "" {
		// Map from CRD enum values to OpenShift Route enum values
		switch sr.Spec.OpenShift.Routes.TLS.InsecureEdgeTerminationPolicy {
		case "Allow":
			insecurePolicy = routev1.InsecureEdgeTerminationPolicyAllow
		case "Redirect":
			insecurePolicy = routev1.InsecureEdgeTerminationPolicyRedirect
		case "None":
			insecurePolicy = routev1.InsecureEdgeTerminationPolicyNone
		}
	}

	route := &routev1.Route{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "route.openshift.io/v1",
			Kind:       "Route",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
		},
		Spec: routev1.RouteSpec{
			Host: sr.Spec.OpenShift.Routes.Hostname, // Optional - OpenShift generates if empty
			To: routev1.RouteTargetReference{
				Kind: "Service",
				Name: sr.Name,
			},
			Port: &routev1.RoutePort{
				TargetPort: intstr.FromString("api"),
			},
			TLS: &routev1.TLSConfig{
				Termination:                   termination,
				InsecureEdgeTerminationPolicy: insecurePolicy,
			},
		},
	}

	// Set owner reference
	if err := ctrl.SetControllerReference(sr, route, scheme); err != nil {
		return err
	}

	// Create or update
	err := c.Patch(ctx, route, client.Apply, client.ForceOwnership, client.FieldOwner("semantic-router-operator"))
	if err != nil {
		logger.Error(err, "Failed to create/update Route")
		return err
	}

	logger.Info("Successfully created/updated Route", "name", route.Name)

	// Update status with Route hostname
	// Note: We need to refetch the Route to get the status with the assigned hostname
	createdRoute := &routev1.Route{}
	if err := c.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, createdRoute); err == nil {
		if len(createdRoute.Status.Ingress) > 0 {
			sr.Status.OpenShiftFeatures = &vllmv1alpha1.OpenShiftFeaturesStatus{
				RoutesEnabled: true,
				RouteHostname: createdRoute.Status.Ingress[0].Host,
			}
		}
	}

	return nil
}

// deleteRouteIfExists removes Route if it exists
func deleteRouteIfExists(ctx context.Context, c client.Client, sr *vllmv1alpha1.SemanticRouter) error {
	logger := log.FromContext(ctx)

	route := &routev1.Route{}
	err := c.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, route)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil // Already deleted
		}
		return err
	}

	logger.Info("Deleting Route", "name", route.Name)
	return c.Delete(ctx, route)
}
