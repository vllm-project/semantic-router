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
	"sort"

	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	// StorageClassDefaultAnnotation is the annotation used to mark a StorageClass as default
	StorageClassDefaultAnnotation = "storageclass.kubernetes.io/is-default-class"
)

// detectDefaultStorageClass finds the default StorageClass in the cluster.
// When more than one StorageClass is marked as default (for example while
// migrating from one default to another), it follows the same rule as the
// Kubernetes DefaultStorageClass admission plugin: the most recently created
// class wins, and ties are broken by name in ascending order.
func detectDefaultStorageClass(ctx context.Context, c client.Client) (string, error) {
	logger := log.FromContext(ctx)

	storageClasses := &storagev1.StorageClassList{}
	if err := c.List(ctx, storageClasses); err != nil {
		return "", fmt.Errorf("failed to list StorageClasses: %w", err)
	}

	var defaults []storagev1.StorageClass
	for _, sc := range storageClasses.Items {
		if sc.Annotations[StorageClassDefaultAnnotation] == "true" {
			defaults = append(defaults, sc)
		}
	}
	if len(defaults) == 0 {
		return "", fmt.Errorf("no default StorageClass found in the cluster")
	}

	sort.Slice(defaults, func(i, j int) bool {
		ti, tj := defaults[i].CreationTimestamp.UnixNano(), defaults[j].CreationTimestamp.UnixNano()
		if ti == tj {
			return defaults[i].Name < defaults[j].Name
		}
		return ti > tj
	})

	if len(defaults) > 1 {
		logger.Info("Multiple default StorageClasses found, using the most recently created", "count", len(defaults), "name", defaults[0].Name)
	} else {
		logger.Info("Found default StorageClass", "name", defaults[0].Name)
	}
	return defaults[0].Name, nil
}

// validateStorageClass validates that a StorageClass is available
// If storageClassName is empty, uses the default StorageClass
// If storageClassName is specified, validates it exists
// Returns error if no StorageClass is available
func validateStorageClass(ctx context.Context, c client.Client, storageClassName string) (string, error) {
	logger := log.FromContext(ctx)

	// If no storage class specified, use default
	if storageClassName == "" {
		defaultSC, err := detectDefaultStorageClass(ctx, c)
		if err != nil {
			return "", fmt.Errorf("no StorageClass specified and no default StorageClass found in cluster")
		}
		logger.Info("Using default StorageClass", "name", defaultSC)
		return defaultSC, nil
	}

	// Validate specified storage class exists
	sc := &storagev1.StorageClass{}
	err := c.Get(ctx, types.NamespacedName{Name: storageClassName}, sc)
	if err != nil {
		if errors.IsNotFound(err) {
			return "", fmt.Errorf("StorageClass %q not found", storageClassName)
		}
		return "", fmt.Errorf("failed to get StorageClass %q: %w", storageClassName, err)
	}

	logger.Info("Validated StorageClass", "name", storageClassName)
	return storageClassName, nil
}
