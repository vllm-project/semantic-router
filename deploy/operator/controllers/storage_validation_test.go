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
	"strings"
	"testing"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestValidateStorageClass(t *testing.T) {
	gp3 := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{
		Name:        "gp3",
		Annotations: map[string]string{StorageClassDefaultAnnotation: "true"},
	}}
	fast := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{Name: "fast"}}

	tests := []struct {
		name      string
		requested string
		want      string
		wantErr   string
	}{
		{name: "empty uses the cluster default", requested: "", want: "gp3"},
		{name: "explicit name wins over the default", requested: "fast", want: "fast"},
		{name: "explicit missing name is refused", requested: "missing", wantErr: `StorageClass "missing" not found`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := runtime.NewScheme()
			_ = scheme.AddToScheme(s)
			cl := fake.NewClientBuilder().WithScheme(s).WithObjects(gp3.DeepCopy(), fast.DeepCopy()).Build()

			got, err := validateStorageClass(context.Background(), cl, tt.requested)
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("validateStorageClass(%q) error = %v, want it to contain %q", tt.requested, err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("validateStorageClass(%q) failed: %v", tt.requested, err)
			}
			if got != tt.want {
				t.Errorf("validateStorageClass(%q) = %q, want %q", tt.requested, got, tt.want)
			}
		})
	}
}

func TestDetectDefaultStorageClassMultipleDefaults(t *testing.T) {
	older := metav1.NewTime(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))
	newer := metav1.NewTime(time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC))

	class := func(name string, created metav1.Time, isDefault bool) *storagev1.StorageClass {
		sc := &storagev1.StorageClass{ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			CreationTimestamp: created,
		}}
		if isDefault {
			sc.Annotations = map[string]string{StorageClassDefaultAnnotation: "true"}
		}
		return sc
	}

	tests := []struct {
		name    string
		classes []*storagev1.StorageClass
		want    string
		wantErr bool
	}{
		{
			name: "single default is used",
			classes: []*storagev1.StorageClass{
				class("gp3", older, true),
				class("fast", newer, false),
			},
			want: "gp3",
		},
		{
			// During a default-class migration both classes carry the annotation;
			// Kubernetes uses the most recently created one.
			name: "most recently created default wins",
			classes: []*storagev1.StorageClass{
				class("a-old-default", older, true),
				class("b-new-default", newer, true),
			},
			want: "b-new-default",
		},
		{
			name: "same creation time falls back to name order",
			classes: []*storagev1.StorageClass{
				class("zeta", newer, true),
				class("alpha", newer, true),
			},
			want: "alpha",
		},
		{
			name: "no default is an error",
			classes: []*storagev1.StorageClass{
				class("fast", newer, false),
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := runtime.NewScheme()
			_ = scheme.AddToScheme(s)
			builder := fake.NewClientBuilder().WithScheme(s)
			for _, sc := range tt.classes {
				builder = builder.WithObjects(sc)
			}
			cl := builder.Build()

			got, err := detectDefaultStorageClass(context.Background(), cl)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("detectDefaultStorageClass() = %q, want an error", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("detectDefaultStorageClass() failed: %v", err)
			}
			if got != tt.want {
				t.Errorf("detectDefaultStorageClass() = %q, want %q", got, tt.want)
			}
		})
	}
}
