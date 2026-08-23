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

package v1alpha1

import (
	"encoding/json"
	"testing"
)

func TestSemanticRouterBootstrapRoundTrip(t *testing.T) {
	original := SemanticRouter{Spec: SemanticRouterSpec{Bootstrap: BootstrapSpec{
		ConfigMapRef: BootstrapConfigMapReference{Name: "router-bootstrap-v3", Key: "production.yaml"},
	}}}
	payload, err := json.Marshal(original)
	if err != nil {
		t.Fatal(err)
	}
	var decoded SemanticRouter
	if err := json.Unmarshal(payload, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Spec.Bootstrap.ConfigMapRef != original.Spec.Bootstrap.ConfigMapRef {
		t.Fatalf("bootstrap ref = %#v, want %#v", decoded.Spec.Bootstrap.ConfigMapRef, original.Spec.Bootstrap.ConfigMapRef)
	}
}
