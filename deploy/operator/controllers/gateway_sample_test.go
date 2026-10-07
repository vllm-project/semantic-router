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
	"os"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	gwapiv1 "sigs.k8s.io/gateway-api/apis/v1"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// Read the public sample rather than a second copy of its route. This checks
// the documented HTTPRoute -> Service -> Router listener contract together.
func TestGatewaySampleTargetsStandaloneInferenceListener(t *testing.T) {
	file, err := os.Open("../config/samples/vllm.ai_v1alpha1_semanticrouter_gateway.yaml")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if closeErr := file.Close(); closeErr != nil {
			t.Errorf("close gateway sample: %v", closeErr)
		}
	}()
	decoder := yamlutil.NewYAMLOrJSONDecoder(file, 4096)
	sr := &vllmv1alpha1.SemanticRouter{}
	if err = decoder.Decode(sr); err != nil {
		t.Fatal(err)
	}
	route := &gwapiv1.HTTPRoute{}
	if err = decoder.Decode(route); err != nil {
		t.Fatal(err)
	}
	if sr.Spec.Gateway != nil {
		t.Fatal("HTTP forwarding to the Router needs standalone mode")
	}
	scheme := runtime.NewScheme()
	r := &SemanticRouterReconciler{Scheme: scheme}
	mode, err := reconcileGatewayIntegration(context.Background(), fake.NewClientBuilder().WithScheme(scheme).Build(), sr)
	if err != nil || mode != GatewayModeStandalone {
		t.Fatalf("sample gateway mode = %q, error = %v", mode, err)
	}
	service := r.generateService(sr, mode)
	if route.Namespace != service.Namespace || len(route.Spec.Rules) != 1 {
		t.Fatalf("route must target its local Router Service: %+v", route.Spec)
	}
	rule := route.Spec.Rules[0]
	if len(rule.BackendRefs) != 1 || rule.BackendRefs[0].Name != gwapiv1.ObjectName(service.Name) {
		t.Fatalf("route backend must reference the generated Service: %+v", rule.BackendRefs)
	}
	if rule.BackendRefs[0].Port == nil || *rule.BackendRefs[0].Port != gwapiv1.PortNumber(DefaultListenerPort) {
		t.Fatal("inference route must use the Router listener port 8801, not management API port 8080")
	}
	if len(rule.Matches) != 1 || rule.Matches[0].Path == nil || *rule.Matches[0].Path.Value != "/v1" {
		t.Fatal("the inference route must not expose management endpoints")
	}
	var targetPort string
	for _, port := range service.Spec.Ports {
		if port.Port == int32(*rule.BackendRefs[0].Port) {
			targetPort = port.TargetPort.StrVal
		}
	}
	if targetPort == "" {
		t.Fatal("route's backend port does not exist on the generated Service")
	}
	containers := r.generateContainers(sr, mode)
	if len(containers) != 1 {
		t.Fatalf("a standalone Pod runs the Router alone, got %d containers", len(containers))
	}
	var containerPort int32
	for _, port := range containers[0].Ports {
		if port.Name == targetPort {
			containerPort = port.ContainerPort
		}
	}
	canonical, err := r.buildCanonicalConfig(context.Background(), sr)
	if err != nil {
		t.Fatal(err)
	}
	if len(canonical.Listeners) != 1 || canonical.Listeners[0].Port != int(containerPort) {
		t.Fatalf("Service target %q is not the Router's configured listener %+v", targetPort, canonical.Listeners)
	}
}

func TestServicePortDefaultsPreserveExplicitConfiguration(t *testing.T) {
	r := &SemanticRouterReconciler{}
	for _, test := range []struct {
		name    string
		service vllmv1alpha1.ServiceSpec
		want    [4]int32
	}{
		{name: "omitted", want: [4]int32{DefaultGRPCPort, DefaultGRPCPort, DefaultAPIPort, DefaultAPIPort}},
		{
			name: "service ports only",
			service: vllmv1alpha1.ServiceSpec{
				GRPC: vllmv1alpha1.PortSpec{Port: 50052},
				API:  vllmv1alpha1.PortSpec{Port: 8081},
			},
			want: [4]int32{50052, DefaultGRPCPort, 8081, DefaultAPIPort},
		},
		{
			name: "explicit target ports",
			service: vllmv1alpha1.ServiceSpec{
				GRPC: vllmv1alpha1.PortSpec{Port: 50052, TargetPort: 50053},
				API:  vllmv1alpha1.PortSpec{Port: 8081, TargetPort: 8082},
			},
			want: [4]int32{50052, 50053, 8081, 8082},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			sr := &vllmv1alpha1.SemanticRouter{Spec: vllmv1alpha1.SemanticRouterSpec{Service: test.service}}
			ports := r.generateService(sr, GatewayModeIntegration).Spec.Ports
			got := [4]int32{ports[0].Port, ports[0].TargetPort.IntVal, ports[1].Port, ports[1].TargetPort.IntVal}
			if got != test.want {
				t.Fatalf("service ports = %v, want %v", got, test.want)
			}
		})
	}
}
