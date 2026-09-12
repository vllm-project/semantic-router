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
	"sigs.k8s.io/yaml"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Read the public sample rather than a second copy of its route. This checks
// the documented HTTPRoute -> Service -> Envoy listener contract together.
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
		t.Fatal("HTTP forwarding needs the standalone Envoy sidecar")
	}
	scheme := runtime.NewScheme()
	r := &SemanticRouterReconciler{Scheme: scheme}
	mode, err := reconcileGatewayIntegration(context.Background(), fake.NewClientBuilder().WithScheme(scheme).Build(), sr)
	if err != nil || mode != "standalone" {
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
	if rule.BackendRefs[0].Port == nil || *rule.BackendRefs[0].Port != 8801 {
		t.Fatal("inference route must use Envoy port 8801, not management API port 8080")
	}
	if len(rule.Matches) != 1 || rule.Matches[0].Path == nil || *rule.Matches[0].Path.Value != "/v1" {
		t.Fatal("the inference route must not expose management endpoints")
	}
	var targetPort int32
	for _, port := range service.Spec.Ports {
		if port.Port < 1 || port.Port > 65535 || port.TargetPort.IntVal < 1 || port.TargetPort.IntVal > 65535 {
			t.Fatalf("the public sample must generate a valid Service: %+v", port)
		}
		if port.Port == int32(*rule.BackendRefs[0].Port) {
			if port.Name != "envoy-http" {
				t.Fatalf("route resolves to %s instead of Envoy", port.Name)
			}
			targetPort = port.TargetPort.IntVal
		}
	}
	if targetPort == 0 {
		t.Fatal("route's backend port does not exist on the generated Service")
	}
	foundContainer := false
	for _, container := range r.generateContainers(sr, mode) {
		if container.Name == "envoy-proxy" {
			for _, port := range container.Ports {
				foundContainer = foundContainer || port.ContainerPort == targetPort
			}
		}
	}
	if !foundContainer {
		t.Fatal("Service target has no Envoy container port")
	}
	var bootstrap struct {
		StaticResources struct {
			Listeners []struct {
				Address struct {
					SocketAddress struct {
						PortValue uint32 `json:"port_value"`
					} `json:"socket_address"`
				} `json:"address"`
			} `json:"listeners"`
		} `json:"static_resources"`
	}
	canonical, err := r.buildCanonicalConfig(context.Background(), sr)
	if err != nil {
		t.Fatal(err)
	}
	if !canonical.Global.Router.ClearRouteCache {
		t.Fatal("the public sample must reselect Envoy routes after ExtProc selects a model")
	}
	rendered, err := generateStandaloneEnvoyConfig(canonical)
	if err != nil {
		t.Fatal(err)
	}
	if err := yaml.Unmarshal([]byte(rendered), &bootstrap); err != nil {
		t.Fatal(err)
	}
	if len(bootstrap.StaticResources.Listeners) != 1 || int64(bootstrap.StaticResources.Listeners[0].Address.SocketAddress.PortValue) != int64(targetPort) {
		t.Fatal("Service target does not match the rendered Envoy listener")
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
			ports := r.generateService(sr, "standalone").Spec.Ports
			got := [4]int32{ports[0].Port, ports[0].TargetPort.IntVal, ports[1].Port, ports[1].TargetPort.IntVal}
			if got != test.want {
				t.Fatalf("service ports = %v, want %v", got, test.want)
			}
		})
	}
}

func TestStandaloneEnvoyRoutesDiscoveredModelsAndPreservesTLS(t *testing.T) {
	canonical := &routerconfig.CanonicalConfig{
		Providers: routerconfig.CanonicalProviders{
			Defaults: routerconfig.CanonicalProviderDefaults{DefaultModel: "model_a"},
			Models: []routerconfig.CanonicalProviderModel{
				{Name: "model-a", BackendRefs: []routerconfig.CanonicalBackendRef{
					{Endpoint: "primary.models.svc.cluster.local:8000", Protocol: "http", Weight: 75},
					{Endpoint: "replica.models.svc.cluster.local:8001", Protocol: "http", Weight: 25},
				}},
				{Name: "model_a", BackendRefs: []routerconfig.CanonicalBackendRef{
					{Endpoint: "secure.models.svc.cluster.local:8443", Protocol: "https"},
				}},
			},
		},
	}
	rendered, err := generateStandaloneEnvoyConfig(canonical)
	if err != nil {
		t.Fatal(err)
	}
	var config map[string]interface{}
	if err := yaml.Unmarshal([]byte(rendered), &config); err != nil {
		t.Fatal(err)
	}
	resources := config["static_resources"].(map[string]interface{})
	clusters := resources["clusters"].([]interface{})
	byName := map[string]map[string]interface{}{}
	for _, value := range clusters {
		cluster := value.(map[string]interface{})
		name := cluster["name"].(string)
		if byName[name] != nil {
			t.Fatalf("duplicate cluster %s", name)
		}
		byName[name] = cluster
	}
	if byName["dynamic_forward_proxy_cluster"] != nil {
		t.Fatal("inference must not use the client Host as its backend")
	}
	listener := resources["listeners"].([]interface{})[0].(map[string]interface{})
	chain := listener["filter_chains"].([]interface{})[0].(map[string]interface{})
	filter := chain["filters"].([]interface{})[0].(map[string]interface{})
	hcm := filter["typed_config"].(map[string]interface{})
	extproc := hcm["http_filters"].([]interface{})[0].(map[string]interface{})["typed_config"].(map[string]interface{})
	if extproc["failure_mode_allow"] != false {
		t.Fatal("routing must fail closed when ExtProc fails")
	}
	routeConfig := hcm["route_config"].(map[string]interface{})
	virtualHost := routeConfig["virtual_hosts"].([]interface{})[0].(map[string]interface{})
	routes := virtualHost["routes"].([]interface{})
	for index, model := range canonical.Providers.Models {
		route := routes[index+1].(map[string]interface{})
		match := route["match"].(map[string]interface{})
		header := match["headers"].([]interface{})[0].(map[string]interface{})
		if header["name"] != "x-selected-model" || header["string_match"].(map[string]interface{})["exact"] != model.Name {
			t.Fatalf("model route does not match Router selection: %+v", match)
		}
		action := route["route"].(map[string]interface{})
		weighted := action["weighted_clusters"].(map[string]interface{})["clusters"].([]interface{})
		if len(weighted) != len(model.BackendRefs) {
			t.Fatal("missing discovered backend")
		}
		for backendIndex, ref := range model.BackendRefs {
			weight := weighted[backendIndex].(map[string]interface{})
			expectedWeight := ref.Weight
			if expectedWeight == 0 {
				expectedWeight = 1
			}
			if weight["weight"] != float64(expectedWeight) {
				t.Fatalf("backend weight changed: %+v", weight)
			}
			cluster := byName[weight["name"].(string)]
			assignment := cluster["load_assignment"].(map[string]interface{})
			endpoint := assignment["endpoints"].([]interface{})[0].(map[string]interface{})["lb_endpoints"].([]interface{})[0].(map[string]interface{})["endpoint"].(map[string]interface{})
			if endpoint["hostname"] != ref.Endpoint {
				t.Fatalf("wrong discovered destination: %+v", endpoint)
			}
			if ref.Protocol == "https" {
				socket := cluster["transport_socket"].(map[string]interface{})
				tls := socket["typed_config"].(map[string]interface{})
				if tls["sni"] != "secure.models.svc.cluster.local" {
					t.Fatal("TLS identity was lost")
				}
			} else if cluster["transport_socket"] != nil {
				t.Fatal("HTTP backend unexpectedly uses TLS")
			}
		}
	}
	defaultRoute := routes[len(routes)-1].(map[string]interface{})["route"].(map[string]interface{})
	defaultCluster := defaultRoute["weighted_clusters"].(map[string]interface{})["clusters"].([]interface{})[0].(map[string]interface{})["name"]
	modelRoute := routes[2].(map[string]interface{})["route"].(map[string]interface{})
	modelCluster := modelRoute["weighted_clusters"].(map[string]interface{})["clusters"].([]interface{})[0].(map[string]interface{})["name"]
	if defaultCluster != modelCluster {
		t.Fatal("default route ignores configured default model")
	}
}
