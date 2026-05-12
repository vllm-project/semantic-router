package helm

import "testing"

func TestInstallOptionsCloneCopiesMutableFields(t *testing.T) {
	original := InstallOptions{
		ReleaseName: "release",
		Chart:       "chart",
		Namespace:   "namespace",
		ValuesFiles: []string{"values.yaml"},
		Set: map[string]string{
			"key": "value",
		},
		Wait:    true,
		Timeout: "10m",
	}

	cloned := original.Clone()
	cloned.ValuesFiles[0] = "mutated.yaml"
	cloned.Set["key"] = "mutated"

	if original.ValuesFiles[0] != "values.yaml" {
		t.Fatalf("expected values files to remain unchanged, got %#v", original.ValuesFiles)
	}
	if original.Set["key"] != "value" {
		t.Fatalf("expected set values to remain unchanged, got %#v", original.Set)
	}
}

func TestCatalogedGatewayInstallOptions(t *testing.T) {
	tests := []struct {
		name      string
		options   InstallOptions
		release   string
		namespace string
		version   string
		timeout   string
	}{
		{
			name:      "semantic router",
			options:   SemanticRouterRelease,
			release:   "semantic-router",
			namespace: "vllm-semantic-router-system",
			version:   "",
			timeout:   "30m",
		},
		{
			name:      "envoy gateway",
			options:   EnvoyGatewayRelease,
			release:   "eg",
			namespace: "envoy-gateway-system",
			version:   "v1.6.0",
			timeout:   "10m",
		},
		{
			name:      "ai gateway crd",
			options:   AIGatewayCRDRelease,
			release:   "aieg-crd",
			namespace: "envoy-ai-gateway-system",
			version:   "v0.4.0",
			timeout:   "10m",
		},
		{
			name:      "ai gateway",
			options:   AIGatewayRelease,
			release:   "aieg",
			namespace: "envoy-ai-gateway-system",
			version:   "v0.4.0",
			timeout:   "10m",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.options.ReleaseName != tt.release {
				t.Fatalf("expected release %q, got %q", tt.release, tt.options.ReleaseName)
			}
			if tt.options.Namespace != tt.namespace {
				t.Fatalf("expected namespace %q, got %q", tt.namespace, tt.options.Namespace)
			}
			if tt.options.Version != tt.version {
				t.Fatalf("expected version %q, got %q", tt.version, tt.options.Version)
			}
			if tt.options.Timeout != tt.timeout {
				t.Fatalf("expected timeout %q, got %q", tt.timeout, tt.options.Timeout)
			}
			if !tt.options.Wait {
				t.Fatal("expected cataloged release to wait for readiness")
			}
		})
	}
}
