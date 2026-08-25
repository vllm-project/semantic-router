package config

import "testing"

func TestBackendDispatchDefaultsAreValid(t *testing.T) {
	config := DefaultBackendDispatchConfig()
	if err := validateBackendDispatch(config); err != nil {
		t.Fatal(err)
	}
	if config.Port != 8180 || config.MaxRequestBodyBytes != 64<<20 {
		t.Fatalf("defaults = %+v", config)
	}
}

func TestBootstrapRejectsMissingBackendEgressEvenWhenDispatchIsOmitted(t *testing.T) {
	config := RouterConfig{}
	if err := config.ValidateRuntimeBootstrap(); err == nil {
		t.Fatal("bootstrap accepted a manifest without backend egress policy")
	}
}

func TestBackendDispatchRejectsPublicContractViolations(t *testing.T) {
	for _, mutate := range []func(*BackendDispatchConfig){
		func(config *BackendDispatchConfig) { config.BindAddress = "router.internal" },
		func(config *BackendDispatchConfig) { config.Port = 0 },
		func(config *BackendDispatchConfig) { config.Audience = "Backend Dispatch" },
		func(config *BackendDispatchConfig) { config.CapabilityTTL = "61s" },
		func(config *BackendDispatchConfig) { config.MaxRequestBodyBytes = 512 },
	} {
		config := DefaultBackendDispatchConfig()
		mutate(&config)
		if err := validateBackendDispatch(config); err == nil {
			t.Fatalf("invalid config accepted: %+v", config)
		}
	}
}
