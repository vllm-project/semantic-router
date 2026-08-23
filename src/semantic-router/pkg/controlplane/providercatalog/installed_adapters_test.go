package providercatalog

import (
	"bytes"
	"testing"
)

func TestRegistryCapabilitiesAreClassSeparatedAndDeterministic(t *testing.T) {
	firstOptions := testRegistryOptions(validDefinition("provider", 1))
	first, err := NewRegistry(firstOptions)
	if err != nil {
		t.Fatal(err)
	}
	secondOptions := firstOptions
	secondOptions.WireFormats = []string{"openai.responses.v1", "openai.chat.v1", "anthropic.messages.v1"}
	secondOptions.CredentialAdapterIDs = []string{"x-api-key", "bearer"}
	second, err := NewRegistry(secondOptions)
	if err != nil {
		t.Fatal(err)
	}
	for _, plane := range []CapabilityPlane{CapabilityPlaneControl, CapabilityPlaneData} {
		firstDigest, firstErr := first.CapabilityDigest(plane)
		secondDigest, secondErr := second.CapabilityDigest(plane)
		if firstErr != nil || secondErr != nil || !bytes.Equal(firstDigest, secondDigest) {
			t.Fatalf("%s capability digest depends on registration order", plane)
		}
	}
	if !first.HasWireFormat("openai.chat.v1") || first.HasCredentialAdapter("openai.chat.v1") ||
		!first.HasCredentialAdapter("bearer") || !first.HasDiscoveryAdapter("openai.models.v1") {
		t.Fatal("installed adapter classes are not isolated")
	}
	digest, err := first.CapabilityDigest(CapabilityPlaneControl)
	if err != nil {
		t.Fatal(err)
	}
	digest[0] ^= 0xff
	stable, err := first.CapabilityDigest(CapabilityPlaneControl)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(digest, stable) {
		t.Fatal("CapabilityDigest returned mutable internal storage")
	}
	if _, err := first.CapabilityDigest("unknown"); err == nil {
		t.Fatal("CapabilityDigest accepted an unknown plane")
	}
}

func TestRegistryPlaneDigestsAreIndependent(t *testing.T) {
	baselineOptions := testRegistryOptions(validDefinition("provider", 1))
	baselineOptions.WireFormats = []string{"openai.chat.v1"}
	baseline, err := NewRegistry(baselineOptions)
	if err != nil {
		t.Fatal(err)
	}
	dataOptions := baselineOptions
	dataOptions.WireFormats = []string{"openai.chat.v1", "anthropic.messages.v1"}
	dataChanged, err := NewRegistry(dataOptions)
	if err != nil {
		t.Fatal(err)
	}
	controlBaseline, _ := baseline.CapabilityDigest(CapabilityPlaneControl)
	controlChanged, _ := dataChanged.CapabilityDigest(CapabilityPlaneControl)
	dataBaseline, _ := baseline.CapabilityDigest(CapabilityPlaneData)
	dataChangedDigest, _ := dataChanged.CapabilityDigest(CapabilityPlaneData)
	if !bytes.Equal(controlBaseline, controlChanged) || bytes.Equal(dataBaseline, dataChangedDigest) {
		t.Fatal("data-plane adapter change was not isolated to the data-plane digest")
	}
}

func TestRegistryRejectsInvalidOrAmbiguousCapabilitySets(t *testing.T) {
	var typedNil *nilBackendCompiler
	for _, test := range []struct {
		name       string
		compilers  []BackendCompiler
		protocol   []string
		credential []string
		discovery  []string
	}{
		{name: "empty compiler", protocol: []string{"openai.chat.v1"}},
		{name: "empty protocol", compilers: []BackendCompiler{StaticBackendCompiler{}}},
		{name: "duplicate", compilers: []BackendCompiler{StaticBackendCompiler{}}, protocol: []string{"openai.chat.v1", "openai.chat.v1"}},
		{name: "invalid credential", compilers: []BackendCompiler{StaticBackendCompiler{}}, protocol: []string{"openai.chat.v1"}, credential: []string{"Bearer"}},
		{name: "invalid discovery", compilers: []BackendCompiler{StaticBackendCompiler{}}, protocol: []string{"openai.chat.v1"}, discovery: []string{" bad"}},
		{name: "typed nil compiler", compilers: []BackendCompiler{typedNil}, protocol: []string{"openai.chat.v1"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			if _, err := NewRegistry(RegistryOptions{
				Integrations:     []Integration{IntegrationFunc(func() Definition { return validDefinition("provider", 1) })},
				BackendCompilers: test.compilers, WireFormats: test.protocol,
				CredentialAdapterIDs: test.credential, DiscoveryAdapterIDs: test.discovery,
			}); err == nil {
				t.Fatal("NewRegistry() unexpectedly succeeded")
			}
		})
	}
}

type nilBackendCompiler struct{ StaticBackendCompiler }
