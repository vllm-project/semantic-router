package routercontract

import "testing"

func TestReadFirstListenerEndpointReturnsTopLevelListener(t *testing.T) {
	configPath := writeRouterConfig(t, `
listeners:
  - address: 0.0.0.0
    port: 18889
`)

	endpoint, ok, err := ReadFirstListenerEndpoint(configPath)
	if err != nil {
		t.Fatalf("ReadFirstListenerEndpoint() error = %v", err)
	}
	if !ok {
		t.Fatal("ReadFirstListenerEndpoint() ok = false, want true")
	}
	if endpoint.Address != "0.0.0.0" || endpoint.Port != 18889 {
		t.Fatalf("endpoint = %+v, want 0.0.0.0:18889", endpoint)
	}
}

func TestReadFirstListenerEndpointFallsBackToAPIServerListener(t *testing.T) {
	configPath := writeRouterConfig(t, `
api_server:
  listeners:
    - address: "::"
      port: "18890"
`)

	endpoint, ok, err := ReadFirstListenerEndpoint(configPath)
	if err != nil {
		t.Fatalf("ReadFirstListenerEndpoint() error = %v", err)
	}
	if !ok {
		t.Fatal("ReadFirstListenerEndpoint() ok = false, want true")
	}
	if endpoint.Address != "::" || endpoint.Port != 18890 {
		t.Fatalf("endpoint = %+v, want :::18890", endpoint)
	}
}

func TestReadFirstListenerEndpointReturnsFalseWithoutValidListener(t *testing.T) {
	configPath := writeRouterConfig(t, `
listeners:
  - address: 127.0.0.1
    port: 0
`)

	_, ok, err := ReadFirstListenerEndpoint(configPath)
	if err != nil {
		t.Fatalf("ReadFirstListenerEndpoint() error = %v", err)
	}
	if ok {
		t.Fatal("ReadFirstListenerEndpoint() ok = true, want false")
	}
}

func TestReadFirstListenerEndpointReturnsParseError(t *testing.T) {
	configPath := writeRouterConfig(t, "listeners: [")

	_, _, err := ReadFirstListenerEndpoint(configPath)
	if err == nil {
		t.Fatal("ReadFirstListenerEndpoint() error = nil, want parse error")
	}
}
