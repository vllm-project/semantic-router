package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestRemoteOperationIdentityUsesActualSelectorAndTarget(t *testing.T) {
	base := config.ExternalModelConfig{ModelName: "model-one", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "Example.COM", Port: 80, Protocol: "http"}}
	for _, test := range []struct{ name, adapter, endpoint, model string }{
		{"classify", config.RemoteClassifierProtocolHTTPClassify, "http://example.com:80/classify", ""},
		{"chat", config.RemoteClassifierProtocolHTTPChat, "http://example.com:80/v1/chat/completions", "model-one"},
	} {
		t.Run(test.name, func(t *testing.T) {
			endpoint, model, err := remoteOperationIdentity(test.adapter, &base)
			if err != nil {
				t.Fatal(err)
			}
			if endpoint != test.endpoint || model != test.model {
				t.Fatalf("identity = (%s, %s), want (%s, %s)", endpoint, model, test.endpoint, test.model)
			}
			alias := base
			alias.Name, alias.ModelName = "catalog-alias", "model-two"
			aliasEndpoint, aliasModel, aliasErr := remoteOperationIdentity(test.adapter, &alias)
			if aliasErr != nil {
				t.Fatal(aliasErr)
			}
			if aliasEndpoint != endpoint {
				t.Fatalf("alias changed operation: %s", aliasEndpoint)
			}
			if test.adapter == config.RemoteClassifierProtocolHTTPClassify && aliasModel != model {
				t.Fatal("unused model name changed classify resource")
			}
			if test.adapter == config.RemoteClassifierProtocolHTTPChat && aliasModel == model {
				t.Fatal("actual chat selector missing from resource")
			}
		})
	}
	// The explicit grounding endpoint and a named chat backend address the same
	// operation, even when the former omits the default port and includes /v1.
	explicit := base
	explicit.ModelEndpoint = config.ClassifierVLLMEndpoint{Address: "http://example.com/v1/"}
	namedEndpoint, namedModel, err := remoteOperationIdentity(config.RemoteClassifierProtocolHTTPChat, &base)
	if err != nil {
		t.Fatal(err)
	}
	explicitEndpoint, explicitModel, explicitErr := remoteOperationIdentity(config.RemoteClassifierProtocolHTTPChat, &explicit)
	if explicitErr != nil {
		t.Fatal(explicitErr)
	}
	if namedEndpoint != explicitEndpoint || namedModel != explicitModel {
		t.Fatalf("equivalent request targets differ: %s vs %s", namedEndpoint, explicitEndpoint)
	}
}

func TestRemoteOperationIdentityRejectsUnsupportedAdapter(t *testing.T) {
	external := &config.ExternalModelConfig{ModelEndpoint: config.ClassifierVLLMEndpoint{Address: "example.com", Port: 80}}
	if _, _, err := remoteOperationIdentity("unknown", external); err == nil {
		t.Fatal("unsupported adapter silently accepted")
	}
}
