package managementcomposition

import (
	"reflect"
	"testing"
)

func TestRoutingManifestCredentialProjectionIsStrictAndSecretFree(t *testing.T) {
	codec := &v03RoutingManifestCodec{}
	document := []byte(`version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - credential: 22222222-2222-4222-8222-222222222222
        - credential: 11111111-1111-4111-8111-111111111111
    - name: model-b
      backend_refs:
        - credential: 22222222-2222-4222-8222-222222222222
`)
	ids, err := codec.CredentialIDs(document)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{
		"11111111-1111-4111-8111-111111111111",
		"22222222-2222-4222-8222-222222222222",
	}
	if !reflect.DeepEqual(ids, want) {
		t.Fatalf("credential IDs = %#v, want %#v", ids, want)
	}
	for name, invalid := range map[string]string{
		"version": "version: v0.4\n",
		"secret": `version: v0.3
providers:
  models:
    - name: model-a
      backend_refs:
        - api_key: secret
`,
		"unknown field": "version: v0.3\nidentity: {}\n",
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := codec.CredentialIDs([]byte(invalid)); err == nil {
				t.Fatal("CredentialIDs accepted a non-routing or secret-bearing manifest")
			}
		})
	}
}
