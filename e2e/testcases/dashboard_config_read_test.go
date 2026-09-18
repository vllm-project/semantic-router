package testcases

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"sigs.k8s.io/yaml"
)

func TestDashboardConfigReadRequiresCanonicalDeployedYAML(t *testing.T) {
	values, err := os.ReadFile("../profiles/dashboard/values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	var profile map[string]interface{}
	if decodeErr := yaml.Unmarshal(values, &profile); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	canonical, err := yaml.Marshal(profile["configOverride"])
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name    string
		body    string
		wantErr bool
	}{
		{name: "active profile", body: string(canonical)},
		{name: "legacy file is not a usable deployment", body: "model: base-model\nmodel_config: {}\ndecisions: []\n", wantErr: true},
		{name: "version alone is insufficient", body: "version: v0.3\n", wantErr: true},
		{name: "malformed YAML", body: "[", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Content-Type", "text/yaml")
				_, _ = w.Write([]byte(test.body))
			}))
			defer server.Close()
			_, err := fetchDashboardYAMLConfig(context.Background(), server.Client(), server.URL, "test-token", false)
			if (err != nil) != test.wantErr {
				t.Fatalf("fetchDashboardYAMLConfig() error = %v, wantErr %t", err, test.wantErr)
			}
		})
	}
}
