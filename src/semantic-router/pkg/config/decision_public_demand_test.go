package config

import "testing"

func TestDecisionPublicIdentityNeverUsesRuntimePrivateNames(t *testing.T) {
	for _, artifact := range []string{"/srv/private/model", "models/private-model", "./org/model", "../org/model", ""} {
		resource := ModelDeployment{Provider: ModelRuntimeProvider, Artifact: artifact, ServedName: "internal-runtime-name"}
		if got := resource.PublicModelName(); got != "" {
			t.Fatalf("local artifact leaked: %s", got)
		}
		resource.PublicName = "public-decision"
		if resource.PublicModelName() != "public-decision" {
			t.Fatal("explicit public identity unavailable")
		}
	}
	if (ModelDeployment{Artifact: "org/decision"}).PublicModelName() != "org/decision" {
		t.Fatal("canonical artifact identity lost")
	}
}

func TestSystemOneListenerIsExactModelDemand(t *testing.T) {
	cfg := DefaultGlobalConfig()
	cfg.ModelDeployments["published"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "org/decision", PublicName: "public-model"}
	cfg.ModelDeployments["unused"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "org/other"}
	cfg.Listeners = []Listener{{Name: "public", SystemOne: &ListenerSystemOne{Models: []string{"public-model"}}}}
	used := ModelRuntimeDeploymentsInUse(&cfg)
	if len(used) != 1 || used["published"].Artifact != "org/decision" {
		t.Fatalf("explicit API grant must be the sole demand: %+v", used)
	}
	cfg.Listeners = nil
	if len(ModelRuntimeDeploymentsInUse(&cfg)) != 0 {
		t.Fatal("unpublished declarations became demand")
	}
}
