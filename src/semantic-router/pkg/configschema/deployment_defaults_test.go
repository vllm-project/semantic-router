package configschema

import (
	"encoding/json"
	"reflect"
	"testing"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDeploymentSchemaDefaultsMatchRuntime(t *testing.T) {
	var document struct {
		Definitions map[string]struct {
			Properties map[string]struct {
				Default json.RawMessage `json:"default"`
			} `json:"properties"`
		} `json:"$defs"`
	}
	if err := json.Unmarshal(Document(), &document); err != nil {
		t.Fatal(err)
	}
	var got map[string]routerconfig.ModelDeployment
	if err := json.Unmarshal(document.Definitions["CanonicalModelCatalog"].Properties["deployments"].Default, &got); err != nil {
		t.Fatal(err)
	}
	want := routerconfig.DefaultCanonicalGlobal().ModelCatalog.Deployments
	if len(got) == 0 || !reflect.DeepEqual(got, want) {
		t.Fatalf("offline deployment defaults differ from runtime: got=%+v want=%+v", got, want)
	}
}
