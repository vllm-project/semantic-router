package localclassifierbackend

import (
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestLocalClassifierUsesCanonicalGuardContract(t *testing.T) {
	read := func(path string) map[string]any {
		t.Helper()
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		var document map[string]any
		if err := yaml.Unmarshal(data, &document); err != nil {
			t.Fatal(err)
		}
		return document
	}
	// A generic local classifier runs a classify head, which Vela 2.0 does not
	// have, so the profile keeps Vela 1.0 Guard.
	profile := read("values.yaml")["config"].(map[string]any)
	routing := profile["routing"].(map[string]any)
	classifiers := routing["signals"].(map[string]any)["classifiers"].([]any)
	if len(classifiers) != 1 {
		t.Fatal("local classifier profile must exercise one isolated real classifier")
	}
	classifier := classifiers[0].(map[string]any)
	name, _ := classifier["deployment"].(string)
	deployments := profile["global"].(map[string]any)["model_catalog"].(map[string]any)["deployments"].(map[string]any)
	guard, _ := deployments[name].(map[string]any)
	if guard["provider"] != "model_runtime" || guard["artifact"] != "vllm-sr/Vela-1.0-Encoder-307M-Guard" || guard["device"] != "cpu" {
		t.Fatalf("local classifier must use the Vela 1.0 Guard model on CPU: %#v %#v", classifier, guard)
	}
	if !reflect.DeepEqual(classifier["labels"], []any{"benign", "jailbreak"}) {
		t.Fatal("local classifier labels must preserve the published Guard output order")
	}
	decision := routing["decisions"].([]any)[0].(map[string]any)
	condition := decision["rules"].(map[string]any)["conditions"].([]any)[0].(map[string]any)
	if condition["name"] != classifier["name"] || condition["label"] != "jailbreak" || condition["predicate"].(map[string]any)["gte"] != 0.5 {
		t.Fatal("the real classifier must drive the unchanged jailbreak decision at 0.5")
	}
}
