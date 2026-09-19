package localclassifierbackend

import (
	"os"
	"path/filepath"
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
	canonical := read(filepath.Join("..", "..", "..", "config", "config.yaml"))
	global := canonical["global"].(map[string]any)
	catalog := global["model_catalog"].(map[string]any)
	guard := catalog["system"].(map[string]any)["prompt_guard"]
	profile := read("values.yaml")["config"].(map[string]any)
	routing := profile["routing"].(map[string]any)
	classifiers := routing["signals"].(map[string]any)["classifiers"].([]any)
	if len(classifiers) != 1 {
		t.Fatal("local classifier profile must exercise one isolated real classifier")
	}
	classifier := classifiers[0].(map[string]any)
	if classifier["model_path"] != guard || classifier["type"] != "local" || classifier["use_cpu"] != true {
		t.Fatalf("local classifier must use the canonical CPU Guard model: %#v", classifier)
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
