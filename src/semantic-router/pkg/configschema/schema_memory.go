package configschema

import (
	"encoding/json"
	"strconv"

	"github.com/invopop/jsonschema"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func setMemoryPersistenceBounds(root *jsonschema.Schema) {
	for _, property := range []string{"timeout_seconds", "concurrency", "queue", "shutdown_grace_seconds"} {
		setDefinitionPropertyMinimum(root, "MemoryPersistenceConfig", property, 0)
	}
	for property, maximum := range map[string]int{
		"concurrency": routerconfig.MaxMemoryPersistenceConcurrency,
		"queue":       routerconfig.MaxMemoryPersistenceQueue,
	} {
		if field := definitionProperty(root, "MemoryPersistenceConfig", property); field != nil {
			field.Maximum = json.Number(strconv.Itoa(maximum))
		}
	}
}
