package configschema

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/invopop/jsonschema"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// setMemoryPersistenceBounds mirrors the bounds Parse enforces. A missing
// definition or property is a generator error: offline validators would
// otherwise accept documents the Router rejects at startup.
func setMemoryPersistenceBounds(root *jsonschema.Schema) error {
	maximums := map[string]int64{
		"timeout_seconds":        routerconfig.MaxMemoryPersistenceDurationSeconds,
		"concurrency":            int64(routerconfig.MaxMemoryPersistenceConcurrency),
		"queue":                  int64(routerconfig.MaxMemoryPersistenceQueue),
		"shutdown_grace_seconds": routerconfig.MaxMemoryPersistenceDurationSeconds,
	}
	for _, property := range []string{"timeout_seconds", "concurrency", "queue", "shutdown_grace_seconds"} {
		field := definitionProperty(root, "MemoryPersistenceConfig", property)
		if field == nil {
			return fmt.Errorf("memory persistence schema is missing %q", property)
		}
		field.Minimum = json.Number("0")
		if maximum, ok := maximums[property]; ok {
			field.Maximum = json.Number(strconv.FormatInt(maximum, 10))
		}
	}
	return nil
}

func setMemoryConsolidationBounds(root *jsonschema.Schema) error {
	maximums := map[string]int64{
		"cooldown_seconds": routerconfig.MaxMemoryPersistenceDurationSeconds,
		"timeout_seconds":  routerconfig.MaxMemoryPersistenceDurationSeconds,
		"concurrency":      int64(routerconfig.MaxMemoryPersistenceConcurrency),
	}
	for _, property := range []string{"cooldown_seconds", "timeout_seconds", "concurrency"} {
		field := definitionProperty(root, "MemoryConsolidationConfig", property)
		if field == nil {
			return fmt.Errorf("memory consolidation schema is missing %q", property)
		}
		field.Minimum = json.Number("0")
		field.Maximum = json.Number(strconv.FormatInt(maximums[property], 10))
	}
	return nil
}
