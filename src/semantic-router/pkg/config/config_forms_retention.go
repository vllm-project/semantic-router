package config

import (
	"fmt"

	"gopkg.in/yaml.v3"
)

// retentionForms are the forms of a decision's retention before it became a
// decision attribute: the emits list, whose only kind was retention.
var retentionForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("decisions[].emits"),
		What:    "emits is removed",
		Becomes: "the decision's retention attribute",
		Migrate: migrateEmits,
	},
}

// migrateEmits writes a decision's retention directive as its retention
// attribute. emits held no other kind.
func migrateEmits(m *Migration, at FormMatch) error {
	decision := at.Parent
	var retention *yaml.Node
	if at.Node.Kind == yaml.SequenceNode {
		for _, item := range at.Node.Content {
			if kind := scalarAt(item, "kind"); kind != "retention" {
				return fmt.Errorf("emits kind %q has no replacement", kind)
			}
			if retention != nil {
				return fmt.Errorf("emits declares retention twice; keep one")
			}
			if retention = mappingValue(item, "retention"); retention == nil {
				return fmt.Errorf("an emits retention entry has no retention block")
			}
		}
	}
	if retention != nil && mappingValue(decision, "retention") != nil {
		return fmt.Errorf("both emits and retention are set; keep retention")
	}
	m.Delete(decision, "emits")
	if retention != nil {
		m.Set(decision, "retention", retention)
	}
	return nil
}
