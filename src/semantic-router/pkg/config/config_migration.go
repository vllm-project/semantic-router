package config

import (
	"bytes"
	"fmt"

	"gopkg.in/yaml.v3"
)

// migrationChangeCode marks a note of a form migrate rewrote; a person may
// review it, but the migrated file works as the original did.
const migrationChangeCode = "migration_change"

// Migration is the rewrite of one document: each form's Migrate edits its
// nodes with these helpers, and the notes say what changed and what a person
// must review (a threshold to re-fit, dropped data).
type Migration struct {
	notes []ConfigWarning
}

// Rename renames a key of a mapping in place, keeping its value. When the
// mapping already holds the new key, two sequences are joined (the renamed
// one's items after the others); any other pair is refused.
func (m *Migration) Rename(mapping *yaml.Node, from, to string) error {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return fmt.Errorf("rename %s to %s: not a mapping", from, to)
	}
	index := -1
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == from {
			index = i
		}
	}
	if index < 0 {
		return nil
	}
	if existing := mappingValue(mapping, to); existing != nil {
		renamed := mapping.Content[index+1]
		if existing.Kind != yaml.SequenceNode || renamed.Kind != yaml.SequenceNode {
			return fmt.Errorf("both %s and %s are set; keep one", from, to)
		}
		existing.Content = append(existing.Content, renamed.Content...)
		mapping.Content = append(mapping.Content[:index], mapping.Content[index+2:]...)
		return nil
	}
	mapping.Content[index].Value = to
	return nil
}

// Set writes value under key, in place of the key's value or after the
// mapping's other keys.
func (m *Migration) Set(mapping *yaml.Node, key string, value *yaml.Node) {
	setNode(mapping, key, value)
}

// SetValue writes a Go value under key.
func (m *Migration) SetValue(mapping *yaml.Node, key string, value any) error {
	return setValue(mapping, key, value, false)
}

// Delete removes a key from a mapping.
func (m *Migration) Delete(mapping *yaml.Node, key string) {
	deleteKey(mapping, key)
}

// Changed notes a rewrite a person may want to review.
func (m *Migration) Changed(path, message string) {
	m.notes = append(m.notes, ConfigWarning{Code: migrationChangeCode, Field: path, Message: message})
}

// Note notes what a person must do for the migrated file to work as the
// original did.
func (m *Migration) Note(path, message string) {
	m.notes = append(m.notes, ConfigWarning{Code: migrationActionCode, Field: path, Message: message})
}

// Scalar is a string scalar node.
func Scalar(value string) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: value}
}

// MigrateConfig rewrites a document's removed and renamed forms in the forms
// that replace them, keeping every other field as written, and then its
// deprecated model fields (MigrateModelResources). The notes name what
// changed and what a person must review. The migrated document must load.
func MigrateConfig(data []byte) ([]byte, []ConfigWarning, error) {
	migrated, notes, err := migrateConfigForms(data, configForms)
	if err != nil {
		return nil, nil, err
	}
	modelsMigrated, kept, err := MigrateModelResources(migrated)
	if err != nil {
		return nil, nil, err
	}
	if !bytes.Equal(modelsMigrated, migrated) {
		notes = append(notes, ConfigWarning{
			Code: migrationChangeCode, Field: "global.model_catalog",
			Message: "deprecated model fields are written as the deployments, bindings and rule deployments the Router reads them as",
		})
	}
	if _, err := ParseYAMLBytesWithoutEnvExpansion(modelsMigrated); err != nil {
		return nil, nil, fmt.Errorf("the migrated configuration does not load: %w", err)
	}
	return modelsMigrated, append(notes, kept...), nil
}

// migrateConfigForms applies every form's rewrite, in registry order, to
// every place it occurs. A form without a rewrite stops the migration with
// what it becomes.
func migrateConfigForms(data []byte, forms []ConfigForm) ([]byte, []ConfigWarning, error) {
	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil, nil, fmt.Errorf("failed to parse config file: %w", err)
	}
	root := documentRoot(&document)
	if root == nil {
		return data, nil, nil
	}
	m := &Migration{}
	changed := false
	for _, form := range forms {
		matches := form.matches(root)
		for _, at := range matches {
			what, becomes := form.describe(at)
			if form.Migrate == nil {
				return nil, nil, CheckError(at.Path, form.Code, what+"; it has no mechanical rewrite: "+becomes)
			}
			if err := form.Migrate(m, at); err != nil {
				return nil, nil, fmt.Errorf("%s: %w", at.Path, err)
			}
			changed = true
		}
		if len(matches) > 0 {
			what, becomes := form.describe(matches[0])
			message := what + "; rewritten as " + becomes
			if len(matches) > 1 {
				message += fmt.Sprintf(" (%d places)", len(matches))
			}
			m.Changed(matches[0].Path, message)
		}
	}
	if !changed {
		return data, m.notes, nil
	}
	migrated, err := encodeDocument(&document)
	if err != nil {
		return nil, nil, err
	}
	return migrated, m.notes, nil
}
