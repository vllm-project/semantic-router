package config

import "encoding/hex"

// GetModelRegistryInfoByRevision identifies a registered source checkpoint by
// its complete, explicitly declared commit. Derived artifact directory names
// are not model identities. Unknown, mutable and ambiguous revisions stay unknown.
// This is source metadata, not a claim that a derived graph is byte-identical.
func GetModelRegistryInfoByRevision(revision string) *ModelRegistryInfo {
	return registryInfoByRevision(DefaultModelRegistry, revision)
}

func registryInfoByRevision(models []ModelSpec, revision string) *ModelRegistryInfo {
	if len(revision) != 40 {
		return nil
	}
	if _, err := hex.DecodeString(revision); err != nil {
		return nil
	}
	var match *ModelRegistryInfo
	for _, model := range models {
		if model.Revision != revision || model.RepoID == "" {
			continue
		}
		if match != nil {
			return nil
		}
		info := model.RegistryInfo()
		match = &info
	}
	return match
}
