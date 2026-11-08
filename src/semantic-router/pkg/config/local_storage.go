package config

import "strings"

// DefaultVectorStoreFileStorageDir is where the vector store keeps uploaded
// files when global.stores.vector_store.file_storage_dir is unset. The Helm
// chart mounts a writable volume at its parent, /var/lib/vsr.
const DefaultVectorStoreFileStorageDir = "/var/lib/vsr/data"

// LocalStorageDir is a directory on local disk that the configuration has the
// Router write.
type LocalStorageDir struct {
	// Setting is the configuration key that chooses the directory.
	Setting string
	// Path is the directory as configured; a relative path is relative to the
	// Router's working directory.
	Path string
}

// LocalStorageDirs lists the directories the configuration has the Router
// write: the vector store's file directory when the vector store is enabled,
// and a Router Flow state directory set for the file backend while a decision
// runs workflows. A Flow state directory left unset lives in the user cache
// directory, which every maintained deployment keeps writable.
func (c *RouterConfig) LocalStorageDirs() []LocalStorageDir {
	if c == nil {
		return nil
	}
	var dirs []LocalStorageDir
	if c.VectorStore != nil && c.VectorStore.Enabled {
		path := strings.TrimSpace(c.VectorStore.FileStorageDir)
		if path == "" {
			path = DefaultVectorStoreFileStorageDir
		}
		dirs = append(dirs, LocalStorageDir{Setting: "global.stores.vector_store.file_storage_dir", Path: path})
	}
	state := c.Looper.Flow.State.WithDefaults()
	if dir := strings.TrimSpace(state.File.Directory); dir != "" &&
		state.StoreBackend == WorkflowStateBackendFile && c.HasFlowDecision() {
		dirs = append(dirs, LocalStorageDir{Setting: "global.integrations.looper.flow.state.file.directory", Path: dir})
	}
	return dirs
}
