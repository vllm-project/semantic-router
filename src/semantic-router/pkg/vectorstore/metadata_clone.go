/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vectorstore

func cloneVectorStore(store *VectorStore) *VectorStore {
	if store == nil {
		return nil
	}
	copied := *store
	copied.ExpiresAfter = cloneExpirationPolicy(store.ExpiresAfter)
	copied.Metadata = cloneMetadata(store.Metadata)
	return &copied
}

func cloneVectorStoreFile(file *VectorStoreFile) *VectorStoreFile {
	if file == nil {
		return nil
	}
	copied := *file
	copied.ChunkingStrategy = cloneChunkingStrategy(file.ChunkingStrategy)
	copied.LastError = cloneFileError(file.LastError)
	return &copied
}

func cloneFileRecord(record *FileRecord) *FileRecord {
	if record == nil {
		return nil
	}
	copied := *record
	return &copied
}

func cloneExpirationPolicy(policy *ExpirationPolicy) *ExpirationPolicy {
	if policy == nil {
		return nil
	}
	copied := *policy
	return &copied
}

func cloneChunkingStrategy(strategy *ChunkingStrategy) *ChunkingStrategy {
	if strategy == nil {
		return nil
	}
	copied := *strategy
	if strategy.Static != nil {
		static := *strategy.Static
		copied.Static = &static
	}
	return &copied
}

func cloneFileError(err *FileError) *FileError {
	if err == nil {
		return nil
	}
	copied := *err
	return &copied
}

func cloneMetadata(metadata map[string]interface{}) map[string]interface{} {
	if metadata == nil {
		return nil
	}
	copied := make(map[string]interface{}, len(metadata))
	for key, value := range metadata {
		copied[key] = value
	}
	return copied
}
