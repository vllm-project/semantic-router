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

import "context"

// StoreRegistry persists VectorStore metadata so the router can recover
// its store inventory after a restart.
type StoreRegistry interface {
	SaveStore(ctx context.Context, vs *VectorStore) error
	GetStore(ctx context.Context, id string) (*VectorStore, error)
	ListStores(ctx context.Context) ([]*VectorStore, error)
	DeleteStore(ctx context.Context, id string) error
}

// FileRegistry persists FileRecord metadata so the router can recover
// its file inventory after a restart.
type FileRegistry interface {
	SaveFile(ctx context.Context, fr *FileRecord) error
	GetFile(ctx context.Context, id string) (*FileRecord, error)
	ListFiles(ctx context.Context) ([]*FileRecord, error)
	DeleteFile(ctx context.Context, id string) error
}
