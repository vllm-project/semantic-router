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

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// FileStore manages uploaded files on the local filesystem and tracks
// their metadata in memory. It is safe for concurrent use.
type FileStore struct {
	baseDir string
	mu      sync.RWMutex
	files   map[string]*FileRecord // file_id -> metadata
}

// NewFileStore creates a new FileStore rooted at the given base directory.
// The directory is created if it does not exist.
func NewFileStore(baseDir string) (*FileStore, error) {
	filesDir := filepath.Join(baseDir, "files")
	if err := os.MkdirAll(filesDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create file storage directory: %w", err)
	}

	return &FileStore{
		baseDir: baseDir,
		files:   make(map[string]*FileRecord),
	}, nil
}

// Save stores file content on disk and records its metadata.
// It generates a unique file ID and returns the created FileRecord.
func (fs *FileStore) Save(filename string, content []byte, purpose string) (*FileRecord, error) {
	// Sanitize filename to prevent path traversal (e.g. "../../etc/passwd").
	filename = filepath.Base(filename)
	if filename == "." || filename == "/" {
		return nil, fmt.Errorf("invalid filename")
	}

	fileID := GenerateFileID()

	// Create the file directory: {baseDir}/files/{file_id}/
	fileDir := filepath.Join(fs.baseDir, "files", fileID)
	if err := os.MkdirAll(fileDir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create file directory: %w", err)
	}

	// Write file content — verify the resolved path stays within the file directory.
	filePath := filepath.Join(fileDir, filename)
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return nil, fmt.Errorf("invalid file path")
	}
	absDir, err := filepath.Abs(fileDir)
	if err != nil {
		return nil, fmt.Errorf("invalid directory path")
	}
	if !strings.HasPrefix(absPath, absDir+string(filepath.Separator)) {
		return nil, fmt.Errorf("invalid filename")
	}

	if err := os.WriteFile(filePath, content, 0o644); err != nil {
		return nil, fmt.Errorf("failed to write file: %w", err)
	}

	record := &FileRecord{
		ID:        fileID,
		Object:    "file",
		Bytes:     int64(len(content)),
		CreatedAt: time.Now().Unix(),
		Filename:  filename,
		Purpose:   purpose,
		Status:    "uploaded",
	}

	fs.mu.Lock()
	fs.files[fileID] = record
	fs.mu.Unlock()

	return record, nil
}

// Read returns the content of a file by its ID.
func (fs *FileStore) Read(fileID string) ([]byte, error) {
	fs.mu.RLock()
	record, ok := fs.files[fileID]
	fs.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("file not found: %s", fileID)
	}

	filePath := filepath.Join(fs.baseDir, "files", fileID, record.Filename)
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return data, nil
}

// Delete removes a file from disk and from the in-memory registry.
func (fs *FileStore) Delete(fileID string) error {
	fs.mu.Lock()
	_, ok := fs.files[fileID]
	if !ok {
		fs.mu.Unlock()
		return fmt.Errorf("file not found: %s", fileID)
	}
	delete(fs.files, fileID)
	fs.mu.Unlock()

	fileDir := filepath.Join(fs.baseDir, "files", fileID)
	if err := os.RemoveAll(fileDir); err != nil {
		return fmt.Errorf("failed to delete file directory: %w", err)
	}

	return nil
}

// List returns all file records.
func (fs *FileStore) List() []*FileRecord {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	records := make([]*FileRecord, 0, len(fs.files))
	for _, r := range fs.files {
		records = append(records, r)
	}
	return records
}

// Get returns the FileRecord for a given file ID.
func (fs *FileStore) Get(fileID string) (*FileRecord, error) {
	fs.mu.RLock()
	defer fs.mu.RUnlock()

	record, ok := fs.files[fileID]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", fileID)
	}
	return record, nil
}
