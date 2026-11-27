//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

type constantEmbedder struct{}

func (constantEmbedder) Embed(context.Context, string) ([]float32, error) {
	return []float32{1, 0, 0}, nil
}
func (constantEmbedder) Dimension() int { return 3 }

// flakyDeleteBackend cannot remove chunks while down.
type flakyDeleteBackend struct {
	*vectorstore.MemoryBackend
	down bool
}

func (b *flakyDeleteBackend) DeleteByFileID(ctx context.Context, storeID, fileID string) error {
	if b.down {
		return errors.New("backend unavailable")
	}
	return b.MemoryBackend.DeleteByFileID(ctx, storeID, fileID)
}

type fileDeleteFixture struct {
	server   *ClassificationAPIServer
	manager  *vectorstore.Manager
	files    *vectorstore.FileStore
	file     *vectorstore.FileRecord
	storeIDs []string
}

// newFileDeleteFixture uploads one file and attaches it, through the HTTP
// route, to two vector stores, waiting until both have its chunk.
func newFileDeleteFixture(t *testing.T, backend vectorstore.VectorStoreBackend) fileDeleteFixture {
	t.Helper()
	ctx := context.Background()
	manager := vectorstore.NewManager(backend, vectorstore.NewMemoryMetadataRegistry(), 3, vectorstore.BackendTypeMemory)
	files, err := vectorstore.NewFileStore(t.TempDir(), vectorstore.NewMemoryMetadataRegistry())
	if err != nil {
		t.Fatal(err)
	}
	pipeline := vectorstore.NewIngestionPipeline(backend, files, manager, constantEmbedder{}, vectorstore.PipelineConfig{Workers: 1})
	pipeline.Start()
	t.Cleanup(func() { _ = pipeline.Stop(ctx) })
	registry := routerruntime.NewRegistry(nil)
	registry.SetVectorStoreRuntime(&routerruntime.VectorStoreRuntime{
		Manager: manager, FileStore: files, Pipeline: pipeline, Embedder: constantEmbedder{},
	})
	f := fileDeleteFixture{server: &ClassificationAPIServer{runtimeRegistry: registry}, manager: manager, files: files}
	if f.file, err = files.Save("secret.txt", []byte("The launch code is 1234."), "assistants"); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"kb-a", "kb-b"} {
		vs, createErr := manager.CreateStore(ctx, vectorstore.CreateStoreRequest{Name: name})
		if createErr != nil {
			t.Fatal(createErr)
		}
		f.storeIDs = append(f.storeIDs, vs.ID)
		attached := serveVectorStoreRoute(f.server.handleAttachFile, http.MethodPost,
			apiStorageVectorStoresPath+"/"+vs.ID+"/files", `{"file_id":"`+f.file.ID+`"}`)
		var vsf vectorstore.VectorStoreFile
		if attached.Code != http.StatusOK || json.Unmarshal(attached.Body.Bytes(), &vsf) != nil {
			t.Fatalf("attach: %d %s", attached.Code, attached.Body.String())
		}
		waitForVectorStoreFile(t, pipeline, vsf.ID, "completed")
		if hits := searchVectorStore(t, f.server, vs.ID); len(hits) != 1 || hits[0].FileID != f.file.ID {
			t.Fatalf("store %s before the delete answers %+v, want one chunk of %s", vs.ID, hits, f.file.ID)
		}
	}
	return f
}

// TestDeleteFileRemovesItFromEveryVectorStore deletes an uploaded file that
// two vector stores hold and checks that neither store still answers with its
// text, lists it, or counts it.
func TestDeleteFileRemovesItFromEveryVectorStore(t *testing.T) {
	f := newFileDeleteFixture(t, vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{}))

	f.deleteFile(t, http.StatusOK)
	f.requireFileGone(t, f.storeIDs)
}

// TestDeleteFileKeepsTheFileWhenItsChunksStay checks that a delete whose
// chunks cannot be removed answers 500 and keeps the file and every
// attachment as they were, so a retry once the backend is back removes the
// chunks.
func TestDeleteFileKeepsTheFileWhenItsChunksStay(t *testing.T) {
	backend := &flakyDeleteBackend{MemoryBackend: vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{})}
	f := newFileDeleteFixture(t, backend)

	backend.down = true
	deleted := f.deleteFile(t, http.StatusInternalServerError)
	if code := parseErrorResponse(t, deleted.Body.Bytes()); code != "DELETE_FAILED" {
		t.Fatalf("delete file code = %s, want DELETE_FAILED", code)
	}
	if _, err := f.files.Get(f.file.ID); err != nil {
		t.Fatalf("file was deleted although its chunks stayed: %v", err)
	}
	for _, id := range f.storeIDs {
		if listed := f.listFiles(t, id); len(listed) != 1 || listed[0].FileID != f.file.ID || listed[0].Status != "completed" {
			t.Fatalf("store %s lists %+v after the failed delete, want the completed attachment", id, listed)
		}
		if counts := f.fileCounts(t, id); counts != (vectorstore.FileCounts{Completed: 1, Total: 1}) {
			t.Fatalf("store %s file_counts = %+v after the failed delete, want completed 1 of 1", id, counts)
		}
	}

	backend.down = false
	f.deleteFile(t, http.StatusOK)
	f.requireFileGone(t, f.storeIDs)
}

// TestDeleteFileDropsAttachmentsOfADeletedStore deletes one of the two stores
// that hold a file, then the file: the delete succeeds and the other store
// loses the file's chunks.
func TestDeleteFileDropsAttachmentsOfADeletedStore(t *testing.T) {
	f := newFileDeleteFixture(t, vectorstore.NewMemoryBackend(vectorstore.MemoryBackendConfig{}))
	gone := serveVectorStoreRoute(f.server.handleDeleteVectorStore, http.MethodDelete, apiStorageVectorStoresPath+"/"+f.storeIDs[0], "")
	if gone.Code != http.StatusOK {
		t.Fatalf("delete store: %d %s", gone.Code, gone.Body.String())
	}

	f.deleteFile(t, http.StatusOK)
	f.requireFileGone(t, f.storeIDs[1:])
	if listed := f.listFiles(t, f.storeIDs[0]); len(listed) != 0 {
		t.Fatalf("deleted store %s still lists %+v", f.storeIDs[0], listed)
	}
}

func (f fileDeleteFixture) deleteFile(t *testing.T, status int) *httptest.ResponseRecorder {
	t.Helper()
	response := serveVectorStoreRoute(f.server.handleDeleteFile, http.MethodDelete, apiStorageFilesPath+"/"+f.file.ID, "")
	if response.Code != status {
		t.Fatalf("delete file: %d %s, want %d", response.Code, response.Body.String(), status)
	}
	return response
}

func (f fileDeleteFixture) listFiles(t *testing.T, storeID string) []*vectorstore.VectorStoreFile {
	t.Helper()
	listed := serveVectorStoreRoute(f.server.handleListVectorStoreFiles, http.MethodGet, apiStorageVectorStoresPath+"/"+storeID+"/files", "")
	var page objectListResponse[*vectorstore.VectorStoreFile]
	if err := json.Unmarshal(listed.Body.Bytes(), &page); err != nil {
		t.Fatalf("list files of %s: %d %s", storeID, listed.Code, listed.Body.String())
	}
	return page.Data
}

func (f fileDeleteFixture) fileCounts(t *testing.T, storeID string) vectorstore.FileCounts {
	t.Helper()
	vs, err := f.manager.GetStore(storeID)
	if err != nil {
		t.Fatal(err)
	}
	return vs.FileCounts
}

// requireFileGone checks that the stores neither answer with, list, nor count
// the file, and that the file store no longer holds it.
func (f fileDeleteFixture) requireFileGone(t *testing.T, storeIDs []string) {
	t.Helper()
	for _, id := range storeIDs {
		if hits := searchVectorStore(t, f.server, id); len(hits) != 0 {
			t.Fatalf("store %s still answers with the deleted file: %+v", id, hits)
		}
		if listed := f.listFiles(t, id); len(listed) != 0 {
			t.Fatalf("store %s still lists the deleted file: %+v", id, listed)
		}
		if counts := f.fileCounts(t, id); counts != (vectorstore.FileCounts{}) {
			t.Fatalf("store %s file_counts = %+v, want all zero", id, counts)
		}
	}
	if _, err := f.files.Get(f.file.ID); err == nil {
		t.Fatalf("file %s is still in the file store", f.file.ID)
	}
}

func waitForVectorStoreFile(t *testing.T, pipeline *vectorstore.IngestionPipeline, id, status string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if vsf, err := pipeline.GetFileStatus(id); err == nil && vsf.Status == status {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("vector store file %s never reached %s", id, status)
}

func searchVectorStore(t *testing.T, server *ClassificationAPIServer, storeID string) []vectorstore.SearchResult {
	t.Helper()
	response := httptest.NewRecorder()
	server.handleSearchVectorStore(response, httptest.NewRequest(http.MethodPost,
		apiStorageVectorStoresPath+"/"+storeID+"/search", strings.NewReader(`{"query":"launch code"}`)))
	if response.Code != http.StatusOK {
		t.Fatalf("search %s: %d %s", storeID, response.Code, response.Body.String())
	}
	var page objectListResponse[vectorstore.SearchResult]
	if err := json.Unmarshal(response.Body.Bytes(), &page); err != nil {
		t.Fatal(err)
	}
	return page.Data
}
