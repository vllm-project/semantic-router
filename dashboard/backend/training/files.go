package training

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

const MaxFileBytes int64 = 64 << 20

var ErrTooLarge = fmt.Errorf("training file exceeds %d bytes", MaxFileBytes)

type ownedFile struct {
	c.File
	AttemptID string `json:"attempt_id,omitempty"`
}

// putFile stores immutable bytes before recording their handle. Failed copies and
// rejected commits remove their bytes. A crash may leave unreferenced bytes, never
// a public handle pointing at a partial upload.
func (s *Service) putFile(ctx context.Context, owner, attempt string, input io.Reader, commit func(*workflowstore.TrainingTx, c.File) error) (c.File, error) {
	handle := metadata("file").ID
	path := filepath.Join(s.directory, handle)
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return c.File{}, err
	}
	committed := false
	defer func() {
		_ = file.Close()
		if !committed {
			_ = os.Remove(path)
		}
	}()
	digest := sha256.New()
	size, err := io.Copy(io.MultiWriter(file, digest), io.LimitReader(input, MaxFileBytes+1))
	if err != nil {
		return c.File{}, err
	}
	if size > MaxFileBytes {
		return c.File{}, ErrTooLarge
	}
	if operationErr := file.Sync(); operationErr != nil {
		return c.File{}, operationErr
	}
	if operationErr := file.Close(); operationErr != nil {
		return c.File{}, operationErr
	}
	// Sync the directory entry before the SQLite commit publishes it.
	dir, err := os.Open(s.directory)
	if err != nil {
		return c.File{}, err
	}
	err = dir.Sync()
	_ = dir.Close()
	if err != nil {
		return c.File{}, err
	}
	value := c.File{Handle: handle, Digest: fmt.Sprintf("sha256:%x", digest.Sum(nil)), SizeBytes: size}
	err = s.update(ctx, func(tx *workflowstore.TrainingTx) error {
		if operationErr := commit(tx, value); operationErr != nil {
			return operationErr
		}
		return insert(tx, "files", owner, handle, ownedFile{File: value, AttemptID: attempt})
	})
	committed = err == nil
	return value, err
}

func (s *Service) Upload(ctx context.Context, owner string, input io.Reader) (c.Upload, error) {
	upload := c.Upload{Metadata: metadata("upload")}
	_, err := s.putFile(ctx, owner, "", input, func(tx *workflowstore.TrainingTx, file c.File) error {
		upload.File = file
		return insert(tx, "uploads", owner, upload.ID, upload)
	})
	return upload, err
}

// PutOutput is an adapter-only boundary. Attempt ownership is checked at commit,
// after copying bytes, so cancellation or completion during upload cannot register
// output bytes against a settled attempt.
func (s *Service) PutOutput(ctx context.Context, owner, runID, attemptID string, input io.Reader) (c.File, error) {
	return s.putFile(ctx, owner, attemptID, input, func(tx *workflowstore.TrainingTx, file c.File) error {
		g, err := read[c.RunGraph](tx, owner, "runs", runID)
		if err != nil {
			return err
		}
		_, err = activeTask(&g, attemptID)
		return err
	})
}

func (s *Service) Download(ctx context.Context, owner, handle string) (*os.File, error) {
	if _, err := s.Get(ctx, owner, "files", handle); err != nil {
		return nil, err
	}
	return os.Open(filepath.Join(s.directory, handle))
}
