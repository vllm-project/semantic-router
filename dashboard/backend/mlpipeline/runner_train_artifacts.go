package mlpipeline

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// snapshotTrainOutputFiles preserves the stable TrainDir contract used by
// deployment while binding every completed training job to immutable paths in
// its own JobDir. The caller holds trainMu, so a later run cannot remove the
// stable names until all snapshots from this run have been created.
func (r *Runner) snapshotTrainOutputFiles(jobID string, stablePaths []string) ([]string, error) {
	validatedSources, err := r.validateOutputFilesInRoot("train", r.TrainDir(), stablePaths)
	if err != nil {
		return nil, err
	}

	jobDir := r.JobDir(jobID)
	if filepath.Dir(jobDir) != r.dataDir {
		return nil, errors.New("training snapshot directory is invalid")
	}
	if mkdirErr := os.Mkdir(jobDir, 0o700); mkdirErr != nil {
		return nil, fmt.Errorf("create training snapshot directory: %w", mkdirErr)
	}
	if chmodErr := os.Chmod(jobDir, 0o700); chmodErr != nil {
		_ = os.RemoveAll(jobDir)
		return nil, fmt.Errorf("secure training snapshot directory: %w", chmodErr)
	}

	completed := false
	defer func() {
		if !completed {
			_ = os.RemoveAll(jobDir)
		}
	}()

	snapshots := make([]string, 0, len(validatedSources))
	for _, source := range validatedSources {
		destination := filepath.Join(jobDir, filepath.Base(source))
		if copyErr := r.copyTrainOutputFile(source, destination); copyErr != nil {
			return nil, copyErr
		}
		snapshots = append(snapshots, destination)
	}
	if syncErr := syncPipelineDirectory(jobDir); syncErr != nil {
		return nil, syncErr
	}
	validatedSnapshots, err := r.validateOutputFiles(jobID, "train", snapshots)
	if err != nil {
		return nil, err
	}
	completed = true
	return validatedSnapshots, nil
}

func (r *Runner) copyTrainOutputFile(source, destination string) error {
	sourceFile, sourceInfo, err := r.OpenManagedFile(source)
	if err != nil {
		return errors.New("training output could not be opened for snapshotting")
	}
	defer sourceFile.Close()
	if sourceInfo.Size() < 0 || sourceInfo.Size() > maxPipelineArtifactFile {
		return errors.New("training output exceeds its snapshot size budget")
	}

	destinationFile, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return errors.New("training snapshot file could not be created")
	}
	keepDestination := false
	defer func() {
		_ = destinationFile.Close()
		if !keepDestination {
			_ = os.Remove(destination)
		}
	}()
	if chmodErr := destinationFile.Chmod(0o600); chmodErr != nil {
		return errors.New("training snapshot permissions could not be secured")
	}

	written, err := io.Copy(destinationFile, io.LimitReader(sourceFile, int64(maxPipelineArtifactFile)+1))
	if err != nil || written != sourceInfo.Size() || written > maxPipelineArtifactFile {
		return errors.New("training output changed or exceeded its snapshot size budget")
	}
	if err := destinationFile.Sync(); err != nil {
		return errors.New("training snapshot could not be synchronized")
	}
	if err := destinationFile.Close(); err != nil {
		return errors.New("training snapshot could not be finalized")
	}
	keepDestination = true
	return nil
}

func syncPipelineDirectory(path string) error {
	dir, err := os.Open(path)
	if err != nil {
		return errors.New("training snapshot directory could not be opened")
	}
	if err := dir.Sync(); err != nil {
		_ = dir.Close()
		return errors.New("training snapshot directory could not be synchronized")
	}
	if err := dir.Close(); err != nil {
		return errors.New("training snapshot directory could not be finalized")
	}
	return nil
}
