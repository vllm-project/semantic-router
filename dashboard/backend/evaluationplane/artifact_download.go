package evaluationplane

import (
	"crypto/sha256"
	"fmt"
	"io"
	"path/filepath"
	"strings"
)

const publicChecksumArtifactName = "checksums.sha256"

func (s *Service) OpenArtifactAs(actor Actor, runID, artifactID string) (*OpenedArtifact, error) {
	releaseOperation, operationErr := s.beginOperation()
	if operationErr != nil {
		return nil, operationErr
	}
	operationOwned := true
	defer func() {
		if operationOwned {
			releaseOperation()
		}
	}()
	release, err := s.acquireAuthorizedEvidenceRead(actor, runID)
	if err != nil {
		return nil, err
	}
	evidenceReadOwned := true
	defer func() {
		if evidenceReadOwned {
			release()
		}
	}()
	report, err := s.decodedReport(runID)
	if err != nil {
		return nil, err
	}
	artifact, ok := findArtifact(report, artifactID)
	if !ok || strings.TrimSpace(artifact.URI) == "" {
		return nil, fmt.Errorf("%w: evaluation artifact", ErrNotFound)
	}
	contract, known := publicArtifactContracts[artifact.Name]
	if artifact.Name != artifact.URI || filepath.Base(artifact.URI) != artifact.URI || !known ||
		artifact.Kind != contract.Kind || artifact.MediaType != contract.MediaType ||
		!digestPattern.MatchString(artifact.Digest) || artifact.SizeBytes < 0 {
		return nil, fmt.Errorf("%w: evaluation artifact metadata is invalid", ErrInvalid)
	}
	opened, err := s.store.OpenArtifact(runID, artifact.URI)
	if err != nil {
		return nil, err
	}
	if err := verifyOpenedArtifact(opened, artifact); err != nil {
		_ = opened.File.Close()
		return nil, err
	}
	if err := s.verifyPublicChecksum(runID, report, artifact); err != nil {
		_ = opened.File.Close()
		return nil, err
	}
	if err := s.rejectConfiguredSecretArtifact(opened.File, contract.MediaType); err != nil {
		_ = opened.File.Close()
		return nil, err
	}
	if _, err := opened.File.Seek(0, io.SeekStart); err != nil {
		_ = opened.File.Close()
		return nil, fmt.Errorf("rewind verified evaluation artifact: %w", err)
	}
	opened.MediaType = contract.MediaType
	// A verified descriptor pins the immutable inode even if a concurrent
	// lifecycle deletion unlinks the run directory. Release store and Service
	// coordination before handing bytes to the network so a slow client cannot
	// block evidence publication, deletion, or shutdown.
	release()
	releaseOperation()
	evidenceReadOwned = false
	operationOwned = false
	return opened, nil
}

func verifyOpenedArtifact(opened *OpenedArtifact, artifact Artifact) error {
	if opened.Size != artifact.SizeBytes {
		return fmt.Errorf("%w: evaluation artifact size does not match its report metadata", ErrInvalid)
	}
	hash := sha256.New()
	written, err := io.Copy(hash, opened.File)
	if err != nil {
		return fmt.Errorf("verify evaluation artifact digest: %w", err)
	}
	if written != artifact.SizeBytes || fmt.Sprintf("sha256:%x", hash.Sum(nil)) != artifact.Digest {
		return fmt.Errorf("%w: evaluation artifact digest does not match its report metadata", ErrInvalid)
	}
	if _, err := opened.File.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind verified evaluation artifact: %w", err)
	}
	return nil
}

func (s *Service) verifyPublicChecksum(runID string, report Report, artifact Artifact) error {
	receipt, ok := findArtifactByName(report, publicChecksumArtifactName)
	if !ok || receipt.URI != publicChecksumArtifactName || !digestPattern.MatchString(receipt.Digest) || receipt.SizeBytes < 0 {
		return fmt.Errorf("%w: public artifact checksum receipt is unavailable", ErrInvalid)
	}
	opened, err := s.store.OpenArtifact(runID, receipt.URI)
	if err != nil {
		return err
	}
	defer func() { _ = opened.File.Close() }()
	if verifyErr := verifyOpenedArtifact(opened, receipt); verifyErr != nil {
		return verifyErr
	}
	if opened.Size > 4*1024*1024 {
		return fmt.Errorf("%w: public artifact checksum receipt is too large", ErrInvalid)
	}
	data, err := io.ReadAll(opened.File)
	if err != nil {
		return fmt.Errorf("read public artifact checksum receipt: %w", err)
	}
	checksums, err := parsePublicChecksumReceipt(data)
	if err != nil {
		return err
	}
	expected := make(map[string]string)
	for _, reported := range reportArtifacts(report) {
		if _, known := publicArtifactContracts[reported.Name]; reported.Name != reported.URI || !known ||
			!digestPattern.MatchString(reported.Digest) || reported.SizeBytes < 0 {
			return fmt.Errorf("%w: report contains an invalid public artifact", ErrInvalid)
		}
		if reported.Name == publicChecksumArtifactName {
			continue
		}
		if _, duplicate := expected[reported.Name]; duplicate {
			return fmt.Errorf("%w: report contains duplicate public artifacts", ErrInvalid)
		}
		expected[reported.Name] = strings.TrimPrefix(reported.Digest, "sha256:")
	}
	if len(checksums) != len(expected) {
		return fmt.Errorf("%w: public artifact checksum receipt set does not match the report", ErrInvalid)
	}
	for name, digest := range expected {
		if checksums[name] != digest {
			return fmt.Errorf("%w: public artifact checksum receipt does not match the report", ErrInvalid)
		}
	}
	if artifact.Name != publicChecksumArtifactName && checksums[artifact.Name] != strings.TrimPrefix(artifact.Digest, "sha256:") {
		return fmt.Errorf("%w: evaluation artifact is absent from the public checksum receipt", ErrInvalid)
	}
	return nil
}

func parsePublicChecksumReceipt(data []byte) (map[string]string, error) {
	if len(data) == 0 || data[len(data)-1] != '\n' {
		return nil, fmt.Errorf("%w: public artifact checksum receipt is invalid", ErrInvalid)
	}
	checksums := make(map[string]string)
	for _, line := range strings.Split(strings.TrimSuffix(string(data), "\n"), "\n") {
		digest, name, found := strings.Cut(line, "  ")
		_, known := publicArtifactContracts[name]
		if !found || !digestPattern.MatchString("sha256:"+digest) ||
			!known || name == publicChecksumArtifactName || checksums[name] != "" {
			return nil, fmt.Errorf("%w: public artifact checksum receipt is invalid", ErrInvalid)
		}
		checksums[name] = digest
	}
	return checksums, nil
}

func reportArtifacts(report Report) []Artifact {
	artifacts := append([]Artifact(nil), report.Artifacts...)
	for _, track := range report.Tracks {
		artifacts = append(artifacts, track.Artifacts...)
	}
	return artifacts
}

func findArtifact(report Report, artifactID string) (Artifact, bool) {
	for _, artifact := range report.Artifacts {
		if artifact.ID == artifactID {
			return artifact, true
		}
	}
	for _, track := range report.Tracks {
		for _, artifact := range track.Artifacts {
			if artifact.ID == artifactID {
				return artifact, true
			}
		}
	}
	return Artifact{}, false
}

func findArtifactByName(report Report, name string) (Artifact, bool) {
	for _, artifact := range report.Artifacts {
		if artifact.Name == name {
			return artifact, true
		}
	}
	for _, track := range report.Tracks {
		for _, artifact := range track.Artifacts {
			if artifact.Name == name {
				return artifact, true
			}
		}
	}
	return Artifact{}, false
}
