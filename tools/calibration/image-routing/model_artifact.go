package main

import (
	"bufio"
	"crypto/sha1" // #nosec G505 -- git blob ids are sha1 by definition; used only to match the download record.
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// requiredModelFiles are the files the candle multimodal loader reads; the
// report cannot describe the weights it scored with unless they are present
// and hashed. optionalModelFiles are hashed when present.
var (
	requiredModelFiles = []string{"config.json", "model.safetensors", "tokenizer.json"}
	optionalModelFiles = []string{"tokenizer_config.json", "special_tokens_map.json"}
)

// downloadRecord is what the Hugging Face CLI writes next to a downloaded
// file (.cache/huggingface/download/<file>.metadata): the snapshot commit on
// the first line and the file's etag on the second. The etag is the sha256
// of an LFS object (the weights) or the git blob sha1 of a small file.
type downloadRecord struct {
	Snapshot string
	Etag     string
}

var (
	sha256Hex = regexp.MustCompile(`^[0-9a-f]{64}$`)
	sha1Hex   = regexp.MustCompile(`^[0-9a-f]{40}$`)
)

// modelArtifact binds the report to the model bytes actually loaded rather
// than to the caller's -artifact-revision claim: every loader input is
// hashed into the report, and where the Hugging Face download cache recorded
// what a file resolved to, the recorded snapshot must equal the claimed
// revision and the recorded etag must match the bytes on disk. A file that
// was replaced after download, or a claim that names another snapshot, fails
// the run instead of producing a report that certifies weights it did not
// score with. The CI gate performs the snapshot half of this check in shell
// before running; doing both here makes a local report carry the same
// evidence.
func modelArtifact(dir, revision string) (map[string]string, error) {
	hashes := map[string]string{}
	check := func(name string, required bool) error {
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		if err != nil {
			if required || !os.IsNotExist(err) {
				return fmt.Errorf("model file %q: %w", name, err)
			}
			return nil
		}
		sum := sha256.Sum256(data)
		digest := hex.EncodeToString(sum[:])
		hashes[name] = "sha256:" + digest
		record, err := readDownloadRecord(dir, name)
		if err != nil {
			return err
		}
		if record == nil {
			return nil
		}
		if record.Snapshot != revision {
			return fmt.Errorf("model file %q resolved to snapshot %s, but -artifact-revision claims %s", name, record.Snapshot, revision)
		}
		switch {
		case sha256Hex.MatchString(record.Etag):
			if record.Etag != digest {
				return fmt.Errorf("model file %q on disk (sha256 %s) is not the file the download record certifies (%s)", name, digest, record.Etag)
			}
		case sha1Hex.MatchString(record.Etag):
			if blob := gitBlobSHA1(data); blob != record.Etag {
				return fmt.Errorf("model file %q on disk (blob %s) is not the file the download record certifies (%s)", name, blob, record.Etag)
			}
		default:
			return fmt.Errorf("download record for %q has an unrecognized etag %q", name, record.Etag)
		}
		return nil
	}
	for _, name := range requiredModelFiles {
		if err := check(name, true); err != nil {
			return nil, err
		}
	}
	for _, name := range optionalModelFiles {
		if err := check(name, false); err != nil {
			return nil, err
		}
	}
	return hashes, nil
}

// readDownloadRecord returns the download record for a file, or nil when no
// record exists (a model copied by hand carries none; the hashes still bind
// the bytes). A record that exists but is incomplete is an error, never
// treated as absent.
func readDownloadRecord(dir, name string) (*downloadRecord, error) {
	path := filepath.Join(dir, ".cache", "huggingface", "download", name+".metadata")
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read download record for %q: %w", name, err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	var lines []string
	for len(lines) < 2 && scanner.Scan() {
		lines = append(lines, strings.TrimSpace(scanner.Text()))
	}
	if len(lines) < 2 || lines[0] == "" || lines[1] == "" {
		return nil, fmt.Errorf("download record for %q lacks a snapshot and etag on its first two lines", name)
	}
	return &downloadRecord{Snapshot: lines[0], Etag: strings.Trim(lines[1], `"`)}, nil
}

// gitBlobSHA1 is the object id git assigns to a file's bytes, which is the
// etag the hub reports for files stored in git rather than LFS.
func gitBlobSHA1(data []byte) string {
	h := sha1.New() // #nosec G401 -- identity match against git's own blob id, not a security digest.
	fmt.Fprintf(h, "blob %d\x00", len(data))
	h.Write(data)
	return hex.EncodeToString(h.Sum(nil))
}
