package routerruntime

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// UnwritableDir is a directory the configuration has the Router write that
// this process cannot write.
type UnwritableDir struct {
	config.LocalStorageDir
	Err error
}

// Problem says what fails and how to fix it, without the setting's name.
func (d UnwritableDir) Problem() string {
	return fmt.Sprintf(
		"%s is not writable (%v); mount a writable volume at that path, or set %s to a writable directory",
		d.Path, d.Err, d.Setting)
}

func (d UnwritableDir) Error() string {
	return d.Setting + ": " + d.Problem()
}

// UnwritableDirs returns the directories of cfg's LocalStorageDirs that this
// process cannot write. It creates nothing: a directory that does not exist
// yet is checked at its nearest existing parent, where the Router creates it.
func UnwritableDirs(cfg *config.RouterConfig) []UnwritableDir {
	var unwritable []UnwritableDir
	for _, dir := range cfg.LocalStorageDirs() {
		if err := checkWritableDir(dir.Path); err != nil {
			unwritable = append(unwritable, UnwritableDir{LocalStorageDir: dir, Err: err})
		}
	}
	return unwritable
}

// CheckLocalStorage fails when cfg has the Router write a directory this
// process cannot write, naming each such setting.
func CheckLocalStorage(cfg *config.RouterConfig) error {
	unwritable := UnwritableDirs(cfg)
	if len(unwritable) == 0 {
		return nil
	}
	messages := make([]string, len(unwritable))
	for i, dir := range unwritable {
		messages[i] = dir.Error()
	}
	return errors.New(strings.Join(messages, "; "))
}

func checkWritableDir(path string) error {
	dir := filepath.Clean(path)
	for {
		info, err := os.Stat(dir)
		if err == nil {
			if !info.IsDir() {
				return fmt.Errorf("%s is not a directory", dir)
			}
			break
		}
		parent := filepath.Dir(dir)
		if !errors.Is(err, fs.ErrNotExist) || parent == dir {
			return err
		}
		dir = parent
	}
	probe, err := os.CreateTemp(dir, ".vllm-sr-write-check-*")
	if err != nil {
		var pathErr *fs.PathError
		if errors.As(err, &pathErr) {
			return fmt.Errorf("%s: %w", dir, pathErr.Err)
		}
		return err
	}
	name := probe.Name()
	return errors.Join(probe.Close(), os.Remove(name))
}
