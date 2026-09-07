//go:build linux

package evaluationplane

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/sys/unix"
)

// deploymentRegistryRoot pins the validated directory by descriptor. Every
// descendant is opened relative to that descriptor with O_NOFOLLOW, so a
// rename or symlink substitution cannot redirect a later registry read.
type deploymentRegistryRoot struct {
	fd int
}

func openDeploymentRegistryRoot(path string) (*deploymentRegistryRoot, error) {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return nil, fmt.Errorf("deployment registry root must be an absolute canonical path")
	}
	current, err := unix.Open("/", unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	for _, component := range strings.Split(strings.TrimPrefix(path, "/"), "/") {
		if component == "" {
			continue
		}
		next, openErr := unix.Openat(
			current,
			component,
			unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			0,
		)
		_ = unix.Close(current)
		if openErr != nil {
			return nil, fmt.Errorf("symlink or non-directory path component is not allowed: %w", openErr)
		}
		current = next
	}
	return &deploymentRegistryRoot{fd: current}, nil
}

func (root *deploymentRegistryRoot) Close() {
	if root != nil && root.fd >= 0 {
		_ = unix.Close(root.fd)
		root.fd = -1
	}
}

func (root *deploymentRegistryRoot) ReadFile(relative string, limit int64) ([]byte, error) {
	if root == nil || root.fd < 0 {
		return nil, fmt.Errorf("deployment registry root is closed")
	}
	if err := validateRelativeDeploymentConfigPath(relative); err != nil {
		return nil, err
	}
	current, duplicateErr := unix.Dup(root.fd)
	if duplicateErr != nil {
		return nil, duplicateErr
	}
	components := strings.Split(relative, "/")
	for _, component := range components[:len(components)-1] {
		next, openErr := unix.Openat(
			current,
			component,
			unix.O_PATH|unix.O_DIRECTORY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
			0,
		)
		_ = unix.Close(current)
		if openErr != nil {
			return nil, fmt.Errorf("symlink or non-directory path component is not allowed: %w", openErr)
		}
		current = next
	}
	fd, openErr := unix.Openat(
		current,
		components[len(components)-1],
		unix.O_RDONLY|unix.O_CLOEXEC|unix.O_NOFOLLOW,
		0,
	)
	_ = unix.Close(current)
	if openErr != nil {
		return nil, fmt.Errorf("symlink or unreadable deployment registry file is not allowed: %w", openErr)
	}
	file := os.NewFile(uintptr(fd), relative)
	if file == nil {
		_ = unix.Close(fd)
		return nil, fmt.Errorf("open deployment registry file descriptor")
	}
	defer file.Close()
	var before unix.Stat_t
	if err := unix.Fstat(fd, &before); err != nil {
		return nil, err
	}
	if before.Mode&unix.S_IFMT != unix.S_IFREG {
		return nil, fmt.Errorf("deployment registry path is not a regular file")
	}
	if before.Size < 0 || before.Size > limit {
		return nil, fmt.Errorf("file exceeds %d bytes", limit)
	}
	data, readErr := io.ReadAll(io.LimitReader(file, limit+1))
	if readErr != nil {
		return nil, readErr
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("file exceeds %d bytes", limit)
	}
	var after unix.Stat_t
	if err := unix.Fstat(fd, &after); err != nil {
		return nil, err
	}
	if !sameDeploymentFileState(before, after) || int64(len(data)) != after.Size {
		return nil, fmt.Errorf("deployment registry file changed while it was read")
	}
	return data, nil
}

func sameDeploymentFileState(left, right unix.Stat_t) bool {
	return left.Dev == right.Dev && left.Ino == right.Ino && left.Mode == right.Mode &&
		left.Size == right.Size && left.Mtim == right.Mtim && left.Ctim == right.Ctim
}
