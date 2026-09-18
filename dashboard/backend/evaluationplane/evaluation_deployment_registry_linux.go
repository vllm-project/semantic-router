//go:build linux

package evaluationplane

import "golang.org/x/sys/unix"

// O_PATH allows traversal without requiring directory read permission.
const deploymentRegistryDirectoryFlags = unix.O_PATH | unix.O_DIRECTORY | unix.O_CLOEXEC
