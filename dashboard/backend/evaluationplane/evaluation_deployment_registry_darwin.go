//go:build darwin

package evaluationplane

import "golang.org/x/sys/unix"

// Darwin has openat and O_NOFOLLOW but no O_PATH. Readable directory
// descriptors retain the same pinned-root semantics; directories must be readable.
const deploymentRegistryDirectoryFlags = unix.O_RDONLY | unix.O_DIRECTORY | unix.O_CLOEXEC
