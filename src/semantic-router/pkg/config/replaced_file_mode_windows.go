package config

import "os"

// KeepReplacedFileMode does nothing on Windows, which has no POSIX file modes.
func KeepReplacedFileMode(*os.File, string) error { return nil }
