// Package setupmode answers "is the dashboard in first-run setup mode?".
//
// Setup mode used to have two independent sources: the setup.mode block in the
// router config, and the --setup-mode / DASHBOARD_SETUP_MODE flag. They drifted,
// because the config is re-read per request while the flag is frozen at startup,
// and activation rewrites the config without restarting the dashboard. See #2795.
//
// The config file decides. The legacy flag is read only so a disagreement can be
// reported.
package setupmode

import (
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

// Source identifies which input produced an active setup state.
type Source string

const (
	// SourceConfig means the config file's setup.mode block is active.
	SourceConfig Source = "config"

	// SourceNone means setup mode is off.
	SourceNone Source = "none"
)

// Resolution is the resolved setup state plus context to explain it.
type Resolution struct {
	// Active reports whether setup mode is in effect. Every setup decision in
	// the dashboard must derive from this field.
	Active bool

	// Source names the input that produced Active.
	Source Source

	// LegacyFlag records the --setup-mode / DASHBOARD_SETUP_MODE value. It is
	// reported and warned about, never merged into Active.
	LegacyFlag bool

	// Conflict is true when the legacy flag disagrees with the config file.
	Conflict bool

	// Reason explains an unreadable config or a conflict. Empty when the answer
	// was clean. It is served from a public endpoint, so it never carries config
	// contents or the absolute config path.
	Reason string
}

// setupBlock is the only part of the router config this package decodes.
// Staying independent of the full config schema means a schema change cannot
// break the bootstrap gate.
type setupBlock struct {
	Setup *struct {
		Mode bool `yaml:"mode"`
	} `yaml:"setup"`
}

// cacheKey identifies the config file well enough to skip a re-read. Unix nanos
// rather than a time.Time so the struct stays comparable with ==.
type cacheKey struct {
	modTimeUnixNano int64
	size            int64
}

func statKey(info os.FileInfo) cacheKey {
	return cacheKey{modTimeUnixNano: info.ModTime().UnixNano(), size: info.Size()}
}

// Resolver answers setup-mode questions from the config file on disk.
// Safe for concurrent use: the auth middleware consults it on every request.
type Resolver struct {
	configPath string
	// configName is the base name, used in Reason. The full path would leak the
	// deployment's directory layout through an unauthenticated endpoint. It is
	// in the startup log instead.
	configName string
	legacyFlag bool

	mu       sync.RWMutex
	cached   Resolution
	cachedAt cacheKey
	valid    bool

	// generation counts invalidations. A read that started before an Invalidate
	// must not publish its result. See Resolve.
	generation uint64

	// Last observed state, used by logTransition to log only on a change.
	lastActive   bool
	lastConflict bool
	haveLast     bool
}

// New returns a Resolver over the router config at configPath. legacyFlag is the
// --setup-mode / DASHBOARD_SETUP_MODE value, recorded for reporting only.
func New(configPath string, legacyFlag bool) *Resolver {
	return &Resolver{
		configPath: configPath,
		configName: filepath.Base(configPath),
		legacyFlag: legacyFlag,
	}
}

// errDetail returns the cause of a filesystem error without the path. os.Stat
// and os.ReadFile return *fs.PathError, whose Error() embeds the absolute path,
// and Reason is served publicly.
func errDetail(err error) string {
	var pathErr *fs.PathError
	if errors.As(err, &pathErr) {
		if pathErr.Err == nil {
			// PathError.Error() dereferences Err, so falling through would panic.
			return "unknown filesystem error"
		}
		return pathErr.Err.Error()
	}
	return err.Error()
}

// Active reports whether setup mode is on. This is what the gates call.
func (r *Resolver) Active() bool { return r.Resolve().Active }

// Invalidate drops the cache. Call it after writing the config file, so a
// resolve in the same second cannot serve a stale answer through the mtime cache.
func (r *Resolver) Invalidate() {
	r.mu.Lock()
	r.valid = false
	r.generation++
	r.mu.Unlock()
}

// Resolve reports the current setup state, re-reading the config file only when
// its identity on disk has changed.
//
// The cache matters because this is reached from an unauthenticated endpoint:
// without it, anyone could force a YAML parse per request.
func (r *Resolver) Resolve() Resolution {
	// Stat before reading. If the file changes in between, we cache fresh
	// content under the old identity and pay one extra read next call. The
	// reverse order could cache stale content under the new identity and keep
	// serving it.
	info, err := os.Stat(r.configPath)
	if err != nil {
		return r.failClosed(fmt.Sprintf("config file %s is unreadable (%s)", r.configName, errDetail(err)))
	}
	key := statKey(info)

	r.mu.RLock()
	if r.valid && r.cachedAt == key {
		cached := r.cached
		r.mu.RUnlock()
		return cached
	}
	generation := r.generation
	r.mu.RUnlock()

	data, err := os.ReadFile(r.configPath)
	if err != nil {
		return r.failClosed(fmt.Sprintf("config file %s is unreadable (%s)", r.configName, errDetail(err)))
	}

	var block setupBlock
	if err := yaml.Unmarshal(data, &block); err != nil {
		// The parser error is dropped on purpose: yaml.v3 quotes the offending
		// value in its messages, and Reason is served publicly.
		return r.failClosed(fmt.Sprintf("config file %s is not valid YAML", r.configName))
	}

	resolution := r.resolve(block.Setup != nil && block.Setup.Mode)
	r.logTransition(resolution)

	r.mu.Lock()
	// Skip the store if Invalidate ran while we were reading. The bytes may
	// pre-date the write that triggered it, and the identity check above cannot
	// tell when mtime granularity is coarse.
	if r.generation == generation {
		r.cached = resolution
		r.cachedAt = key
		r.valid = true
	}
	r.mu.Unlock()

	return resolution
}

// resolve builds the answer for a config file that was read and parsed cleanly.
func (r *Resolver) resolve(active bool) Resolution {
	resolution := Resolution{
		Active:     active,
		Source:     SourceNone,
		LegacyFlag: r.legacyFlag,
		Conflict:   r.legacyFlag != active,
	}
	if active {
		resolution.Source = SourceConfig
	}
	if resolution.Conflict {
		resolution.Reason = r.conflictReason(active)
	}
	return resolution
}

// conflictReason names both inputs and says which one won. This message is the
// whole operator-facing value of detecting the conflict.
func (r *Resolver) conflictReason(active bool) string {
	if active {
		return fmt.Sprintf(
			"DASHBOARD_SETUP_MODE / --setup-mode is false or unset, but the config at %s declares an active setup.mode block; "+
				"the config file is canonical, so setup mode is ON and first-run registration is available. "+
				"Completing setup clears the block and closes it.",
			r.configName)
	}
	return fmt.Sprintf(
		"DASHBOARD_SETUP_MODE / --setup-mode is set to true, but the config at %s has no active setup.mode block; "+
			"the config file is canonical, so setup mode is OFF and the open bootstrap endpoint is closed. "+
			"Remove the stale DASHBOARD_SETUP_MODE value.",
		r.configName)
}

// failClosed is the answer when the config cannot be read or parsed: never
// active, never derived from the legacy flag.
//
// Falling back to the flag here would reintroduce #2795 through the error path.
// The result is not cached, because the file may become readable a moment later.
func (r *Resolver) failClosed(problem string) Resolution {
	reason := problem + "; setup mode is OFF and the open bootstrap endpoint stays closed."
	if r.legacyFlag {
		reason += " DASHBOARD_SETUP_MODE / --setup-mode is set to true but is not consulted: the config file is the only source."
	}
	resolution := Resolution{
		Active:     false,
		Source:     SourceNone,
		LegacyFlag: r.legacyFlag,
		Conflict:   r.legacyFlag,
		Reason:     reason,
	}
	r.logTransition(resolution)
	return resolution
}

// logTransition logs at most one WARNING per distinct (Active, Conflict) pair.
//
// Rate limiting is a security property, not tidiness: can-register is
// unauthenticated, so warning on every resolve would turn it into a
// log-amplification endpoint. Never called from the cache-hit path.
func (r *Resolver) logTransition(resolution Resolution) {
	r.mu.Lock()
	changed := !r.haveLast || r.lastActive != resolution.Active || r.lastConflict != resolution.Conflict
	r.lastActive = resolution.Active
	r.lastConflict = resolution.Conflict
	r.haveLast = true
	r.mu.Unlock()

	if changed && resolution.Conflict {
		log.Printf("WARNING: %s", resolution.Reason)
	}
}
