// Package setupmode owns the single canonical answer to "is the dashboard in
// first-run setup mode?".
//
// Two inputs used to answer this independently: the `setup.mode` block in the
// router config file, and the --setup-mode / DASHBOARD_SETUP_MODE process flag.
// They could disagree, because the config block is re-read from disk while the
// flag is frozen at process start, and activation rewrites the config without
// restarting the dashboard. See #2795.
//
// The config file is canonical. The legacy flag is read only so a disagreement
// can be reported; it never decides.
package setupmode

import (
	"fmt"
	"log"
	"os"
	"sync"

	"gopkg.in/yaml.v3"
)

// Source identifies which input produced an active setup state.
type Source string

const (
	// SourceConfig means an active setup.mode block in the router config file
	// produced the answer. It is the only source that can produce Active.
	SourceConfig Source = "config"

	// SourceNone means setup mode is not active.
	SourceNone Source = "none"
)

// Resolution is the resolved setup state plus enough context to explain it.
type Resolution struct {
	// Active reports whether first-run setup mode is in effect. Every setup
	// decision in the dashboard must derive from this field.
	Active bool

	// Source names the input that produced Active. SourceNone when inactive.
	Source Source

	// LegacyFlag records the --setup-mode / DASHBOARD_SETUP_MODE value. It is
	// informational: it is reported and warned about, never merged into Active.
	LegacyFlag bool

	// Conflict is true when the legacy flag disagrees with the config file.
	// Callers surface this; it is the signal that a deployment carries a stale
	// environment value.
	Conflict bool

	// Reason explains a non-obvious resolution: an unreadable or unparseable
	// config, or a conflict. Empty when the answer was clean. Safe to show to
	// an operator; it never contains config contents.
	Reason string
}

// setupBlock is the only part of the router config this package cares about.
// Decoding into a minimal struct keeps setupmode independent of the canonical
// config schema, so a schema change cannot break the bootstrap gate.
type setupBlock struct {
	Setup *struct {
		Mode bool `yaml:"mode"`
	} `yaml:"setup"`
}

// cacheKey is the file identity we trust to detect a change without re-reading.
//
// The modification time is held as Unix nanoseconds rather than a time.Time so
// the struct stays comparable with ==; time.Time carries a location pointer and
// an optional monotonic reading, neither of which belongs in an equality test.
type cacheKey struct {
	modTimeUnixNano int64
	size            int64
}

func statKey(info os.FileInfo) cacheKey {
	return cacheKey{modTimeUnixNano: info.ModTime().UnixNano(), size: info.Size()}
}

// Resolver answers setup-mode questions from the config file on disk.
//
// Safe for concurrent use: net/http serves each request in its own goroutine and
// the auth middleware consults this on every request.
type Resolver struct {
	configPath string
	legacyFlag bool

	mu       sync.RWMutex
	cached   Resolution
	cachedAt cacheKey
	valid    bool

	// generation counts invalidations. A resolve that reads the file records the
	// generation it started from and refuses to publish its result if the count
	// moved while it was reading. See Invalidate.
	generation uint64

	// lastActive/lastConflict/haveLast record the most recently *observed*
	// (Active, Conflict) pair, across every fresh read - successful or failed -
	// not just the ones that got cached. logTransition uses them to decide
	// whether a resolution is new enough to be worth a log line. See
	// logTransition.
	lastActive   bool
	lastConflict bool
	haveLast     bool
}

// New returns a Resolver over the router config at configPath. legacyFlag is the
// --setup-mode / DASHBOARD_SETUP_MODE value, recorded for conflict reporting
// only.
func New(configPath string, legacyFlag bool) *Resolver {
	return &Resolver{configPath: configPath, legacyFlag: legacyFlag}
}

// Active is the hot path: the boolean every gate needs.
func (r *Resolver) Active() bool { return r.Resolve().Active }

// Invalidate drops the cache. Call it immediately after writing the config file
// so a resolve in the same second cannot serve a stale answer through the
// mtime cache.
func (r *Resolver) Invalidate() {
	r.mu.Lock()
	r.valid = false
	r.generation++
	r.mu.Unlock()
}

// Resolve reports the current setup state, reading the config file only when its
// identity on disk has changed since the last successful read.
//
// The stat is what makes this safe to call from an unauthenticated endpoint: an
// uncached implementation would let anyone who can reach the port force a YAML
// parse per request.
func (r *Resolver) Resolve() Resolution {
	// Stat before reading, never after. If the file changes in between, the
	// parsed content gets stored under the previous identity, which costs one
	// extra read on the next call and nothing else. The reverse order could
	// store pre-write content under the post-write identity and keep serving it
	// until the file changed again.
	info, err := os.Stat(r.configPath)
	if err != nil {
		return r.failClosed(fmt.Sprintf("config file %s is unreadable (%v)", r.configPath, err))
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
		return r.failClosed(fmt.Sprintf("config file %s is unreadable (%v)", r.configPath, err))
	}

	var block setupBlock
	if err := yaml.Unmarshal(data, &block); err != nil {
		// The parser error is deliberately dropped. gopkg.in/yaml.v3 quotes the
		// offending value in its messages ("cannot unmarshal !!str `notabool`
		// into bool"), and Reason is served from an unauthenticated endpoint, so
		// echoing it would leak config contents to anyone who can reach the port.
		return r.failClosed(fmt.Sprintf("config file %s is not valid YAML", r.configPath))
	}

	resolution := r.resolve(block.Setup != nil && block.Setup.Mode)
	r.logTransition(resolution)

	r.mu.Lock()
	// Drop the result rather than cache it if Invalidate ran while we were
	// reading: the bytes we parsed may pre-date the write that prompted it, and
	// on a filesystem with coarse mtime granularity the identity check above
	// cannot tell. Returning it is still correct -- it was true of some state of
	// the file -- but caching it would make the next caller see it too.
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

// conflictReason names both inputs and states which one won, because this
// message is the entire operator-facing value of detecting the conflict.
func (r *Resolver) conflictReason(active bool) string {
	if active {
		return fmt.Sprintf(
			"DASHBOARD_SETUP_MODE / --setup-mode is false or unset, but the config at %s declares an active setup.mode block; "+
				"the config file is canonical, so setup mode is ON and first-run registration is available. "+
				"Completing setup clears the block and closes it.",
			r.configPath)
	}
	return fmt.Sprintf(
		"DASHBOARD_SETUP_MODE / --setup-mode is set to true, but the config at %s has no active setup.mode block; "+
			"the config file is canonical, so setup mode is OFF and the open bootstrap endpoint is closed. "+
			"Remove the stale DASHBOARD_SETUP_MODE value.",
		r.configPath)
}

// failClosed is the answer whenever the config cannot be read or parsed: never
// active, and never derived from the legacy flag.
//
// Falling back to the flag here would reintroduce #2795 through the error path.
// "I could not read the config, so I will assume this is the trusted first-run
// window" is the wrong way for a gate on unauthenticated admin creation to fail.
//
// Error results are never cached: the file may become readable a moment later,
// and a cached error would hold the dashboard out of its own first run.
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

// logTransition emits at most one WARNING per distinct (Active, Conflict)
// pair, so a conflicting deployment is reported when it first appears and
// again if the resolution later changes, but an unauthenticated caller cannot
// drive log volume by polling can-register: repeated calls that keep
// resolving the same way log nothing after the first.
//
// This is a security property, not tidiness. It is only reached from the
// paths that just did real work - a fresh parse or a stat/read/parse failure -
// never from the mtime-cache hit in Resolve, so a stable config costs one log
// check total, not one per request.
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
