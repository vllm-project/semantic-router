package setupmode

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

const (
	// configSetupModeOn is the shape the CLI writes on first run.
	configSetupModeOn = `version: v0.3
listeners:
  - name: http-8801
    address: 0.0.0.0
    port: 8801
    timeout: 300s
setup:
  mode: true
  state: bootstrap
`

	// configActivated is the same file after activation strips the setup block,
	// which is what handlers.SetupActivateHandler writes.
	configActivated = `version: v0.3
listeners:
  - name: http-8801
    address: 0.0.0.0
    port: 8801
    timeout: 300s
`

	configSetupModeExplicitlyOff = `version: v0.3
setup:
  mode: false
`

	configMalformed = "::: not yaml :::\n"
)

// These three fixtures are the same length on purpose. Writing one over another
// and restoring the mtime produces a file the cache cannot tell from the
// original, which is how the tests below prove the cache is real without
// depending on filesystem timestamp granularity. TestFixtures_SameSize keeps
// them honest.
const (
	configModeOnSameSize    = "setup:\n  mode: true\n\n"
	configModeOffSameSize   = "setup:\n  mode: false\n"
	configMalformedSameSize = "::: not yaml :::\n\n\n\n\n"
)

func writeConfig(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write config %s: %v", path, err)
	}
}

// newConfigFile writes content to a fresh temp file and returns its path.
func newConfigFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	writeConfig(t, path, content)
	return path
}

// freezeMTime rewrites the file but restores its previous mtime, leaving the
// cache key unchanged.
func freezeMTime(t *testing.T, path, content string) {
	t.Helper()
	before, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	writeConfig(t, path, content)
	if err = os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
		t.Fatalf("chtimes %s: %v", path, err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatalf("re-stat %s: %v", path, err)
	}
	if statKey(after) != statKey(before) {
		t.Fatalf("file identity changed despite same size and restored mtime: %+v -> %+v", statKey(before), statKey(after))
	}
}

// The most important test here: it proves the error path cannot reopen the
// unauthenticated bootstrap door. An unparsable config resolves to "not in
// setup mode" even when the legacy flag says true.
func TestResolve_MalformedConfigDoesNotFallBackToLegacyFlag(t *testing.T) {
	path := newConfigFile(t, configMalformed)
	resolver := New(path, true)

	got := resolver.Resolve()

	if got.Active {
		t.Fatalf("Active = true for a malformed config with the legacy flag set; the error path must fail closed")
	}
	// Active() is what the bootstrap gate calls, so assert it directly.
	if resolver.Active() {
		t.Fatalf("Active() = true, want false")
	}
	if got.Source != SourceNone {
		t.Fatalf("Source = %q, want %q", got.Source, SourceNone)
	}
	if !got.LegacyFlag {
		t.Fatalf("LegacyFlag = false, want the flag recorded as true")
	}
	if !got.Conflict {
		t.Fatalf("Conflict = false, want true: the flag says on and the resolved state is off")
	}
	if got.Reason == "" {
		t.Fatalf("Reason is empty, want an explanation of the unparsable config")
	}
}

// Reason is served from a public endpoint, so it must never echo config
// contents. yaml.v3 quotes the offending value in its errors, which is why the
// parser message is dropped.
func TestResolve_ReasonNeverContainsConfigContents(t *testing.T) {
	const secret = "sekret"
	path := newConfigFile(t, "setup:\n  mode: "+secret+"\n")
	resolver := New(path, true)

	got := resolver.Resolve()

	if got.Active {
		t.Fatalf("Active = true for a config whose setup.mode is not a boolean")
	}
	if got.Reason == "" {
		t.Fatalf("Reason is empty, want an explanation of the unparsable config")
	}
	if strings.Contains(got.Reason, secret) {
		t.Fatalf("Reason leaks config contents: %q", got.Reason)
	}
}

func TestResolve(t *testing.T) {
	tests := []struct {
		name         string
		config       string
		legacyFlag   bool
		wantActive   bool
		wantSource   Source
		wantConflict bool
		wantReason   bool
	}{
		{
			name:         "config only: the file declares setup mode, the flag does not",
			config:       configSetupModeOn,
			legacyFlag:   false,
			wantActive:   true,
			wantSource:   SourceConfig,
			wantConflict: true,
			wantReason:   true,
		},
		{
			name:         "legacy flag only: the flag declares setup mode, the file does not",
			config:       configActivated,
			legacyFlag:   true,
			wantActive:   false,
			wantSource:   SourceNone,
			wantConflict: true,
			wantReason:   true,
		},
		{
			name:         "both agree, on",
			config:       configSetupModeOn,
			legacyFlag:   true,
			wantActive:   true,
			wantSource:   SourceConfig,
			wantConflict: false,
			wantReason:   false,
		},
		{
			name:         "both agree, off",
			config:       configActivated,
			legacyFlag:   false,
			wantActive:   false,
			wantSource:   SourceNone,
			wantConflict: false,
			wantReason:   false,
		},
		{
			name:         "setup.mode is explicitly false",
			config:       configSetupModeExplicitlyOff,
			legacyFlag:   false,
			wantActive:   false,
			wantSource:   SourceNone,
			wantConflict: false,
			wantReason:   false,
		},
		{
			name:         "setup.mode is explicitly false and the flag disagrees",
			config:       configSetupModeExplicitlyOff,
			legacyFlag:   true,
			wantActive:   false,
			wantSource:   SourceNone,
			wantConflict: true,
			wantReason:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resolver := New(newConfigFile(t, tt.config), tt.legacyFlag)

			got := resolver.Resolve()

			if got.Active != tt.wantActive {
				t.Fatalf("Active = %v, want %v", got.Active, tt.wantActive)
			}
			if got.Source != tt.wantSource {
				t.Fatalf("Source = %q, want %q", got.Source, tt.wantSource)
			}
			if got.LegacyFlag != tt.legacyFlag {
				t.Fatalf("LegacyFlag = %v, want %v", got.LegacyFlag, tt.legacyFlag)
			}
			if got.Conflict != tt.wantConflict {
				t.Fatalf("Conflict = %v, want %v", got.Conflict, tt.wantConflict)
			}
			if (got.Reason != "") != tt.wantReason {
				t.Fatalf("Reason = %q, want non-empty: %v", got.Reason, tt.wantReason)
			}
			if active := resolver.Active(); active != tt.wantActive {
				t.Fatalf("Active() = %v, want %v", active, tt.wantActive)
			}
		})
	}
}

// A conflict message has to be actionable, so it names both inputs and says
// which won. Assert the key parts, not the whole string.
func TestResolve_ConflictReasonNamesBothInputs(t *testing.T) {
	tests := []struct {
		name       string
		config     string
		legacyFlag bool
		wantParts  []string
	}{
		{
			name:       "stale flag against an activated config",
			config:     configActivated,
			legacyFlag: true,
			wantParts:  []string{"DASHBOARD_SETUP_MODE", "--setup-mode", "setup.mode", "canonical", "OFF"},
		},
		{
			name:       "config in setup mode with no flag",
			config:     configSetupModeOn,
			legacyFlag: false,
			wantParts:  []string{"DASHBOARD_SETUP_MODE", "--setup-mode", "setup.mode", "canonical", "ON"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := newConfigFile(t, tt.config)
			got := New(path, tt.legacyFlag).Resolve()

			if !got.Conflict {
				t.Fatalf("Conflict = false, want true")
			}
			for _, part := range tt.wantParts {
				if !strings.Contains(got.Reason, part) {
					t.Fatalf("Reason %q does not mention %q", got.Reason, part)
				}
			}
			if !strings.Contains(got.Reason, filepath.Base(path)) {
				t.Fatalf("Reason %q does not name the config file %q", got.Reason, filepath.Base(path))
			}
		})
	}
}

// Activation strips the setup block and does not restart the dashboard, so the
// resolver must see it on the next call. The legacy flag stays true throughout,
// as it does in a real deployment.
func TestResolve_ActivationTurnsSetupModeOffWithoutRestart(t *testing.T) {
	path := newConfigFile(t, configSetupModeOn)
	resolver := New(path, true)

	if before := resolver.Resolve(); !before.Active {
		t.Fatalf("Active = false before activation, want true (%+v)", before)
	}

	writeConfig(t, path, configActivated)
	resolver.Invalidate()

	after := resolver.Resolve()
	if after.Active {
		t.Fatalf("Active = true after activation stripped the setup block, want false")
	}
	if after.Source != SourceNone {
		t.Fatalf("Source = %q, want %q", after.Source, SourceNone)
	}
	if !after.Conflict {
		t.Fatalf("Conflict = false, want true: the flag is still true and the config is now off")
	}
}

func TestResolve_MissingConfigFileFailsClosed(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.yaml")

	got := New(path, false).Resolve()

	if got.Active {
		t.Fatalf("Active = true for a missing config file, want false")
	}
	if got.Source != SourceNone {
		t.Fatalf("Source = %q, want %q", got.Source, SourceNone)
	}
	if !strings.Contains(got.Reason, "unreadable") {
		t.Fatalf("Reason = %q, want it to mention the file being unreadable", got.Reason)
	}
}

func TestResolve_UnreadableConfigFileFailsClosed(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("running as root: chmod 000 does not deny access")
	}
	path := newConfigFile(t, configSetupModeOn)
	if err := os.Chmod(path, 0o000); err != nil {
		t.Fatalf("chmod %s: %v", path, err)
	}
	t.Cleanup(func() { _ = os.Chmod(path, 0o600) })

	got := New(path, true).Resolve()

	if got.Active {
		t.Fatalf("Active = true for an unreadable config file with the legacy flag set, want false")
	}
	if !got.Conflict {
		t.Fatalf("Conflict = false, want true")
	}
	if !strings.Contains(got.Reason, "unreadable") {
		t.Fatalf("Reason = %q, want it to mention the file being unreadable", got.Reason)
	}
}

func TestResolve_RepeatedResolveIsStable(t *testing.T) {
	resolver := New(newConfigFile(t, configSetupModeOn), false)

	first := resolver.Resolve()
	second := resolver.Resolve()

	if first != second {
		t.Fatalf("two resolves of an unchanged file disagree:\n first  = %+v\n second = %+v", first, second)
	}
	if !first.Active {
		t.Fatalf("Active = false, want true")
	}
}

// Proves the cache is really consulted: the contents change while the identity
// does not, and the previous answer is served.
func TestResolve_CacheServesUnchangedFileIdentity(t *testing.T) {
	path := newConfigFile(t, configModeOnSameSize)
	resolver := New(path, false)

	if !resolver.Resolve().Active {
		t.Fatalf("Active = false for the initial config, want true")
	}

	freezeMTime(t, path, configModeOffSameSize)

	if !resolver.Resolve().Active {
		t.Fatalf("Active = false after a change the file identity cannot show; the mtime+size cache was not used")
	}
}

// The case mtime cannot cover: activation writes the config and answers the
// request within the same timestamp tick. Invalidate is what makes that safe.
func TestResolve_InvalidateForcesReread(t *testing.T) {
	path := newConfigFile(t, configModeOnSameSize)
	resolver := New(path, false)

	if !resolver.Resolve().Active {
		t.Fatalf("Active = false for the initial config, want true")
	}

	freezeMTime(t, path, configModeOffSameSize)
	resolver.Invalidate()

	if resolver.Resolve().Active {
		t.Fatalf("Active = true after Invalidate; the cache was not dropped")
	}
}

// Errors are never cached: a missing file may appear a moment later, and a
// cached error would lock the dashboard out of its own first run. The second
// case keeps the file identity unchanged, so a cached error could not hide
// behind a changed cache key.
func TestResolve_ErrorResultsAreNeverCached(t *testing.T) {
	t.Run("missing file that appears later", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "config.yaml")
		// legacyFlag=true so the config agrees with it once it appears.
		resolver := New(path, true)

		if got := resolver.Resolve(); got.Active || got.Reason == "" {
			t.Fatalf("Resolve() = %+v, want inactive with a reason", got)
		}

		writeConfig(t, path, configSetupModeOn)

		got := resolver.Resolve()
		if !got.Active {
			t.Fatalf("Active = false after the config appeared; the error result was cached")
		}
		if got.Reason != "" {
			t.Fatalf("Reason = %q, want empty once the config resolved cleanly", got.Reason)
		}
	})

	t.Run("malformed file repaired without changing its identity", func(t *testing.T) {
		path := newConfigFile(t, configMalformedSameSize)
		resolver := New(path, false)

		if got := resolver.Resolve(); got.Active || got.Reason == "" {
			t.Fatalf("Resolve() = %+v, want inactive with a reason", got)
		}

		freezeMTime(t, path, configModeOnSameSize)

		if !resolver.Resolve().Active {
			t.Fatalf("Active = false after the config was repaired; the error result was cached under the file identity")
		}
	})
}

// The cache tests only prove anything while these stay the same length.
func TestFixtures_SameSize(t *testing.T) {
	sizes := map[string]int{
		"configModeOnSameSize":    len(configModeOnSameSize),
		"configModeOffSameSize":   len(configModeOffSameSize),
		"configMalformedSameSize": len(configMalformedSameSize),
	}
	for name, size := range sizes {
		if size != len(configModeOnSameSize) {
			t.Fatalf("%s is %d bytes, want %d: the cache tests need identical sizes (%v)", name, size, len(configModeOnSameSize), sizes)
		}
	}
}

// The resolver is read on every request while activation may be rewriting the
// config. This asserts race-freedom and self-consistency, not a particular
// value: os.WriteFile truncates first, so a reader can observe a partial file.
func TestResolve_ConcurrentAccessIsRaceFree(t *testing.T) {
	path := newConfigFile(t, configSetupModeOn)
	resolver := New(path, true)

	const (
		readers           = 100
		resolvesPerReader = 50
	)

	stop := make(chan struct{})
	var writer, readerGroup sync.WaitGroup

	writer.Add(1)
	go func() {
		defer writer.Done()
		for i := 0; ; i++ {
			select {
			case <-stop:
				return
			default:
			}
			content := configSetupModeOn
			if i%2 == 1 {
				content = configActivated
			}
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Errorf("write config: %v", err)
				return
			}
			resolver.Invalidate()
		}
	}()

	for i := 0; i < readers; i++ {
		readerGroup.Add(1)
		go func() {
			defer readerGroup.Done()
			for j := 0; j < resolvesPerReader; j++ {
				got := resolver.Resolve()
				if got.Active && got.Source != SourceConfig {
					t.Errorf("Active resolution has Source %q, want %q", got.Source, SourceConfig)
				}
				if !got.Active && got.Source != SourceNone {
					t.Errorf("inactive resolution has Source %q, want %q", got.Source, SourceNone)
				}
				if !got.LegacyFlag {
					t.Errorf("LegacyFlag = false, want the constructor value true")
				}
				_ = resolver.Active()
			}
		}()
	}

	readerGroup.Wait()
	close(stop)
	writer.Wait()
}

// --- Conflict warning logging (#2795) ---------------------------------------
//
// Resolve() runs on every request, including from the unauthenticated
// can-register. A conflict must warn when it first appears, but repeating an
// unchanged resolution must not warn again, or the endpoint becomes a
// log-amplification vector.

// captureLog redirects the standard logger into a buffer and restores it on
// cleanup. No test here uses t.Parallel(), so touching shared logger state is safe.
func captureLog(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	origOutput := log.Writer()
	origFlags := log.Flags()
	log.SetOutput(&buf)
	log.SetFlags(0)
	t.Cleanup(func() {
		log.SetOutput(origOutput)
		log.SetFlags(origFlags)
	})
	return &buf
}

// countWarnings counts the WARNING lines the resolver logged.
func countWarnings(buf *bytes.Buffer) int {
	count := 0
	for _, line := range strings.Split(buf.String(), "\n") {
		if strings.HasPrefix(line, "WARNING:") {
			count++
		}
	}
	return count
}

// Baseline for the next two tests: one WARNING naming the stale input.
func TestResolve_LogsConflictWarningOnFirstResolve(t *testing.T) {
	buf := captureLog(t)
	path := newConfigFile(t, configActivated)

	got := New(path, true).Resolve()

	if !got.Conflict {
		t.Fatalf("Conflict = false, want true")
	}
	if n := countWarnings(buf); n != 1 {
		t.Fatalf("logged %d WARNING lines, want 1; log:\n%s", n, buf.String())
	}
	if !strings.Contains(buf.String(), "DASHBOARD_SETUP_MODE") {
		t.Fatalf("log does not name the stale input: %s", buf.String())
	}
}

// Hammering an unauthenticated endpoint against a stable conflicting config
// must not multiply the log line. This is the security property.
func TestResolve_ConflictWarningLogsOnceForRepeatedUnchangedConflict(t *testing.T) {
	buf := captureLog(t)
	path := newConfigFile(t, configActivated)
	resolver := New(path, true)

	const requests = 20
	for i := 0; i < requests; i++ {
		resolver.Resolve()
	}

	if n := countWarnings(buf); n != 1 {
		t.Fatalf("logged %d WARNING lines across %d identical resolves, want 1; log:\n%s", n, requests, buf.String())
	}
}

// The dedup key is the last observed (Active, Conflict) pair, not full history.
// A conflict that clears and later returns must warn again, so silence cannot
// be misread as "still fine". Transitions that stay clean log nothing.
func TestResolve_ConflictWarningLogsAgainAfterReturningFromClean(t *testing.T) {
	buf := captureLog(t)
	path := newConfigFile(t, configActivated)
	// legacyFlag stays true throughout, as it does across a real activation.
	resolver := New(path, true)

	// Step 1: config off, flag on -> conflict. First WARNING.
	if got := resolver.Resolve(); !got.Conflict {
		t.Fatalf("step 1: Conflict = false, want true")
	}
	if n := countWarnings(buf); n != 1 {
		t.Fatalf("step 1: logged %d WARNING lines, want 1; log:\n%s", n, buf.String())
	}

	// Step 2: config on, flag on. They agree, so no new WARNING.
	writeConfig(t, path, configSetupModeOn)
	resolver.Invalidate()
	if got := resolver.Resolve(); got.Conflict {
		t.Fatalf("step 2: Conflict = true, want false (config and flag agree)")
	}
	if n := countWarnings(buf); n != 1 {
		t.Fatalf("step 2: logged %d WARNING lines, want still 1 (nothing to report); log:\n%s", n, buf.String())
	}

	// Step 3: back to step 1's conflicting shape. Step 2 changed the last
	// observed pair, so this is a new transition and must warn again.
	writeConfig(t, path, configActivated)
	resolver.Invalidate()
	if got := resolver.Resolve(); !got.Conflict {
		t.Fatalf("step 3: Conflict = false, want true")
	}
	if n := countWarnings(buf); n != 2 {
		t.Fatalf("step 3: logged %d WARNING lines, want 2 (a second, distinct conflict transition); log:\n%s", n, buf.String())
	}
}

// Reason is served from the unauthenticated /api/setup/state. It names the
// config file so an operator can act on it, but must never carry the absolute
// path, which would disclose the directory layout and the account name.
//
// The filesystem error is the easy one to get wrong: *fs.PathError embeds the
// path in Error(), so the cause has to be unwrapped rather than formatted.
func TestResolve_ReasonNeverContainsTheAbsoluteConfigPath(t *testing.T) {
	tests := []struct {
		name string
		// legacyFlag is chosen per case so that every one produces a non-empty
		// Reason: the flag must disagree with the config to force a conflict.
		legacyFlag bool
		setup      func(t *testing.T) string // returns the config path
	}{
		{
			name:       "missing file",
			legacyFlag: true,
			setup:      func(t *testing.T) string { return filepath.Join(t.TempDir(), "config.yaml") },
		},
		{
			name:       "malformed config",
			legacyFlag: true,
			setup:      func(t *testing.T) string { return newConfigFile(t, configMalformed) },
		},
		{
			name:       "conflict, config off",
			legacyFlag: true,
			setup:      func(t *testing.T) string { return newConfigFile(t, configActivated) },
		},
		{
			name:       "conflict, config on",
			legacyFlag: false,
			setup:      func(t *testing.T) string { return newConfigFile(t, configSetupModeOn) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := tt.setup(t)
			got := New(path, tt.legacyFlag).Resolve()

			if got.Reason == "" {
				t.Fatalf("Reason is empty; nothing to check")
			}
			if strings.Contains(got.Reason, path) {
				t.Fatalf("Reason leaks the absolute config path %q: %s", path, got.Reason)
			}
			if dir := filepath.Dir(path); strings.Contains(got.Reason, dir) {
				t.Fatalf("Reason leaks the config directory %q: %s", dir, got.Reason)
			}
			if !strings.Contains(got.Reason, filepath.Base(path)) {
				t.Fatalf("Reason %q dropped the config file name, losing its diagnostic value", got.Reason)
			}
		})
	}
}

// The Reason keeps the cause, which tells an operator whether the file is
// missing or unreadable, while dropping the path.
func TestResolve_UnreadableReasonKeepsCauseWithoutPath(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")

	got := New(path, false).Resolve()

	if !strings.Contains(got.Reason, "no such file or directory") {
		t.Fatalf("Reason = %q, want it to keep the underlying cause", got.Reason)
	}
	if strings.Contains(got.Reason, path) {
		t.Fatalf("Reason leaks the absolute path: %s", got.Reason)
	}
}

// errDetail unwraps *fs.PathError so the path never reaches a Reason. The
// fallback matters too: a non-PathError must still explain itself.
func TestErrDetail(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "path error keeps the cause and drops the path",
			err:  &fs.PathError{Op: "stat", Path: "/very/secret/path/config.yaml", Err: fs.ErrNotExist},
			want: fs.ErrNotExist.Error(),
		},
		{
			name: "wrapped path error is still unwrapped",
			err:  fmt.Errorf("reading config: %w", &fs.PathError{Op: "open", Path: "/secret/x.yaml", Err: fs.ErrPermission}),
			want: fs.ErrPermission.Error(),
		},
		{
			name: "plain error falls back to its own message",
			err:  errors.New("some other failure"),
			want: "some other failure",
		},
		{
			// PathError.Error() dereferences Err, so this input panics if
			// errDetail falls through to it. The stdlib never builds one, but a
			// guard that still panics is worse than no guard.
			name: "path error with no cause is reported without panicking",
			err:  &fs.PathError{Op: "stat", Path: "/secret/x.yaml"},
			want: "unknown filesystem error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := errDetail(tt.err); got != tt.want {
				t.Fatalf("errDetail() = %q, want %q", got, tt.want)
			}
		})
	}
}
