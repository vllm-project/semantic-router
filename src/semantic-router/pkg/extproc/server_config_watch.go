package extproc

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	configReloadDebounceWindow = 250 * time.Millisecond
	configReloadSettleDelay    = 300 * time.Millisecond
)

type configFileReloadLoop struct {
	server  *Server
	watcher *fsnotify.Watcher
	cfgFile string
	cfgDir  string
	pending bool
	last    time.Time
}

// watchConfigAndReload watches the active config source and reloads the router on changes.
func (s *Server) watchConfigAndReload(ctx context.Context) {
	if s.usesKubernetesConfigSource() {
		logging.ComponentEvent("extproc", "config_update_watch_started", map[string]interface{}{
			"source": "kubernetes",
		})
		s.watchKubernetesConfigUpdates(ctx)
		return
	}

	s.watchFileConfigAndReload(ctx)
}

func (s *Server) watchFileConfigAndReload(ctx context.Context) {
	watcher, cfgDir, ok := newConfigFileWatcher(s.configPath)
	if !ok {
		return
	}
	defer func() {
		_ = watcher.Close()
	}()

	loop := configFileReloadLoop{
		server:  s,
		watcher: watcher,
		cfgFile: s.configPath,
		cfgDir:  cfgDir,
	}
	loop.run(ctx)
}

func newConfigFileWatcher(cfgFile string) (*fsnotify.Watcher, string, bool) {
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
			"stage": "create_watcher",
			"error": err.Error(),
		})
		return nil, "", false
	}

	cfgDir := filepath.Dir(cfgFile)
	if err := watcher.Add(cfgDir); err != nil {
		logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
			"stage": "watch_dir",
			"dir":   cfgDir,
			"error": err.Error(),
		})
		_ = watcher.Close()
		return nil, "", false
	}

	_ = watcher.Add(cfgFile)
	return watcher, cfgDir, true
}

func (l *configFileReloadLoop) run(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case ev, ok := <-l.watcher.Events:
			if !ok {
				return
			}
			l.handleEvent(ev)
		case err, ok := <-l.watcher.Errors:
			if !ok {
				return
			}
			logging.ComponentErrorEvent("extproc", "config_watcher_error", map[string]interface{}{
				"stage": "watch_loop",
				"error": err.Error(),
			})
		}
	}
}

func (l *configFileReloadLoop) handleEvent(ev fsnotify.Event) {
	logging.ComponentDebugEvent("extproc", "config_watcher_event", map[string]interface{}{
		"name": ev.Name,
		"op":   ev.Op.String(),
	})
	if !isConfigMutationOp(ev.Op) || !shouldReloadForConfigEvent(l.cfgFile, l.cfgDir, ev.Name) {
		return
	}
	l.scheduleReload(ev)
}

func isConfigMutationOp(op fsnotify.Op) bool {
	return op&(fsnotify.Write|fsnotify.Create|fsnotify.Rename|fsnotify.Remove|fsnotify.Chmod) != 0
}

func shouldReloadForConfigEvent(cfgFile, cfgDir, eventPath string) bool {
	if eventPath == "" {
		return false
	}

	cleanEventPath := filepath.Clean(eventPath)
	if cleanEventPath == filepath.Clean(cfgFile) {
		return true
	}

	if filepath.Dir(cleanEventPath) != filepath.Clean(cfgDir) {
		return false
	}

	base := filepath.Base(cleanEventPath)
	if base == filepath.Base(cfgFile) {
		return true
	}
	if strings.HasPrefix(base, ".vllm-sr-write-check-") {
		return false
	}

	return strings.HasPrefix(base, "..data")
}

func (l *configFileReloadLoop) scheduleReload(ev fsnotify.Event) {
	if l.pending && time.Since(l.last) <= configReloadDebounceWindow {
		logging.ComponentDebugEvent("extproc", "config_reload_debounced", map[string]interface{}{
			"file": ev.Name,
		})
		return
	}

	l.pending = true
	l.last = time.Now()
	logging.ComponentEvent("extproc", "config_reload_scheduled", map[string]interface{}{
		"file":     ev.Name,
		"event":    ev.Op.String(),
		"delay_ms": int(configReloadSettleDelay / time.Millisecond),
	})
	goSafely("config_reload_debouncer", func() {
		time.Sleep(configReloadSettleDelay)
		l.reload()
	})
}

func (l *configFileReloadLoop) reload() {
	logging.ComponentDebugEvent("extproc", "config_reload_triggered", map[string]interface{}{
		"file": l.cfgFile,
	})
	l.logConfigFileStat()

	if err := l.server.reloadRouterFromFile(l.cfgFile); err != nil {
		l.logReloadFailure(err)
		return
	}
	l.logReloadSuccess()
}

func (l *configFileReloadLoop) logConfigFileStat() {
	info, err := os.Stat(l.cfgFile)
	if err != nil {
		logging.ComponentDebugEvent("extproc", "config_file_stat_failed", map[string]interface{}{
			"file":  l.cfgFile,
			"error": err.Error(),
		})
		return
	}

	logging.ComponentDebugEvent("extproc", "config_file_stat", map[string]interface{}{
		"file":       l.cfgFile,
		"size_bytes": info.Size(),
		"mod_time":   info.ModTime().Format("2006-01-02 15:04:05"),
	})
}

func (l *configFileReloadLoop) logReloadFailure(err error) {
	event := map[string]interface{}{
		"file":  l.cfgFile,
		"error": err.Error(),
	}
	if strings.Contains(err.Error(), "model download preflight failed") {
		event["stage"] = "model_download"
	}
	logging.ComponentErrorEvent("extproc", "config_reload_failed", event)
}

func (l *configFileReloadLoop) logReloadSuccess() {
	newRouter := l.server.service.GetRouter()
	event := map[string]interface{}{
		"file": l.cfgFile,
	}
	if newRouter != nil && newRouter.Config != nil {
		event["decision_count"] = len(newRouter.Config.Decisions)
	}
	logging.ComponentEvent("extproc", "config_reloaded", event)
}

// watchKubernetesConfigUpdates watches for config updates from the Kubernetes controller.
func (s *Server) watchKubernetesConfigUpdates(ctx context.Context) {
	subscription := config.SubscribeConfigUpdates(1)
	defer subscription.Close()
	updateCh := subscription.Updates()

	for {
		select {
		case <-ctx.Done():
			return
		case newCfg := <-updateCh:
			s.handleKubernetesConfigUpdate(newCfg)
		}
	}
}

func (s *Server) handleKubernetesConfigUpdate(newCfg *config.RouterConfig) {
	if newCfg == nil {
		return
	}

	err := s.reloadRouterFromConfig("kubernetes", s.configPath, newCfg)
	if err != nil {
		logging.ComponentErrorEvent("extproc", "config_reload_failed", map[string]interface{}{
			"source": "kubernetes",
			"error":  err.Error(),
		})
		return
	}

	logging.ComponentEvent("extproc", "config_reloaded", map[string]interface{}{
		"source":         "kubernetes",
		"decision_count": len(newCfg.Decisions),
	})
}
