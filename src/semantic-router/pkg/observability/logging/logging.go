package logging

import (
	"os"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

// componentKey is the structured field name for the component identifier.
const componentKey = "component"

// Config holds logger configuration.
type Config struct {
	// Level is one of: debug, info, warn, error, dpanic, panic, fatal
	Level string
	// Encoding is one of: json, console
	Encoding string
	// Development enables dev-friendly logging (stacktraces on error, etc.)
	Development bool
	// AddCaller enables caller annotations.
	AddCaller bool
}

// InitLogger initializes a global zap logger using the provided config.
// It also redirects the standard library logger to zap and returns the logger.
func InitLogger(cfg Config) (*zap.Logger, error) {
	zcfg := zap.NewProductionConfig()

	// Level
	lvl := strings.ToLower(strings.TrimSpace(cfg.Level))
	switch lvl {
	case "", "info":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	case "debug":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.DebugLevel)
	case "warn", "warning":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.WarnLevel)
	case "error":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.ErrorLevel)
	case "dpanic":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.DPanicLevel)
	case "panic":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.PanicLevel)
	case "fatal":
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.FatalLevel)
	default:
		zcfg.Level = zap.NewAtomicLevelAt(zapcore.InfoLevel)
	}

	// Encoding
	enc := strings.ToLower(strings.TrimSpace(cfg.Encoding))
	if enc == "console" {
		zcfg.Encoding = "console"
	} else {
		zcfg.Encoding = "json"
	}

	if cfg.Development {
		zcfg = zap.NewDevelopmentConfig()
		// Apply encoding override if specified
		if enc != "" {
			zcfg.Encoding = enc
		}
	}

	// Common fields
	zcfg.EncoderConfig.TimeKey = "ts"
	// ISO 8601 with millisecond precision
	zcfg.EncoderConfig.EncodeTime = zapcore.TimeEncoderOfLayout("2006-01-02T15:04:05.000")
	zcfg.EncoderConfig.MessageKey = "msg"
	zcfg.EncoderConfig.LevelKey = "level"
	zcfg.EncoderConfig.EncodeLevel = zapcore.LowercaseLevelEncoder
	zcfg.EncoderConfig.CallerKey = "caller"
	// Custom caller encoder: only filename:line (no package path)
	zcfg.EncoderConfig.EncodeCaller = func(caller zapcore.EntryCaller, enc zapcore.PrimitiveArrayEncoder) {
		// Extract just the filename from the full path
		// e.g., "pkg/modeldownload/downloader.go:82" -> "downloader.go:82"
		file := caller.TrimmedPath()
		// Find the last slash to get just filename
		for i := len(file) - 1; i >= 0; i-- {
			if file[i] == '/' {
				file = file[i+1:]
				break
			}
		}
		enc.AppendString(file)
	}

	// Build logger
	logger, err := zcfg.Build()
	if err != nil {
		return nil, err
	}

	if cfg.AddCaller {
		logger = logger.WithOptions(zap.AddCaller(), zap.AddCallerSkip(1))
	}

	// Replace globals and redirect stdlib log
	zap.ReplaceGlobals(logger)
	_ = zap.RedirectStdLog(logger)

	return logger, nil
}

// InitLoggerFromEnv builds a logger from environment variables and initializes it.
// Supported env vars:
//
//	SR_LOG_LEVEL       (debug|info|warn|error|dpanic|panic|fatal) default: info
//	SR_LOG_ENCODING    (json|console) default: json
//	SR_LOG_DEVELOPMENT (true|false) default: false
//	SR_LOG_ADD_CALLER  (true|false) default: true
func InitLoggerFromEnv() (*zap.Logger, error) {
	cfg := Config{
		Level:       getenvDefault("SR_LOG_LEVEL", "info"),
		Encoding:    getenvDefault("SR_LOG_ENCODING", "json"),
		Development: parseBool(getenvDefault("SR_LOG_DEVELOPMENT", "false")),
		AddCaller:   parseBool(getenvDefault("SR_LOG_ADD_CALLER", "true")),
	}
	return InitLogger(cfg)
}

func getenvDefault(k, d string) string {
	v := os.Getenv(k)
	if v == "" {
		return d
	}
	return v
}

func parseBool(s string) bool {
	s = strings.TrimSpace(strings.ToLower(s))
	return s == "1" || s == "true" || s == "yes" || s == "on"
}

// LogEvent emits a structured log at info level with a standard envelope.
// Fields provided by callers take precedence and will not be overwritten.
func LogEvent(event string, fields map[string]interface{}) {
	logEventAt(zapcore.InfoLevel, event, fields)
}

// DebugEvent emits a structured log at debug level.
func DebugEvent(event string, fields map[string]interface{}) {
	logEventAt(zapcore.DebugLevel, event, fields)
}

// WarnEvent emits a structured log at warn level.
func WarnEvent(event string, fields map[string]interface{}) {
	logEventAt(zapcore.WarnLevel, event, fields)
}

// ErrorEvent emits a structured log at error level.
func ErrorEvent(event string, fields map[string]interface{}) {
	logEventAt(zapcore.ErrorLevel, event, fields)
}

func logEventAt(level zapcore.Level, event string, fields map[string]interface{}) {
	prepared := prepareEventFields(event, fields)
	zfields := make([]zap.Field, 0, len(prepared))
	for k, v := range prepared {
		zfields = append(zfields, zap.Any(k, v))
	}

	logger := zap.L().With(zfields...)
	switch level {
	case zapcore.DebugLevel:
		logger.Debug(event)
	case zapcore.WarnLevel:
		logger.Warn(event)
	case zapcore.ErrorLevel:
		logger.Error(event)
	case zapcore.FatalLevel:
		logger.Fatal(event)
	default:
		logger.Info(event)
	}
}

func prepareEventFields(event string, fields map[string]interface{}) map[string]interface{} {
	prepared := make(map[string]interface{}, len(fields)+1)
	for k, v := range fields {
		prepared[k] = v
	}
	if _, ok := prepared["event"]; !ok {
		prepared["event"] = event
	}
	return prepared
}

// Helper printf-style wrappers to ease migration from log.Printf.
func Infof(format string, args ...interface{})  { zap.S().Infof(format, args...) }
func Warnf(format string, args ...interface{})  { zap.S().Warnf(format, args...) }
func Errorf(format string, args ...interface{}) { zap.S().Errorf(format, args...) }
func Debugf(format string, args ...interface{}) { zap.S().Debugf(format, args...) }
func Fatalf(format string, args ...interface{}) { zap.S().Fatalf(format, args...) }

// WithComponent returns a named sub-logger that carries a "component" field
// in every log line, making it trivial to filter by subsystem.
//
// Usage:
//
//	log := logging.WithComponent("extproc")
//	log.Infof("request received")   // {"component":"extproc","level":"info",...}
func WithComponent(name string) *zap.SugaredLogger {
	return zap.L().With(zap.String(componentKey, name)).Sugar()
}

// ComponentEvent is like LogEvent but includes the "component" field.
func ComponentEvent(component, event string, fields map[string]interface{}) {
	logComponentEventAt(zapcore.InfoLevel, component, event, fields)
}

// ComponentDebugEvent is like DebugEvent but includes the "component" field.
func ComponentDebugEvent(component, event string, fields map[string]interface{}) {
	logComponentEventAt(zapcore.DebugLevel, component, event, fields)
}

// ComponentWarnEvent is like WarnEvent but includes the "component" field.
func ComponentWarnEvent(component, event string, fields map[string]interface{}) {
	logComponentEventAt(zapcore.WarnLevel, component, event, fields)
}

// ComponentErrorEvent is like ErrorEvent but includes the "component" field.
func ComponentErrorEvent(component, event string, fields map[string]interface{}) {
	logComponentEventAt(zapcore.ErrorLevel, component, event, fields)
}

// ComponentFatalEvent is like FatalEvent but includes the "component" field.
func ComponentFatalEvent(component, event string, fields map[string]interface{}) {
	logComponentEventAt(zapcore.FatalLevel, component, event, fields)
}

func logComponentEventAt(level zapcore.Level, component, event string, fields map[string]interface{}) {
	prepared := prepareEventFields(event, fields)
	if _, ok := prepared[componentKey]; !ok {
		prepared[componentKey] = component
	}
	logEventAt(level, event, prepared)
}
