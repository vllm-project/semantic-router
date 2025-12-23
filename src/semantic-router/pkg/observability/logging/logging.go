package logging

import (
	"os"
	"strings"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

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
	// Custom time format: "2006-01-02T15:04:05" (no timezone)
	zcfg.EncoderConfig.EncodeTime = zapcore.TimeEncoderOfLayout("2006-01-02T15:04:05")
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
	if fields == nil {
		fields = map[string]interface{}{}
	}
	if _, ok := fields["event"]; !ok {
		fields["event"] = event
	}
	// Zap already includes a timestamp; preserve provided ts if any

	// Convert the map to zap fields
	zfields := make([]zap.Field, 0, len(fields))
	for k, v := range fields {
		zfields = append(zfields, zap.Any(k, v))
	}
	zap.L().With(zfields...).Info(event)
}

// Helper printf-style wrappers to ease migration from log.Printf.
func Infof(format string, args ...interface{})  { zap.S().Infof(format, args...) }
func Warnf(format string, args ...interface{})  { zap.S().Warnf(format, args...) }
func Errorf(format string, args ...interface{}) { zap.S().Errorf(format, args...) }
func Debugf(format string, args ...interface{}) { zap.S().Debugf(format, args...) }
func Fatalf(format string, args ...interface{}) { zap.S().Fatalf(format, args...) }
