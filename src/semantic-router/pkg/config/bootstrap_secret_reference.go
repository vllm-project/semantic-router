package config

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

var bootstrapEnvNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func validateSecretSource(path, file, env string, required bool) error {
	rawFile, rawEnv := file, env
	file = strings.TrimSpace(rawFile)
	env = strings.TrimSpace(rawEnv)
	if file != rawFile || env != rawEnv {
		return fmt.Errorf("%s file and env references must not contain surrounding whitespace", path)
	}
	if file != "" && env != "" {
		return fmt.Errorf("%s_file and %s_env are mutually exclusive", path, path)
	}
	if required && file == "" && env == "" {
		return fmt.Errorf("%s requires exactly one file or env reference", path)
	}
	if file != "" {
		if !filepath.IsAbs(file) || strings.ContainsAny(file, "\r\n") {
			return fmt.Errorf("%s_file must be an absolute secret-file path", path)
		}
	}
	if env != "" && !bootstrapEnvNamePattern.MatchString(env) {
		return fmt.Errorf("%s_env must name an environment variable", path)
	}
	return nil
}
