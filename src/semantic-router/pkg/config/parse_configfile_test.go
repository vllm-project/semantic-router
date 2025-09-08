package config_test

import (
	"os"
	"path/filepath"
	"runtime"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/semantic-router/pkg/config"
)

var _ = Describe("ParseConfigFile and ReplaceGlobalConfig", func() {
	var tempDir string

	BeforeEach(func() {
		var err error
		tempDir, err = os.MkdirTemp("", "config_parse_test")
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		os.RemoveAll(tempDir)
		config.ResetConfig()
	})

	It("should parse configuration via symlink path", func() {
		if runtime.GOOS == "windows" {
			Skip("symlink test is skipped on Windows")
		}

		// Create real config target
		target := filepath.Join(tempDir, "real-config.yaml")
		content := []byte("default_model: test-model\n")
		Expect(os.WriteFile(target, content, 0o644)).To(Succeed())

		// Create symlink pointing to target
		link := filepath.Join(tempDir, "link-config.yaml")
		Expect(os.Symlink(target, link)).To(Succeed())

		cfg, err := config.ParseConfigFile(link)
		Expect(err).NotTo(HaveOccurred())
		Expect(cfg).NotTo(BeNil())
		Expect(cfg.DefaultModel).To(Equal("test-model"))
	})

	It("should return error when file does not exist", func() {
		_, err := config.ParseConfigFile(filepath.Join(tempDir, "no-such.yaml"))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("failed to read config file"))
	})

	It("should replace global config and reflect via GetConfig", func() {
		// new config instance
		newCfg := &config.RouterConfig{DefaultModel: "new-default"}
		config.ReplaceGlobalConfig(newCfg)
		got := config.GetConfig()
		Expect(got).To(Equal(newCfg))
		Expect(got.DefaultModel).To(Equal("new-default"))
	})
})
