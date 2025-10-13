package config

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("IP Address Validation", func() {
	Describe("validateIPAddress", func() {
		Context("with valid IPv4 addresses", func() {
			It("should accept standard IPv4 addresses", func() {
				validIPv4Addresses := []string{
					"127.0.0.1",
					"192.168.1.1",
					"10.0.0.1",
					"172.16.0.1",
					"8.8.8.8",
					"255.255.255.255",
					"0.0.0.0",
				}

				for _, addr := range validIPv4Addresses {
					err := validateIPAddress(addr)
					Expect(err).NotTo(HaveOccurred(), "Expected %s to be valid", addr)
				}
			})
		})

		Context("with valid IPv6 addresses", func() {
			It("should accept standard IPv6 addresses", func() {
				validIPv6Addresses := []string{
					"::1",
					"2001:db8::1",
					"fe80::1",
					"2001:0db8:85a3:0000:0000:8a2e:0370:7334",
					"2001:db8:85a3::8a2e:370:7334",
					"::",
					"::ffff:192.0.2.1",
				}

				for _, addr := range validIPv6Addresses {
					err := validateIPAddress(addr)
					Expect(err).NotTo(HaveOccurred(), "Expected %s to be valid", addr)
				}
			})
		})

		Context("with domain names", func() {
			It("should reject domain names", func() {
				domainNames := []string{
					"example.com",
					"localhost",
					"api.openai.com",
					"subdomain.example.org",
					"test.local",
				}

				for _, domain := range domainNames {
					err := validateIPAddress(domain)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", domain)
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})

		Context("with protocol prefixes", func() {
			It("should reject HTTP/HTTPS prefixes", func() {
				protocolAddresses := []string{
					"http://127.0.0.1",
					"https://192.168.1.1",
					"http://example.com",
					"https://api.openai.com",
				}

				for _, addr := range protocolAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("protocol prefixes"))
					Expect(err.Error()).To(ContainSubstring("are not supported"))
				}
			})
		})

		Context("with paths", func() {
			It("should reject addresses with paths", func() {
				pathAddresses := []string{
					"127.0.0.1/api",
					"192.168.1.1/health",
					"example.com/v1/api",
					"localhost/status",
				}

				for _, addr := range pathAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("paths are not supported"))
				}
			})
		})

		Context("with port numbers", func() {
			It("should reject IPv4 addresses with port numbers", func() {
				ipv4PortAddresses := []string{
					"127.0.0.1:8080",
					"192.168.1.1:3000",
					"10.0.0.1:443",
				}

				for _, addr := range ipv4PortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("port numbers in address are not supported"))
					Expect(err.Error()).To(ContainSubstring("use 'port' field instead"))
				}
			})

			It("should reject IPv6 addresses with port numbers", func() {
				ipv6PortAddresses := []string{
					"[::1]:8080",
					"[2001:db8::1]:3000",
					"[fe80::1]:443",
				}

				for _, addr := range ipv6PortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					Expect(err.Error()).To(ContainSubstring("port numbers in address are not supported"))
					Expect(err.Error()).To(ContainSubstring("use 'port' field instead"))
				}
			})

			It("should reject domain names with port numbers", func() {
				domainPortAddresses := []string{
					"localhost:8000",
					"example.com:443",
				}

				for _, addr := range domainPortAddresses {
					err := validateIPAddress(addr)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", addr)
					// 这些会被域名检测捕获，而不是端口检测
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})

		Context("with empty or invalid input", func() {
			It("should reject empty strings", func() {
				emptyInputs := []string{
					"",
					"   ",
					"\t",
					"\n",
				}

				for _, input := range emptyInputs {
					err := validateIPAddress(input)
					Expect(err).To(HaveOccurred(), "Expected '%s' to be rejected", input)
					Expect(err.Error()).To(ContainSubstring("address cannot be empty"))
				}
			})

			It("should reject invalid formats", func() {
				invalidFormats := []string{
					"not-an-ip",
					"256.256.256.256",
					"192.168.1",
					"192.168.1.1.1",
					"gggg::1",
				}

				for _, format := range invalidFormats {
					err := validateIPAddress(format)
					Expect(err).To(HaveOccurred(), "Expected %s to be rejected", format)
					Expect(err.Error()).To(ContainSubstring("invalid IP address format"))
				}
			})
		})
	})

	Describe("validateVLLMEndpoints", func() {
		Context("with valid endpoints", func() {
			It("should accept endpoints with valid IP addresses", func() {
				endpoints := []VLLMEndpoint{
					{
						Name:    "endpoint1",
						Address: "127.0.0.1",
						Port:    8000,
					},
					{
						Name:    "endpoint2",
						Address: "::1",
						Port:    8001,
					},
				}

				err := validateVLLMEndpoints(endpoints)
				Expect(err).NotTo(HaveOccurred())
			})
		})

		Context("with invalid endpoints", func() {
			It("should reject endpoints with domain names", func() {
				endpoints := []VLLMEndpoint{
					{
						Name:    "invalid-endpoint",
						Address: "example.com",
						Port:    8000,
					},
				}

				err := validateVLLMEndpoints(endpoints)
				Expect(err).To(HaveOccurred())
				Expect(err.Error()).To(ContainSubstring("invalid-endpoint"))
				Expect(err.Error()).To(ContainSubstring("address validation failed"))
				Expect(err.Error()).To(ContainSubstring("Supported formats"))
				Expect(err.Error()).To(ContainSubstring("IPv4: 192.168.1.1"))
				Expect(err.Error()).To(ContainSubstring("IPv6: ::1"))
				Expect(err.Error()).To(ContainSubstring("Unsupported formats"))
			})

			It("should provide detailed error messages", func() {
				endpoints := []VLLMEndpoint{
					{
						Name:    "test-endpoint",
						Address: "http://127.0.0.1",
						Port:    8000,
					},
				}

				err := validateVLLMEndpoints(endpoints)
				Expect(err).To(HaveOccurred())

				errorMsg := err.Error()
				Expect(errorMsg).To(ContainSubstring("test-endpoint"))
				Expect(errorMsg).To(ContainSubstring("protocol prefixes"))
				Expect(errorMsg).To(ContainSubstring("Domain names: example.com, localhost"))
				Expect(errorMsg).To(ContainSubstring("Protocol prefixes: http://, https://"))
				Expect(errorMsg).To(ContainSubstring("use 'port' field instead"))
			})
		})
	})

	Describe("helper functions", func() {
		Describe("isValidIPv4", func() {
			It("should correctly identify IPv4 addresses", func() {
				Expect(isValidIPv4("127.0.0.1")).To(BeTrue())
				Expect(isValidIPv4("192.168.1.1")).To(BeTrue())
				Expect(isValidIPv4("::1")).To(BeFalse())
				Expect(isValidIPv4("example.com")).To(BeFalse())
			})
		})

		Describe("isValidIPv6", func() {
			It("should correctly identify IPv6 addresses", func() {
				Expect(isValidIPv6("::1")).To(BeTrue())
				Expect(isValidIPv6("2001:db8::1")).To(BeTrue())
				Expect(isValidIPv6("127.0.0.1")).To(BeFalse())
				Expect(isValidIPv6("example.com")).To(BeFalse())
			})
		})

		Describe("getIPAddressType", func() {
			It("should return correct IP address types", func() {
				Expect(getIPAddressType("127.0.0.1")).To(Equal("IPv4"))
				Expect(getIPAddressType("::1")).To(Equal("IPv6"))
				Expect(getIPAddressType("example.com")).To(Equal("invalid"))
			})
		})
	})
})

var _ = Describe("MCP Configuration Validation", func() {
	Describe("IsMCPCategoryClassifierEnabled", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when MCP is fully configured", func() {
			It("should return true", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeTrue())
			})
		})

		Context("when MCP is not enabled", func() {
			It("should return false", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = false
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when MCP tool name is empty", func() {
			It("should return false", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = ""

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when both enabled and tool name are missing", func() {
			It("should return false", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = false
				cfg.Classifier.MCPCategoryModel.ToolName = ""

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})
	})

	Describe("MCP Configuration Structure", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when configuring stdio transport", func() {
			It("should accept valid stdio configuration", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
				cfg.Classifier.MCPCategoryModel.Command = "python"
				cfg.Classifier.MCPCategoryModel.Args = []string{"server.py"}
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
				cfg.Classifier.MCPCategoryModel.Threshold = 0.5
				cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 30

				Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
				Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("stdio"))
				Expect(cfg.Classifier.MCPCategoryModel.Command).To(Equal("python"))
				Expect(cfg.Classifier.MCPCategoryModel.Args).To(HaveLen(1))
				Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify_text"))
				Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.5))
				Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(30))
			})

			It("should accept environment variables", func() {
				cfg.Classifier.MCPCategoryModel.Env = map[string]string{
					"PYTHONPATH": "/app/lib",
					"LOG_LEVEL":  "debug",
				}

				Expect(cfg.Classifier.MCPCategoryModel.Env).To(HaveLen(2))
				Expect(cfg.Classifier.MCPCategoryModel.Env["PYTHONPATH"]).To(Equal("/app/lib"))
				Expect(cfg.Classifier.MCPCategoryModel.Env["LOG_LEVEL"]).To(Equal("debug"))
			})
		})

		Context("when configuring HTTP transport", func() {
			It("should accept valid HTTP configuration", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.TransportType = "http"
				cfg.Classifier.MCPCategoryModel.URL = "http://localhost:8080/mcp"
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("http"))
				Expect(cfg.Classifier.MCPCategoryModel.URL).To(Equal("http://localhost:8080/mcp"))
			})
		})

		Context("when threshold is not set", func() {
			It("should default to zero", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.0))
			})
		})

		Context("when configuring custom threshold", func() {
			It("should accept threshold values between 0 and 1", func() {
				testCases := []float32{0.0, 0.3, 0.5, 0.7, 0.9, 1.0}

				for _, threshold := range testCases {
					cfg.Classifier.MCPCategoryModel.Threshold = threshold
					Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", threshold))
				}
			})
		})

		Context("when timeout is not set", func() {
			It("should default to zero", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(0))
			})
		})
	})

	Describe("MCP vs In-tree Classifier Priority", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		Context("when both in-tree and MCP are configured", func() {
			It("should have both configurations available", func() {
				// Configure in-tree classifier
				cfg.Classifier.CategoryModel.ModelID = "/path/to/model"
				cfg.Classifier.CategoryModel.CategoryMappingPath = "/path/to/mapping.json"
				cfg.Classifier.CategoryModel.Threshold = 0.7

				// Configure MCP classifier
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"
				cfg.Classifier.MCPCategoryModel.Threshold = 0.5

				// Both should be configured
				Expect(cfg.Classifier.CategoryModel.ModelID).ToNot(BeEmpty())
				Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			})
		})

		Context("when only in-tree is configured", func() {
			It("should not have MCP enabled", func() {
				cfg.Classifier.CategoryModel.ModelID = "/path/to/model"
				cfg.Classifier.CategoryModel.CategoryMappingPath = "/path/to/mapping.json"

				Expect(cfg.Classifier.CategoryModel.ModelID).ToNot(BeEmpty())
				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})

		Context("when only MCP is configured", func() {
			It("should have MCP enabled and no in-tree model", func() {
				cfg.Classifier.MCPCategoryModel.Enabled = true
				cfg.Classifier.MCPCategoryModel.ToolName = "classify_text"

				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeTrue())
				Expect(cfg.Classifier.CategoryModel.ModelID).To(BeEmpty())
			})
		})

		Context("when neither is configured", func() {
			It("should have neither enabled", func() {
				Expect(cfg.Classifier.CategoryModel.ModelID).To(BeEmpty())
				Expect(cfg.IsMCPCategoryClassifierEnabled()).To(BeFalse())
			})
		})
	})

	Describe("MCP Configuration Fields", func() {
		var cfg *RouterConfig

		BeforeEach(func() {
			cfg = &RouterConfig{}
		})

		It("should support all required fields for stdio transport", func() {
			cfg.Classifier.MCPCategoryModel.Enabled = true
			cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
			cfg.Classifier.MCPCategoryModel.Command = "python3"
			cfg.Classifier.MCPCategoryModel.Args = []string{"-m", "server"}
			cfg.Classifier.MCPCategoryModel.Env = map[string]string{"DEBUG": "1"}
			cfg.Classifier.MCPCategoryModel.ToolName = "classify"
			cfg.Classifier.MCPCategoryModel.Threshold = 0.6
			cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 60

			Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("stdio"))
			Expect(cfg.Classifier.MCPCategoryModel.Command).To(Equal("python3"))
			Expect(cfg.Classifier.MCPCategoryModel.Args).To(Equal([]string{"-m", "server"}))
			Expect(cfg.Classifier.MCPCategoryModel.Env).To(HaveKeyWithValue("DEBUG", "1"))
			Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify"))
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("~", 0.6, 0.01))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(60))
		})

		It("should support all required fields for HTTP transport", func() {
			cfg.Classifier.MCPCategoryModel.Enabled = true
			cfg.Classifier.MCPCategoryModel.TransportType = "http"
			cfg.Classifier.MCPCategoryModel.URL = "https://mcp-server:443/api"
			cfg.Classifier.MCPCategoryModel.ToolName = "classify"
			cfg.Classifier.MCPCategoryModel.Threshold = 0.8
			cfg.Classifier.MCPCategoryModel.TimeoutSeconds = 120

			Expect(cfg.Classifier.MCPCategoryModel.Enabled).To(BeTrue())
			Expect(cfg.Classifier.MCPCategoryModel.TransportType).To(Equal("http"))
			Expect(cfg.Classifier.MCPCategoryModel.URL).To(Equal("https://mcp-server:443/api"))
			Expect(cfg.Classifier.MCPCategoryModel.ToolName).To(Equal("classify"))
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("~", 0.8, 0.01))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(120))
		})

		It("should allow optional fields to be omitted", func() {
			cfg.Classifier.MCPCategoryModel.Enabled = true
			cfg.Classifier.MCPCategoryModel.TransportType = "stdio"
			cfg.Classifier.MCPCategoryModel.Command = "server"
			cfg.Classifier.MCPCategoryModel.ToolName = "classify"

			// Optional fields should have zero values
			Expect(cfg.Classifier.MCPCategoryModel.Args).To(BeNil())
			Expect(cfg.Classifier.MCPCategoryModel.Env).To(BeNil())
			Expect(cfg.Classifier.MCPCategoryModel.URL).To(BeEmpty())
			Expect(cfg.Classifier.MCPCategoryModel.Threshold).To(BeNumerically("==", 0.0))
			Expect(cfg.Classifier.MCPCategoryModel.TimeoutSeconds).To(Equal(0))
		})
	})
})
