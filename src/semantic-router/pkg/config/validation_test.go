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
						Models:  []string{"model1"},
					},
					{
						Name:    "endpoint2",
						Address: "::1",
						Port:    8001,
						Models:  []string{"model2"},
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
						Models:  []string{"model1"},
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
						Models:  []string{"model1"},
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
