package mcp

import (
	"fmt"
	"log"
)

// ClientFactory creates MCP clients based on configuration
type ClientFactory struct{}

// NewClientFactory creates a new client factory
func NewClientFactory() *ClientFactory {
	return &ClientFactory{}
}

// CreateClient creates an MCP client based on the configuration
func (f *ClientFactory) CreateClient(name string, config ClientConfig) (MCPClient, error) {
	transportType := f.determineTransportType(config)

	switch transportType {
	case string(TransportStdio):
		return NewStdioClient(name, config), nil
	case string(TransportStreamableHTTP), "http":
		return NewHTTPClient(name, config), nil
	default:
		return nil, fmt.Errorf("unsupported transport type: %s", transportType)
	}
}

// determineTransportType determines the transport type from configuration
func (f *ClientFactory) determineTransportType(config ClientConfig) string {
	if config.TransportType != "" {
		return config.TransportType
	}

	if config.Command != "" {
		return string(TransportStdio)
	}

	if config.URL != "" {
		return string(TransportStreamableHTTP)
	}

	return string(TransportStdio)
}

// CreateClientWithDefaults creates a client with default configuration
func (f *ClientFactory) CreateClientWithDefaults(name string, transportType TransportType) (MCPClient, error) {
	config := ClientConfig{
		TransportType: string(transportType),
		Options: ClientOptions{
			LogEnabled: true,
		},
	}

	return f.CreateClient(name, config)
}

// NewClient is a convenience function that creates a client using the factory
func NewClient(name string, config ClientConfig) (MCPClient, error) {
	factory := NewClientFactory()
	return factory.CreateClient(name, config)
}

// NewClientFromConfig creates a client from configuration with validation
func NewClientFromConfig(name string, config ClientConfig) (MCPClient, error) {
	if err := validateConfig(config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return NewClient(name, config)
}

// validateConfig validates the client configuration
func validateConfig(config ClientConfig) error {
	transportType := config.TransportType
	if transportType == "" {
		// Auto-detect transport type
		if config.Command == "" && config.URL == "" {
			return fmt.Errorf("either command or URL must be specified")
		}
		return nil
	}

	switch TransportType(transportType) {
	case TransportStdio:
		if config.Command == "" {
			return fmt.Errorf("command is required for stdio transport")
		}
	case TransportStreamableHTTP:
		if config.URL == "" {
			return fmt.Errorf("URL is required for %s transport", transportType)
		}
	default:
		// Also accept "http" as an alias for "streamable-http"
		if transportType == "http" {
			if config.URL == "" {
				return fmt.Errorf("URL is required for http transport")
			}
		} else {
			return fmt.Errorf("unsupported transport type: %s", transportType)
		}
	}

	return nil
}

// CreateMultipleClients creates multiple clients from a map of configurations
func CreateMultipleClients(configs map[string]ClientConfig) (map[string]MCPClient, error) {
	factory := NewClientFactory()
	clients := make(map[string]MCPClient)

	for name, config := range configs {
		client, err := factory.CreateClient(name, config)
		if err != nil {
			// Close any already created clients on error
			for _, c := range clients {
				c.Close()
			}
			return nil, fmt.Errorf("failed to create client '%s': %w", name, err)
		}
		clients[name] = client
	}

	return clients, nil
}

// LoggingClientWrapper wraps any client with enhanced logging
type LoggingClientWrapper struct {
	MCPClient
	name string
}

// NewLoggingClientWrapper creates a new logging wrapper
func NewLoggingClientWrapper(client MCPClient, name string) *LoggingClientWrapper {
	wrapper := &LoggingClientWrapper{
		MCPClient: client,
		name:      name,
	}

	// Set up enhanced logging
	client.SetLogHandler(func(level LoggingLevel, message string) {
		log.Printf("[%s] %s: %s", level, name, message)
	})

	return wrapper
}

// Connect wraps the connection with additional logging
func (w *LoggingClientWrapper) Connect() error {
	log.Printf("Connecting client: %s", w.name)
	err := w.MCPClient.Connect()
	if err != nil {
		log.Printf("Failed to connect client %s: %v", w.name, err)
	} else {
		log.Printf("Successfully connected client: %s", w.name)
	}
	return err
}

// Close wraps the close with additional logging
func (w *LoggingClientWrapper) Close() error {
	log.Printf("Closing client: %s", w.name)
	err := w.MCPClient.Close()
	if err != nil {
		log.Printf("Error closing client %s: %v", w.name, err)
	} else {
		log.Printf("Successfully closed client: %s", w.name)
	}
	return err
}
