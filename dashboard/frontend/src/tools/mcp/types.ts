/**
 * MCP (Model Context Protocol) Type Definitions
 * 遵循官方规范 2025-06-18
 */

// ========== Transport Types ==========

/**
 * MCP 传输类型 (官方规范 2025-06-18)
 */
export type MCPTransportType = 
  | 'stdio'           // 本地命令行 - 官方支持
  | 'streamable-http' // 流式 HTTP - 官方推荐

// ========== Server Configuration ==========

/**
 * MCP 服务器配置
 */
export interface MCPServerConfig {
  /** 唯一标识符 (UUID) */
  id: string
  
  /** 显示名称 */
  name: string
  
  /** 服务器描述 */
  description?: string
  
  /** 传输协议类型 (仅官方支持的 2 种) */
  transport: MCPTransportType
  
  /** 连接配置 */
  connection: MCPConnectionConfig
  
  /** 是否启用 */
  enabled: boolean
  
  /** 安全配置 (新增 - OAuth 2.1) */
  security?: MCPSecurityConfig
  
  /** 高级选项 */
  options?: MCPServerOptions
}

/**
 * 连接配置
 */
export interface MCPConnectionConfig {
  // ═══════════════════════════════════════════
  // Stdio 传输配置
  // ═══════════════════════════════════════════
  /** 可执行命令 */
  command?: string
  /** 命令参数 */
  args?: string[]
  /** 环境变量 */
  env?: Record<string, string>
  /** 工作目录 */
  cwd?: string
  
  // ═══════════════════════════════════════════
  // Streamable HTTP 传输配置
  // ═══════════════════════════════════════════
  /** 服务器 URL (单一端点) */
  url?: string
  /** 自定义请求头 */
  headers?: Record<string, string>
}

/**
 * 安全配置 (新增 - 官方 2025-06-18 规范)
 */
export interface MCPSecurityConfig {
  /** OAuth 2.1 认证配置 */
  oauth?: {
    /** OAuth 客户端 ID */
    clientId: string
    /** OAuth 客户端密钥 (仅机密客户端) */
    clientSecret?: string
    /** 授权端点 */
    authorizationUrl: string
    /** Token 端点 */
    tokenUrl: string
    /** 请求的权限范围 */
    scopes?: string[]
    /** 是否使用 PKCE (公共客户端强制要求) */
    usePKCE?: boolean
  }
  
  /** 允许的 Origin (防止 DNS 重绑定攻击) */
  allowedOrigins?: string[]
  
  /** 是否仅限本地访问 */
  localOnly?: boolean
}

/**
 * 高级选项
 */
export interface MCPServerOptions {
  /** 自动重连 */
  autoReconnect?: boolean
  /** 重连间隔 (ms) */
  reconnectInterval?: number
  /** 请求超时 (ms) */
  timeout?: number
  /** 最大重试次数 */
  maxRetries?: number
}

// ========== Server Status ==========

/**
 * 服务器连接状态
 */
export type MCPServerStatus = 'disconnected' | 'connecting' | 'connected' | 'error'

/**
 * 服务器运行时状态
 */
export interface MCPServerState {
  config: MCPServerConfig
  status: MCPServerStatus
  error?: string
  tools?: MCPToolDefinition[]
  connected_at?: string
}

// ========== Tool Types ==========

/**
 * JSON Schema 属性
 */
export interface JSONSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object'
  description?: string
  enum?: unknown[]
  default?: unknown
  items?: JSONSchemaProperty
  properties?: Record<string, JSONSchemaProperty>
  required?: string[]
}

/**
 * JSON Schema 定义
 */
export interface JSONSchema {
  type: 'object' | 'string' | 'number' | 'integer' | 'boolean' | 'array'
  properties?: Record<string, JSONSchemaProperty>
  required?: string[]
  description?: string
}

/**
 * MCP 工具定义 (官方 2025-06-18 规范)
 */
export interface MCPToolDefinition {
  /** 工具名称 */
  name: string
  
  /** 工具描述 */
  description?: string
  
  /** 输入参数 Schema (JSON Schema) */
  inputSchema: JSONSchema
  
  /** 🆕 输出结果 Schema (JSON Schema) - 官方新增 */
  outputSchema?: JSONSchema
}

/**
 * 完整的 MCP 工具信息 (包含来源服务器)
 */
export interface MCPTool extends MCPToolDefinition {
  /** 所属 MCP 服务器 ID */
  serverId: string
  
  /** 所属 MCP 服务器名称 */
  serverName: string
}

// ========== Tool Execution ==========

/**
 * 工具执行请求
 */
export interface MCPToolExecuteRequest {
  server_id: string
  tool_name: string
  arguments: unknown
}

/**
 * 工具执行结果
 */
export interface MCPToolResult {
  /** 是否为流式响应 */
  is_streaming: boolean
  
  /** 执行是否成功 */
  success: boolean
  
  /** 执行结果 */
  result?: unknown
  
  /** 🆕 结构化内容 (如果工具定义了 outputSchema) */
  structured_content?: unknown
  
  /** 错误信息 */
  error?: string
  
  /** 执行耗时 (ms) */
  execution_time_ms?: number
}

/**
 * 流式响应数据块
 */
export interface MCPStreamChunk {
  /** 块类型 */
  type: 'progress' | 'partial' | 'complete' | 'error'
  
  /** 数据内容 */
  data: unknown
  
  /** 进度 (0-100) */
  progress?: number
  
  /** 时间戳 */
  timestamp?: number
}

// ========== Elicitation Types (官方新增) ==========

/**
 * Elicitation 请求 (服务器请求用户输入)
 * 官方 2025-06-18 新增能力
 */
export interface ElicitationRequest {
  /** 请求 ID */
  id: string
  
  /** 提示消息 */
  message: string
  
  /** 期望的输入格式 (JSON Schema) */
  schema: JSONSchema
  
  /** 请求的权限 (可选) */
  requestedPermission?: string
  
  /** 来源服务器 */
  serverId: string
}

/**
 * Elicitation 响应
 */
export interface ElicitationResponse {
  /** 请求 ID */
  requestId: string
  
  /** 用户操作 */
  action: 'approve' | 'deny'
  
  /** 用户输入数据 */
  data?: unknown
}

// ========== API Response Types ==========

export interface MCPServersResponse {
  servers: MCPServerState[]
}

export interface MCPToolsResponse {
  tools: MCPTool[]
}

export interface MCPTestConnectionResponse {
  success: boolean
  error?: string
}
