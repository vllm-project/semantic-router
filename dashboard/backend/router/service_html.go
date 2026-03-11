package router

import (
	"net/http"
	"strings"
)

const serviceNotConfiguredTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{SERVICE_NAME}} Not Configured</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 40px;
            max-width: 480px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        .icon {
            width: 64px;
            height: 64px;
            margin: 0 auto 24px;
            background: rgba(245, 158, 11, 0.15);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .icon svg {
            width: 32px;
            height: 32px;
            stroke: #f59e0b;
        }
        h1 {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        p {
            color: #a0a0a0;
            font-size: 14px;
            line-height: 1.6;
            margin-bottom: 24px;
        }
        .config-box {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            text-align: left;
        }
        .config-box h2 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .config-box .hint {
            font-size: 13px;
            color: #808080;
            margin-bottom: 12px;
        }
        code {
            display: block;
            background: rgba(0, 0, 0, 0.3);
            padding: 10px 14px;
            border-radius: 6px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            color: #60a5fa;
            word-break: break-all;
        }
        .example {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed rgba(255, 255, 255, 0.1);
        }
        .example-label {
            font-size: 12px;
            color: #606060;
            margin-bottom: 6px;
        }
        .example code {
            font-size: 12px;
            color: #808080;
        }
        .docs-link {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            margin-top: 24px;
            padding: 12px 24px;
            background: rgba(96, 165, 250, 0.1);
            border: 1px solid rgba(96, 165, 250, 0.3);
            border-radius: 8px;
            color: #60a5fa;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .docs-link:hover {
            background: rgba(96, 165, 250, 0.2);
            border-color: rgba(96, 165, 250, 0.5);
            transform: translateY(-2px);
        }
        .docs-link svg {
            width: 16px;
            height: 16px;
        }
    </style>
</head>
<body>
    <div class="card">
        <div class="icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
        </div>
        <h1>{{SERVICE_NAME}} Not Configured</h1>
        <p>{{SERVICE_NAME}} is not configured for this dashboard. Please set the required environment variable to enable this service.</p>
        <div class="config-box">
            <h2>Configuration Required</h2>
            <p class="hint">Set the following environment variable:</p>
            <code>{{ENV_VAR}}</code>
            <div class="example">
                <p class="example-label">Example:</p>
                <code>{{ENV_VAR}}={{EXAMPLE_VALUE}}</code>
            </div>
        </div>
        <a href="https://vllm-semantic-router.com/docs/tutorials/observability/dashboard" target="_blank" rel="noopener noreferrer" class="docs-link">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>
                <polyline points="15 3 21 3 21 9"/>
                <line x1="10" y1="14" x2="21" y2="3"/>
            </svg>
            View Documentation
        </a>
    </div>
</body>
</html>`

func serviceNotConfiguredHTML(serviceName, envVar, exampleValue string) string {
	replacer := strings.NewReplacer(
		"{{SERVICE_NAME}}", serviceName,
		"{{ENV_VAR}}", envVar,
		"{{EXAMPLE_VALUE}}", exampleValue,
	)
	return replacer.Replace(serviceNotConfiguredTemplate)
}

func serviceUnavailableHTMLHandler(serviceName, envVar, exampleValue string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(serviceNotConfiguredHTML(serviceName, envVar, exampleValue)))
	}
}
