package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

// ========================
// Configuration constants for OpenWeb
// ========================

const (
	openWebMaxContentLength = 15000               // 最大内容长度（字符）
	openWebDefaultTimeout   = 10 * time.Second    // 默认超时时间
	openWebMaxTimeout       = 30 * time.Second    // 最大超时时间
	jinaReaderBaseURL       = "https://r.jina.ai" // Jina Reader API
)

// ========================
// Pre-compiled regex patterns for HTML cleaning
// ========================

var (
	// 需要移除的标签（Go regexp不支持反向引用，所以分别处理每个标签）
	scriptPattern   = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
	stylePattern    = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
	navPattern      = regexp.MustCompile(`(?is)<nav[^>]*>.*?</nav>`)
	headerPattern   = regexp.MustCompile(`(?is)<header[^>]*>.*?</header>`)
	footerPattern   = regexp.MustCompile(`(?is)<footer[^>]*>.*?</footer>`)
	noscriptPattern = regexp.MustCompile(`(?is)<noscript[^>]*>.*?</noscript>`)
	iframePattern   = regexp.MustCompile(`(?is)<iframe[^>]*>.*?</iframe>`)
	svgPattern      = regexp.MustCompile(`(?is)<svg[^>]*>.*?</svg>`)
	canvasPattern   = regexp.MustCompile(`(?is)<canvas[^>]*>.*?</canvas>`)
	// HTML注释
	htmlCommentPattern = regexp.MustCompile(`<!--[\s\S]*?-->`)
	// HTML标签（用于提取纯文本）
	htmlTagsPattern = regexp.MustCompile(`<[^>]*>`)
	// 多个空白字符
	multiWhitespacePattern = regexp.MustCompile(`\s+`)
	// 多个换行
	multiNewlinePattern = regexp.MustCompile(`\n{3,}`)
	// 标题提取
	titleTagPattern = regexp.MustCompile(`(?is)<title[^>]*>([^<]*)</title>`)
	h1TagPattern    = regexp.MustCompile(`(?is)<h1[^>]*>([^<]*)</h1>`)
)

// ========================
// Data structures
// ========================

// OpenWebRequest 网页抓取请求
type OpenWebRequest struct {
	URL       string `json:"url"`
	Timeout   int    `json:"timeout,omitempty"`    // 超时（秒）
	ForceJina bool   `json:"force_jina,omitempty"` // 强制使用Jina
}

// OpenWebResponse 网页抓取响应
type OpenWebResponse struct {
	URL       string `json:"url"`
	Title     string `json:"title"`
	Content   string `json:"content"`
	Length    int    `json:"length"`
	Truncated bool   `json:"truncated"`
	Method    string `json:"method"` // "direct" 或 "jina"
	Error     string `json:"error,omitempty"`
}

// ========================
// HTML Cleaning Functions
// ========================

// cleanHTMLContent 清洗HTML内容，提取纯文本
func cleanHTMLContent(html string) (title string, content string) {
	// 提取标题
	titleMatch := titleTagPattern.FindStringSubmatch(html)
	if len(titleMatch) > 1 {
		title = strings.TrimSpace(titleMatch[1])
	}
	if title == "" {
		h1Match := h1TagPattern.FindStringSubmatch(html)
		if len(h1Match) > 1 {
			title = strings.TrimSpace(h1Match[1])
		}
	}
	if title == "" {
		title = "Untitled"
	}

	// 移除script、style等标签及内容（Go regexp不支持反向引用，分别处理）
	content = scriptPattern.ReplaceAllString(html, "")
	content = stylePattern.ReplaceAllString(content, "")
	content = navPattern.ReplaceAllString(content, "")
	content = headerPattern.ReplaceAllString(content, "")
	content = footerPattern.ReplaceAllString(content, "")
	content = noscriptPattern.ReplaceAllString(content, "")
	content = iframePattern.ReplaceAllString(content, "")
	content = svgPattern.ReplaceAllString(content, "")
	content = canvasPattern.ReplaceAllString(content, "")

	// 移除HTML注释
	content = htmlCommentPattern.ReplaceAllString(content, "")

	// 移除所有HTML标签
	content = htmlTagsPattern.ReplaceAllString(content, " ")

	// 清理空白字符
	content = multiWhitespacePattern.ReplaceAllString(content, " ")
	content = strings.ReplaceAll(content, " \n", "\n")
	content = strings.ReplaceAll(content, "\n ", "\n")
	content = multiNewlinePattern.ReplaceAllString(content, "\n\n")
	content = strings.TrimSpace(content)

	return title, content
}

// ========================
// Fetch Functions
// ========================

// fetchDirect 直接抓取网页
func fetchWebDirect(targetURL string, timeout time.Duration) (*OpenWebResponse, error) {
	log.Printf("[OpenWeb:Direct] 开始抓取: %s", targetURL)
	startTime := time.Now()

	client := &http.Client{
		Timeout: timeout,
		// 不跟随重定向太多次
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			return nil
		},
	}

	req, err := http.NewRequest("GET", targetURL, nil)
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}

	// 设置请求头模拟浏览器
	req.Header.Set("User-Agent", getRandomUserAgent())
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8")
	req.Header.Set("DNT", "1")
	req.Header.Set("Connection", "keep-alive")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("请求超时")
		}
		return nil, fmt.Errorf("请求失败: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[OpenWeb:Direct] 响应状态: %d, 耗时: %v", resp.StatusCode, time.Since(startTime))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// 读取响应体
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	if len(body) == 0 {
		return nil, fmt.Errorf("响应内容为空")
	}

	log.Printf("[OpenWeb:Direct] 原始HTML长度: %d 字符", len(body))

	// 清洗HTML
	title, content := cleanHTMLContent(string(body))

	log.Printf("[OpenWeb:Direct] 清洗后内容长度: %d 字符", len(content))

	truncated := false
	if len(content) > openWebMaxContentLength {
		content = content[:openWebMaxContentLength] + "\n\n...[Content truncated]"
		truncated = true
		log.Printf("[OpenWeb:Direct] 内容已截断至 %d 字符", openWebMaxContentLength)
	}

	log.Printf("[OpenWeb:Direct] ✅ 抓取成功，总耗时: %v", time.Since(startTime))

	return &OpenWebResponse{
		URL:       targetURL,
		Title:     title,
		Content:   content,
		Length:    len(content),
		Truncated: truncated,
		Method:    "direct",
	}, nil
}

// fetchWithJina 使用Jina Reader API抓取
func fetchWebWithJina(targetURL string, timeout time.Duration) (*OpenWebResponse, error) {
	log.Printf("[OpenWeb:Jina] 开始抓取: %s", targetURL)
	startTime := time.Now()

	jinaURL := fmt.Sprintf("%s/%s", jinaReaderBaseURL, targetURL)
	log.Printf("[OpenWeb:Jina] Jina URL: %s", jinaURL)

	client := &http.Client{Timeout: timeout}

	req, err := http.NewRequest("GET", jinaURL, nil)
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}

	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-Timeout", fmt.Sprintf("%d", int(timeout.Seconds())))
	req.Header.Set("X-No-Cache", "true")

	resp, err := client.Do(req)
	if err != nil {
		if strings.Contains(err.Error(), "timeout") {
			return nil, fmt.Errorf("请求超时")
		}
		return nil, fmt.Errorf("请求失败: %w", err)
	}
	defer resp.Body.Close()

	log.Printf("[OpenWeb:Jina] 响应状态: %d, 耗时: %v", resp.StatusCode, time.Since(startTime))

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	// 解析JSON响应
	var result struct {
		Data struct {
			URL     string `json:"url"`
			Title   string `json:"title"`
			Content string `json:"content"`
		} `json:"data"`
		URL     string `json:"url"`
		Title   string `json:"title"`
		Content string `json:"content"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	// 优先使用data字段
	content := result.Data.Content
	if content == "" {
		content = result.Content
	}
	title := result.Data.Title
	if title == "" {
		title = result.Title
	}
	if title == "" {
		title = "Untitled"
	}
	actualURL := result.Data.URL
	if actualURL == "" {
		actualURL = result.URL
	}
	if actualURL == "" {
		actualURL = targetURL
	}

	if content == "" {
		return nil, fmt.Errorf("响应内容为空")
	}

	log.Printf("[OpenWeb:Jina] 获取标题: %s", title)
	log.Printf("[OpenWeb:Jina] 原始内容长度: %d 字符", len(content))

	truncated := false
	if len(content) > openWebMaxContentLength {
		content = content[:openWebMaxContentLength] + "\n\n...[Content truncated]"
		truncated = true
		log.Printf("[OpenWeb:Jina] 内容已截断至 %d 字符", openWebMaxContentLength)
	}

	log.Printf("[OpenWeb:Jina] ✅ 抓取成功，总耗时: %v", time.Since(startTime))

	return &OpenWebResponse{
		URL:       actualURL,
		Title:     title,
		Content:   content,
		Length:    len(content),
		Truncated: truncated,
		Method:    "jina",
	}, nil
}

// ========================
// HTTP Handler
// ========================

// OpenWebHandler 处理网页抓取请求
func OpenWebHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 设置CORS头
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

		// 处理预检请求
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// 只允许POST请求
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 解析请求
		var req OpenWebRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("[OpenWeb] 解析请求失败: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				Error: "无效的请求格式",
			})
			return
		}

		// 验证URL
		if req.URL == "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				Error: "URL不能为空",
			})
			return
		}

		// 验证URL格式
		parsedURL, err := url.Parse(req.URL)
		if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(OpenWebResponse{
				URL:   req.URL,
				Error: "无效的URL格式",
			})
			return
		}

		// 计算超时时间
		timeout := openWebDefaultTimeout
		if req.Timeout > 0 {
			timeout = time.Duration(req.Timeout) * time.Second
			if timeout > openWebMaxTimeout {
				timeout = openWebMaxTimeout
			}
		}

		log.Printf("[OpenWeb] 请求: url=%s, timeout=%v, force_jina=%v", req.URL, timeout, req.ForceJina)

		var result *OpenWebResponse
		var fetchErr error

		// 策略1：如果不强制使用Jina，优先直接抓取
		if !req.ForceJina {
			log.Printf("[OpenWeb] 策略1: 尝试直接抓取...")
			result, fetchErr = fetchWebDirect(req.URL, timeout)
			if fetchErr == nil {
				log.Printf("[OpenWeb] ✅ 直接抓取成功")
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(result)
				return
			}
			log.Printf("[OpenWeb] ⚠️ 直接抓取失败: %v", fetchErr)
			log.Printf("[OpenWeb] 策略2: 回退到Jina Reader...")
		} else {
			log.Printf("[OpenWeb] 跳过直接抓取，直接使用Jina Reader")
		}

		// 策略2：使用Jina Reader
		result, fetchErr = fetchWebWithJina(req.URL, timeout)
		if fetchErr == nil {
			log.Printf("[OpenWeb] ✅ Jina Reader抓取成功")
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(result)
			return
		}

		// 两种方法都失败
		log.Printf("[OpenWeb] ❌ 所有抓取方式都失败: %v", fetchErr)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(OpenWebResponse{
			URL:   req.URL,
			Error: fmt.Sprintf("无法获取网页内容: %v", fetchErr),
		})
	}
}
