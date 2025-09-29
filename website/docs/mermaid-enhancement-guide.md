# 可放大 Mermaid 图表组件使用指南

## 概述

`ZoomableMermaid` 组件为 Docusaurus 网站提供了可点击放大的 Mermaid 图表功能。用户可以点击图表在模态框中查看放大版本，提供更好的阅读体验。

## 使用方法

### 基本用法

在任何 Markdown 文档中，首先导入组件：

```jsx
import ZoomableMermaid from '@site/src/components/ZoomableMermaid';
```

然后使用组件包装 Mermaid 图表代码：

```jsx
<ZoomableMermaid title="系统架构图">
{`graph TB
    A[开始] --> B[处理]
    B --> C[结束]`}
</ZoomableMermaid>
```

### 参数说明

- `children` (string, 必需): Mermaid 图表的代码字符串
- `title` (string, 可选): 图表标题，会显示在模态框顶部

### 功能特性

- **点击放大**: 点击图表在模态框中查看放大版本
- **键盘支持**: 支持 Enter 和空格键打开模态框，Escape 键关闭
- **无障碍访问**: 完整的 ARIA 属性支持
- **响应式设计**: 在各种屏幕尺寸下都能正常工作
- **主题支持**: 自动适应明暗主题
- **动画效果**: 平滑的打开/关闭动画

### 示例

<ZoomableMermaid title="简单流程图">
{`graph LR
    A[用户访问] --> B{检查权限}
    B -->|有权限| C[显示内容]
    B -->|无权限| D[显示错误]`}
</ZoomableMermaid>

## 实现原理

组件基于以下技术栈：

1. **React Hooks**: 使用 `useState`、`useRef` 和 `useEffect` 管理状态
2. **CSS Modules**: 样式隔离和主题支持
3. **事件处理**: 键盘、鼠标和触摸事件
4. **无障碍访问**: ARIA 属性和焦点管理

## 注意事项

- 确保 Mermaid 代码字符串使用模板字符串语法 (backticks)
- 在 JSX 中使用时，需要将代码包装在 `{``}` 中
- 组件会自动处理焦点管理和页面滚动