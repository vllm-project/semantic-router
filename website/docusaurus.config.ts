import type { Config } from '@docusaurus/types'
import type * as Preset from '@docusaurus/preset-classic'
import { themes } from 'prism-react-renderer'
import { SITE_SOCIAL_PREVIEW_IMAGE } from './src/data/socialPreview'
import blogSearchIndexPlugin from './src/plugins/blogSearchIndex'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

const lightCodeTheme = themes.github
const darkCodeTheme = themes.vsDark
const siteUrl = 'https://vllm-sr.ai'
const siteDefaultDescription
  = 'We believe Mixture-of-Models is the next-generation model architecture for heterogeneous LLM inference. vLLM Semantic Router makes it executable.'
const siteSocialTitle
  = 'Mixture-of-Models for Heterogeneous LLM Inference | vLLM Semantic Router'
const siteSocialPreviewImageUrl = `${siteUrl}/${SITE_SOCIAL_PREVIEW_IMAGE}`

const config: Config = {
  title: 'vLLM Semantic Router',
  tagline: 'Building Mixture-of-Models: The Next-Generation Model Architecture for Heterogeneous LLM Inference',
  favicon: 'img/vllm.png',

  // Set the production url of your site here
  url: siteUrl,
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'vllm-project', // Usually your GitHub org/user name.
  projectName: 'semantic-router', // Usually your repo name.

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'zh-Hans'],
    localeConfigs: {
      'en': { label: 'English', htmlLang: 'en-US' },
      'zh-Hans': { label: '简体中文', htmlLang: 'zh-Hans' },
    },
  },

  markdown: {
    mermaid: true,
    format: 'mdx',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true, // cache-bust the index between deploys
        indexDocs: true,
        indexBlog: true,
        indexPages: false, // homepage/community are marketing pages, not docs
        docsRouteBasePath: '/docs',
        blogRouteBasePath: '/blog',
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        // v1 scope: index the current docs version only, per the decision on #2737.
        // To make archived versions searchable later, drop this and add
        // searchContextByPaths: ['docs', 'docs/v0.3', 'docs/v0.2', 'docs/v0.1']
        // so results stay scoped to the version the reader is on.
        // NOTE: ignoreFiles matches the route *without* a leading slash (the
        // plugin strips baseUrl, which is "/" here, off the front) and without
        // the base URL itself, so the pattern must not anchor on "/".
        ignoreFiles: [/^docs\/v\d+\.\d+\//],
        // language / styling are handled in later phases
      },
    ],
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          beforeDefaultRemarkPlugins: [[remarkMath, {}]],
          beforeDefaultRehypePlugins: [[rehypeKatex, {}]],
          sidebarPath: './sidebars.ts',
          lastVersion: 'current',
          versions: {
            'current': {
              label: 'Latest',
              path: '',
              badge: true,
            },
            'v0.3': {
              label: 'v0.3',
              path: 'v0.3',
              badge: true,
            },
            'v0.2': {
              label: 'v0.2',
              path: 'v0.2',
              badge: true,
            },
            'v0.1': {
              label: 'v0.1',
              path: 'v0.1',
              badge: true,
            },
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          // Custom editUrl function to always point to the "current" (main) version
          editUrl: ({ locale, docPath }) => {
            if (locale !== 'en') {
              return `https://github.com/vllm-project/semantic-router/edit/main/website/i18n/${locale}/docusaurus-plugin-content-docs/current/${docPath}`
            }
            return `https://github.com/vllm-project/semantic-router/edit/main/website/docs/${docPath}`
          },
        },
        blog: {
          showReadingTime: true,
          postsPerPage: 10,
          blogTitle: 'vLLM Semantic Router Blog',
          blogDescription:
            'Latest updates, insights, and technical articles about vLLM Semantic Router',
          blogSidebarTitle: 'Recent Posts',
          blogSidebarCount: 10,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/vllm-project/semantic-router/tree/main/website/blog/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          filename: 'sitemap.xml',
          ignorePatterns: ['/tags/**', '/search'],
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    // Publishes every published blog post as global data so the blog list page can
    // search the whole archive instead of just the posts on the current page.
    blogSearchIndexPlugin,
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          {
            from: '/docs/installation/kubernetes',
            to: '/docs/installation/k8s/ai-gateway',
          },
          {
            from: '/docs/cli/overview',
            to: '/docs/installation/',
          },
          {
            from: '/docs/cli/commands-reference',
            to: '/docs/installation/',
          },
          {
            from: '/docs/cli/troubleshooting',
            to: '/docs/troubleshooting/common-errors',
          },
          {
            from: '/docs/tutorials/signal/projections',
            to: '/docs/tutorials/projection/overview',
          },
          {
            from: '/docs/tutorials/signal/heuristic/modality',
            to: '/docs/tutorials/signal/learned/modality',
          },
        ],
      },
    ],
  ],

  themeConfig: {
    image: SITE_SOCIAL_PREVIEW_IMAGE,
    metadata: [
      { name: 'description', content: siteDefaultDescription },
      {
        name: 'keywords',
        content:
          'Mixture-of-Models, heterogeneous LLM inference, preference-driven AI, open-source LLM router, multi-model routing, model orchestration, model selection, model cascade, Fusion API, semantic router, vLLM',
      },
      { name: 'author', content: 'vLLM Semantic Router Team' },
      { name: 'application-name', content: 'vLLM Semantic Router' },
      { property: 'og:title', content: siteSocialTitle },
      { property: 'og:description', content: siteDefaultDescription },
      { property: 'og:type', content: 'website' },
      { property: 'og:site_name', content: 'vLLM Semantic Router' },
      { property: 'og:image', content: siteSocialPreviewImageUrl },
      {
        property: 'og:image:alt',
        content: 'vLLM Semantic Router social preview',
      },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: siteSocialTitle },
      { name: 'twitter:description', content: siteDefaultDescription },
      { name: 'twitter:image', content: siteSocialPreviewImageUrl },
      {
        name: 'twitter:image:alt',
        content: 'vLLM Semantic Router social preview',
      },

      // GEO metadata config
      { name: 'geo.region', content: 'US-CA' },
      { name: 'geo.placename', content: 'San Francisco' },
      { name: 'geo.position', content: '37.7749;-122.4194' },
      { name: 'ICBM', content: '37.7749, -122.4194' },
    ],
    navbar: {
      logo: {
        alt: 'vLLM Semantic Router Logo',
        src: 'img/vllm-sr-logo.white.png',
        srcDark: 'img/vllm-sr-logo.white.png',
      },
      items: [
        {
          type: 'localeDropdown',
          className: 'nav-locale',
          position: 'right',
        },
        {
          type: 'docsVersionDropdown',
          className: 'nav-docs-only',
          position: 'right',
          dropdownActiveClassDisabled: true,
        },
        {
          label: 'Docs',
          to: '/docs/intro',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Research',
          to: '/publications',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Blog',
          to: '/blog',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'Community',
          to: '/community/team',
          className: 'nav-primary',
          position: 'left',
        },
        {
          label: 'GitHub',
          href: 'https://github.com/vllm-project/semantic-router',
          className: 'nav-utility',
          position: 'right',
        },
        {
          label: 'Dashboard',
          href: 'https://app.vllm-sr.ai',
          className: 'nav-dashboard-cta',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Quick Start',
              to: '/docs/intro',
            },
            {
              label: 'Installation',
              to: '/docs/installation',
            },
            {
              label: 'Governance',
              to: '/community/governance',
            },
            {
              label: 'Contributing',
              to: '/community/contributing',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/vllm-project/semantic-router',
            },
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co/LLM-Semantic-Router',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/vllm-project/semantic-router/discussions',
            },
            {
              label: 'Leaderboard',
              to: '/community/contributors',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Publications',
              to: '/publications',
            },
            {
              label: 'White Paper',
              to: '/white-paper',
            },
            {
              label: 'Vision Paper',
              to: '/vision-paper',
            },

            {
              label: 'License',
              href: 'https://github.com/vllm-project/semantic-router/blob/main/LICENSE',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} vLLM Semantic Router Team.`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['bash', 'json', 'yaml', 'go', 'rust', 'python'],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
      respectPrefersColorScheme: false,
    },
  } satisfies Preset.ThemeConfig,
  headTags: [
    {
      tagName: 'script',
      attributes: { type: 'application/ld+json' },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        'name': 'vLLM Semantic Router',
        'applicationCategory': 'AIInfrastructure',
        'operatingSystem': 'Cross-platform',
        'description': siteDefaultDescription,
        'url': 'https://vllm-sr.ai',
        'publisher': {
          '@type': 'Organization',
          'name': 'vLLM Semantic Router Team',
          'url': 'https://github.com/vllm-project/semantic-router',
        },
      }),
    },
  ],
}

export default config
