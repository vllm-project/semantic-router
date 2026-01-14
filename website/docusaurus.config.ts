import type { Config } from '@docusaurus/types'
import type * as Preset from '@docusaurus/preset-classic'
import { themes } from 'prism-react-renderer'

const lightCodeTheme = themes.github
const darkCodeTheme = themes.vsDark

const config: Config = {
  title: 'vLLM Semantic Router',
  tagline: 'System Level Intelligent Router for Mixture-of-Models',
  favicon: 'img/vllm.png',

  // Set the production url of your site here
  url: 'https://vllm-semantic-router.com',
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
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          versions: {
            'current': {
              label: '🚧 Next',
              path: 'next',
              badge: true,
            },
            'v0.1': {
              label: 'v0.1',
              path: '',
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
          blogDescription: 'Latest updates, insights, and technical articles about vLLM Semantic Router',
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
        ],
      },
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    metadata: [
      { name: 'description', content: 'System Level Intelligent Router for Mixture-of-Models' },
      { name: 'keywords', content: 'LLM, Semantic Router, Mixture of Models, vLLM, Routing, AI Gateway, Envoy, ExtProc' },
      { name: 'author', content: 'vLLM Semantic Router Team' },
      { property: 'og:title', content: 'vLLM Semantic Router' },
      { property: 'og:description', content: 'System Level Intelligent Router for Mixture-of-Models' },
      { property: 'og:type', content: 'website' },
      { property: 'og:site_name', content: 'vLLM Semantic Router' },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: 'vLLM Semantic Router' },
      { name: 'twitter:description', content: 'System Level Intelligent Router for Mixture-of-Models' },

      // GEO metadata config
      { name: 'geo.region', content: 'US-CA' },
      { name: 'geo.placename', content: 'San Francisco' },
      { name: 'geo.position', content: '37.7749;-122.4194' },
      { name: 'ICBM', content: '37.7749, -122.4194' },
    ],
    navbar: {
      title: 'vLLM Semantic Router',
      logo: {
        alt: 'vLLM Semantic Router Logo',
        src: 'img/vllm.png',
        srcDark: 'img/vllm.png',
      },
      items: [
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          type: 'docsVersionDropdown',
          position: 'right',
          dropdownActiveClassDisabled: true,
        },
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          to: '/publications',
          label: 'Publications',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },

        {
          type: 'dropdown',
          label: 'Community',
          position: 'left',
          items: [
            {
              label: 'vLLM-SR Team',
              to: '/community/team',
            },
            {
              label: 'Work Groups',
              to: '/community/work-groups',
            },
            {
              label: 'Membership Promotion',
              to: '/community/promotion',
            },
            {
              label: 'Contributing Guide',
              to: '/community/contributing',
            },
            {
              label: 'Code of Conduct',
              to: '/community/code-of-conduct',
            },

            {
              type: 'html',
              value: '<hr style="margin: 0.3rem 0;">',
            },
            {
              label: 'GitHub Issues',
              href: 'https://github.com/vllm-project/semantic-router/issues',
            },
          ],
        },
        {
          href: 'https://github.com/vllm-project/semantic-router',
          className: 'header-github-link',
          position: 'right',
        },
        {
          href: 'https://huggingface.co/LLM-Semantic-Router',
          className: 'header-hf-link',
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
              label: 'Tutorials',
              to: '/docs/tutorials/intelligent-route/embedding-routing',
            },
            {
              label: 'API Reference',
              to: '/docs/api/router',
            },
            {
              label: 'CRD Reference',
              to: '/docs/api/crd-reference',
            },
            {
              label: 'Troubleshooting',
              to: '/docs/troubleshooting/common-errors',
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
              label: 'Team',
              to: '/community/team',
            },
            {
              label: 'Contributing',
              to: '/community/contributing',
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
              label: 'License',
              href: 'https://github.com/vllm-project/semantic-router/blob/main/LICENSE',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} vLLM Semantic Router Team. Built with Docusaurus.`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['bash', 'json', 'yaml', 'go', 'rust', 'python'],
    },
    colorMode: {
      defaultMode: 'light',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
  } satisfies Preset.ThemeConfig,
  headTags: [
    {
      tagName: 'link',
      attributes: {
        rel: 'alternate',
        hreflang: 'en',
        href: 'https://vllm-semantic-router.com/',
      },
    },
    {
      tagName: 'link',
      attributes: {
        rel: 'alternate',
        hreflang: 'zh-Hans',
        href: 'https://vllm-semantic-router.com/zh-Hans/',
      },
    },
    {
      tagName: 'script',
      attributes: { type: 'application/ld+json' },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        'name': 'vLLM Semantic Router',
        'applicationCategory': 'AIInfrastructure',
        'operatingSystem': 'Cross-platform',
        'description': 'System Level Intelligent Router for Mixture-of-Models',
        'url': 'https://vllm-semantic-router.com',
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
