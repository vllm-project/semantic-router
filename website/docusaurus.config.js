// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const { themes } = require('prism-react-renderer')
const lightCodeTheme = themes.github
const darkCodeTheme = themes.vsDark

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'vLLM Semantic Router',
  tagline: 'Intelligent Auto Reasoning Router for Efficient LLM Inference on Mixture-of-Models',
  favicon: 'img/vllm.png',

  // Set the production url of your site here
  url: 'https://your-docusaurus-test-site.com',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'vllm-project', // Usually your GitHub org/user name.
  projectName: 'semantic-router', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internalization, you can use this field to set useful
  // metadata like html lang. For example, if your site is Chinese, you may want
  // to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/vllm-project/semantic-router/tree/main/docs/',
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
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'vLLM Semantic Router',
        logo: {
          alt: 'vLLM Semantic Router Logo',
          src: 'img/vllm.png',
          srcDark: 'img/vllm.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
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
                label: 'Team',
                to: '/community/team',
              },
              {
                label: 'Work Groups',
                to: '/community/work-groups',
              },
              {
                label: 'Promotion',
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
                label: 'GitHub Discussions',
                href: 'https://github.com/vllm-project/semantic-router/discussions',
              },
              {
                label: 'GitHub Issues',
                href: 'https://github.com/vllm-project/semantic-router/issues',
              },
            ],
          },
          {
            type: 'dropdown',
            label: 'Roadmap',
            position: 'left',
            items: [
              {
                label: 'v0.1',
                to: '/roadmap/v0.1',
              },
            ],
          },
          {
            href: 'https://github.com/vllm-project/semantic-router',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://huggingface.co/LLM-Semantic-Router',
            label: '🤗 Hugging Face',
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
                label: 'Getting Started',
                to: '/docs/getting-started/installation',
              },
              {
                label: 'Architecture',
                to: '/docs/architecture/system-architecture',
              },
              {
                label: 'API Reference',
                to: '/docs/api/router',
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
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'License',
                href: 'https://github.com/vllm-project/semantic-router/blob/main/LICENSE',
              },
              {
                label: 'Contributing',
                href: 'https://github.com/vllm-project/semantic-router/blob/main/CONTRIBUTING.md',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} vLLM Semantic Router Team. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
        additionalLanguages: ['bash', 'json', 'yaml', 'go', 'rust'],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
    }),
}

module.exports = config
