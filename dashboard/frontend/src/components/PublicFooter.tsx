import styles from './PublicFooter.module.css'

const FOOTER_GROUPS = [
  {
    title: 'Product',
    links: [
      { label: 'Documentation', href: 'https://vllm-semantic-router.com/docs/intro/' },
      { label: 'Blog', href: 'https://vllm-semantic-router.com/blog/' },
      { label: 'Publications', href: 'https://vllm-semantic-router.com/publications/' },
    ],
  },
  {
    title: 'Build',
    links: [
      { label: 'GitHub', href: 'https://github.com/vllm-project/semantic-router' },
      { label: 'Hugging Face', href: 'https://huggingface.co/LLM-Semantic-Router' },
      { label: 'Installation', href: 'https://vllm-semantic-router.com/docs/installation/' },
    ],
  },
  {
    title: 'Community',
    links: [
      { label: 'Slack', href: 'https://slack.vllm.ai/' },
      {
        label: 'Working Groups',
        href: 'https://vllm-semantic-router.com/community/work-groups',
      },
      {
        label: 'Contributing',
        href: 'https://vllm-semantic-router.com/community/contributing',
      },
    ],
  },
] as const

export default function PublicFooter() {
  return (
    <footer className={styles.footer} data-testid="public-footer">
      <div className={styles.inner}>
        <div className={styles.grid}>
          <div className={styles.intro}>
            <div className={styles.brand}>
              <img className={styles.logo} src="/vllm.png" alt="" aria-hidden="true" />
              <span>vLLM Semantic Router</span>
            </div>
            <p>Building Mixture-of-Models—the next-generation model architecture.</p>
          </div>

          {FOOTER_GROUPS.map((group) => (
            <section key={group.title} className={styles.group} data-footer-group={group.title}>
              <h2>{group.title}</h2>
              <ul>
                {group.links.map((link) => (
                  <li key={link.label}>
                    <a href={link.href} target="_blank" rel="noopener noreferrer">
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>

        <div className={styles.bottomLine}>
          <span>Open source. Built around preference.</span>
          <span>© {new Date().getFullYear()} vLLM Semantic Router</span>
        </div>
      </div>
    </footer>
  )
}
