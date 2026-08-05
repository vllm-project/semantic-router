import styles from './PublicFooter.module.css'

const FOOTER_GROUPS = [
  {
    title: 'Product',
    links: [
      { label: 'Documentation', href: 'https://vllm-sr.ai/docs/intro/' },
      { label: 'Blog', href: 'https://vllm-sr.ai/blog/' },
      { label: 'Publications', href: 'https://vllm-sr.ai/publications/' },
    ],
  },
  {
    title: 'Build',
    links: [
      { label: 'GitHub', href: 'https://github.com/vllm-project/semantic-router' },
      { label: 'Hugging Face', href: 'https://huggingface.co/LLM-Semantic-Router' },
      { label: 'Installation', href: 'https://vllm-sr.ai/docs/installation/' },
    ],
  },
  {
    title: 'Community',
    links: [
      { label: 'Slack', href: 'https://slack.vllm.ai/' },
      {
        label: 'Working Groups',
        href: 'https://vllm-sr.ai/community/work-groups',
      },
      {
        label: 'Contributing',
        href: 'https://vllm-sr.ai/community/contributing',
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
            <p>Mixture-of-Models for heterogeneous LLM inference.</p>
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
          <span>Open source. Preference-driven.</span>
          <span>© {new Date().getFullYear()} vLLM Semantic Router</span>
        </div>
      </div>
    </footer>
  )
}
