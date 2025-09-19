import React from 'react'
import Layout from '@theme/Layout'
import styles from './community-page.module.css'

export default function Contributing() {
  return (
    <Layout
      title="Contributing Guide"
      description="How to contribute to vLLM Semantic Router"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1>Contributing to vLLM Semantic Router 🤝</h1>
          <p className={styles.subtitle}>
            We welcome contributions from the community! Here's how you can help make vLLM Semantic Router better.
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>🎯 Ways to Contribute</h2>
            <div className={styles.contributeGrid}>
              <div className={styles.card}>
                <h3>🐛 Bug Reports</h3>
                <p>
                  Found a bug? Please report it on our
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  .
                </p>
                <ul>
                  <li>Use a clear and descriptive title</li>
                  <li>Provide steps to reproduce</li>
                  <li>Include system information</li>
                  <li>Add relevant logs or error messages</li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3>✨ Feature Requests</h3>
                <p>Have an idea for a new feature? We'd love to hear it!</p>
                <ul>
                  <li>Check existing issues first</li>
                  <li>Describe the problem you're solving</li>
                  <li>Explain your proposed solution</li>
                  <li>Consider implementation complexity</li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3>📝 Documentation</h3>
                <p>Help improve our documentation and examples.</p>
                <ul>
                  <li>Fix typos and grammar</li>
                  <li>Add missing documentation</li>
                  <li>Create tutorials and guides</li>
                  <li>Improve code examples</li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3>💻 Code Contributions</h3>
                <p>Contribute to the core functionality.</p>
                <ul>
                  <li>Fix bugs and issues</li>
                  <li>Implement new features</li>
                  <li>Optimize performance</li>
                  <li>Add test coverage</li>
                </ul>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>📋 Contribution Process</h2>
            <div className={styles.card}>
              <div className={styles.steps}>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4>Create an Issue</h4>
                    <p>Discuss your idea or bug report with the community first.</p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4>Fork & Branch</h4>
                    <p>Create a new branch for your changes from the main branch.</p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>3</span>
                  <div>
                    <h4>Make Changes</h4>
                    <p>Implement your changes following our coding standards.</p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>4</span>
                  <div>
                    <h4>Test</h4>
                    <p>Run tests and ensure your changes don't break existing functionality.</p>
                    <div className={styles.stepNumberTips}>
                      <p>1. Run precommit hooks, ensure compliance with the project submission guidelines;</p>
                      <p>
                        2. You can refer to
                        {' '}
                        <a href="/docs/getting-started/installation">Install the local</a>
                        {' '}
                        to start semantic-router locally.
                      </p>
                    </div>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>5</span>
                  <div>
                    <h4>Submit PR</h4>
                    <p>Create a pull request with a clear description of your changes.</p>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>⚙️ Precommit hooks</h2>
            <p>The Semantic-router project provides a precommit hook to standardize the entire project, including Go, Python, Rust, Markdown, and spelling error checking.</p>
            <p>Although these measures may increase the difficulty of contributions, they are necessary. We are currently building a portable Docker precommit environment to reduce the difficulty of contributions, allowing you to focus on functional pull requests.</p>

            <div className={styles.card}>
              <h3>Manual</h3>

              <h4>Some Tips: </h4>
              <div className={styles.stepNumberTips}>
                <p>1. If the precommit check fails, don't worry. You can also get more information by executing "make help". </p>
                <p>2. For the pip installation tool, we recommend that you use venv for installation.</p>
                <p>3. You can also directly submit the PR and let GitHub CI test it for you, but this will take a lot of time!</p>
              </div>

              <div className={styles.steps}>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4>Install precommit</h4>
                    <p>Run pip install --user precommit</p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4>Install check tools</h4>
                    <div className={styles.stepNumberTips}>
                      <p>Markdown: npm install -g markdownlint-cli</p>
                      <p>Yaml: pip install --user yamllint</p>
                      <p>CodeSpell: pip install --user codespell</p>
                      <p>JavaScript: cd website && npm lint</p>
                    </div>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>3</span>
                  <div>
                    <h4>Install precommit to git</h4>
                    <p>Run pre-commit install, then pre-commit installed at .git/hooks/pre-commit</p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>4</span>
                  <div>
                    <h4>Run</h4>
                    <p>Run make precommit-check to check.</p>
                  </div>
                </div>

                <hr />

                <h3>Docker</h3>
                <p>Coming soon!</p>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>🏷️ Working Group Areas</h2>
            <p>
              Consider joining one of our
              {' '}
              <a href="/community/work-groups">Working Groups</a>
              {' '}
              to focus your contributions:
            </p>
            <div className={styles.tagGrid}>
              <span className={styles.tag}>area/document</span>
              <span className={styles.tag}>area/environment</span>
              <span className={styles.tag}>area/core</span>
              <span className={styles.tag}>area/networking</span>
              <span className={styles.tag}>area/benchmark</span>
              <span className={styles.tag}>area/tooling</span>
              <span className={styles.tag}>area/user-experience</span>
            </div>
          </section>

          <section className={styles.section}>
            <h2>📞 Get Help</h2>
            <div className={styles.card}>
              <p>Need help with your contribution? Reach out to us:</p>
              <ul>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/discussions" target="_blank" rel="noopener noreferrer">GitHub Discussions</a>
                  {' '}
                  - For general questions and discussions
                </li>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  {' '}
                  - For bug reports and feature requests
                </li>
                <li>
                  <a href="/community/work-groups">Work Groups</a>
                  {' '}
                  - Join a specific working group
                </li>
              </ul>
            </div>
          </section>
        </main>
      </div>
    </Layout>
  )
}
