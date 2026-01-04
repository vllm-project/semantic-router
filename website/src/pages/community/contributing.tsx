import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import styles from './community-page.module.css'

const Contributing: React.FC = () => {
  return (
    <Layout
      title="Contributing Guide"
      description="How to contribute to vLLM Semantic Router"
    >
      <div className={styles.container}>
        <header className={styles.header}>
          <h1><Translate id="contributing.title">Contributing to vLLM Semantic Router 🤝</Translate></h1>
          <p className={styles.subtitle}>
            <Translate id="contributing.subtitle">We welcome contributions from the community! Here's how you can help make vLLM Semantic Router better.</Translate>
          </p>
        </header>

        <main className={styles.main}>
          <section className={styles.section}>
            <h2>
              🎯
              <Translate id="contributing.waysToContribute">Ways to Contribute</Translate>
            </h2>
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
            <h2>
              📋
              <Translate id="contributing.process.title">Contribution Process</Translate>
            </h2>
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
                        <a href="/docs/installation">Install the local</a>
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
            <h2>
              ⚙️
              <Translate id="contributing.precommit.title">Precommit hooks</Translate>
            </h2>
            <p>The Semantic-router project provides a precommit hook to standardize the entire project, including Go, Python, Rust, Markdown, and spelling error checking.</p>
            <p>Although these measures may increase the difficulty of contributions, they are necessary. We are currently building a portable Docker precommit environment to reduce the difficulty of contributions, allowing you to focus on functional pull requests.</p>

            <div className={styles.card}>
              <h3>Manual</h3>

              <h4>Some Tips: </h4>
              <div className={styles.stepNumberTips}>
                <p>1. If the precommit check fails, don't worry. You can also get more information by executing "make help". </p>
                <p>2. For the pip installation tool, we recommend that you use venv for installation.</p>
                <p>3. We recommend installing pre-commit through Python's virtual environment.</p>
                <p>4. You can also directly submit the PR and let GitHub CI test it for you, but this will take a lot of time!</p>
              </div>

              <div className={styles.steps}>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4>Install precommit</h4>
                    <p>
                      Run
                      <code>pip install --user pre-commit</code>
                    </p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4>Install check tools</h4>
                    <div className={styles.stepNumberTips}>
                      <p>
                        1. Markdown:
                        <code>npm install -g markdownlint-cli</code>
                      </p>
                      <p>
                        2. Yaml:
                        <code>pip install --user yamllint</code>
                      </p>
                      <p>
                        3. CodeSpell:
                        <code>pip install --user codespell</code>
                      </p>
                      <p>
                        4. JavaScript:
                        <code>cd website && npm lint</code>
                      </p>
                      <p>
                        5. Shell: take Mac as an example, execute
                        <code>brew install shellcheck</code>
                      </p>
                    </div>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>3</span>
                  <div>
                    <h4>Install precommit to git</h4>
                    <p>
                      Run
                      <code>pre-commit install</code>
                      , then pre-commit installed at
                      <code>.git/hooks/pre-commit</code>
                    </p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>4</span>
                  <div>
                    <h4>Run</h4>
                    <p>
                      Run
                      <code>make precommit-check</code>
                      {' '}
                      to check.
                    </p>
                  </div>
                </div>

                <hr />

                <h3>Docker/Podman</h3>
                <p>From the above local running method, it can be seen that the process is very troublesome and complicated. Therefore, we have provided running methods based on Docker or Podman. There is no need to install various dependent software; all you need is a container runtime.</p>

                <h4>Some Tips: </h4>
                <div className={styles.stepNumberTips}>
                  <p>
                    Although Docker can help avoid installing too many detection tools locally, this does not mean that it will be automatically executed during the commit process. Therefore, when committing, you can use
                    <code>git commit -s -m -n</code>
                    {' '}
                    to skip the detection.
                    {' '}
                  </p>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4>Make sure Docker/Podman is installed</h4>
                    <p><code>docker --version</code></p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4>Run precommit by Docker/Podman</h4>
                    <p><code>make precommit-local</code></p>
                  </div>
                </div>
                <div>
                  <p>You can also manually enter the container and perform the operation:</p>
                  <div className={styles.codeBlock}>
                    <div className={styles.codeHeader}>
                      <div className={styles.windowControls}>
                        <span className={styles.controlButton}></span>
                        <span className={styles.controlButton}></span>
                        <span className={styles.controlButton}></span>
                      </div>
                      <div className={styles.title}>Manual Docker Setup</div>
                    </div>
                    <div className={styles.codeContent}>
                      <pre className={styles.codeText}>
                        {`# Set the container image
export PRECOMMIT_CONTAINER=ghcr.io/vllm-project/semantic-router/precommit:latest

# Run the container interactively
docker run --rm -it \\
     -v $(pwd):/app \\
     -w /app \\
     --name precommit-container \${PRECOMMIT_CONTAINER} \\
     bash

# Inside the container, run the precommit commands
pre-commit install && pre-commit run --all-files`}
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className={styles.section}>
            <h2>
              🏷️
              <Translate id="contributing.workGroups.title">Working Group Areas</Translate>
            </h2>
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
            <h2>
              📞
              <Translate id="contributing.getHelp.title">Get Help</Translate>
            </h2>
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

export default Contributing
