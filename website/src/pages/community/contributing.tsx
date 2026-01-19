import React from 'react'
import Layout from '@theme/Layout'
import Translate from '@docusaurus/Translate'
import Link from '@docusaurus/Link'
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
                <h3><Translate id="contributing.bugReports.title">🐛 Bug Reports</Translate></h3>
                <p>
                  <Translate id="contributing.bugReports.desc.prefix">Found a bug? Please report it on our</Translate>
                  {' '}
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  .
                </p>
                <ul>
                  <li>
                    <Translate id="contributing.bugReports.list.1">Use a clear and descriptive title</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.bugReports.list.2">Provide steps to reproduce</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.bugReports.list.3">Include system information</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.bugReports.list.4">Add relevant logs or error messages</Translate>
                  </li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3><Translate id="contributing.featureRequests.title">✨ Feature Requests</Translate></h3>
                <p><Translate id="contributing.featureRequests.desc">Have an idea for a new feature? We'd love to hear it!</Translate></p>
                <ul>
                  <li>
                    <Translate id="contributing.featureRequests.list.1">Check existing issues first</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.featureRequests.list.2">Describe the problem you're solving</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.featureRequests.list.3">Explain your proposed solution</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.featureRequests.list.4">Consider implementation complexity</Translate>
                  </li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3><Translate id="contributing.documentation.title">📝 Documentation</Translate></h3>
                <p><Translate id="contributing.documentation.desc">Help improve our documentation and examples.</Translate></p>
                <ul>
                  <li>
                    <Translate id="contributing.documentation.list.1">Fix typos and grammar</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.documentation.list.2">Add missing documentation</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.documentation.list.3">Create tutorials and guides</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.documentation.list.4">Improve code examples</Translate>
                  </li>
                </ul>
              </div>

              <div className={styles.card}>
                <h3><Translate id="contributing.code.title">💻 Code Contributions</Translate></h3>
                <p><Translate id="contributing.code.desc">Contribute to the core functionality.</Translate></p>
                <ul>
                  <li>
                    <Translate id="contributing.code.list.1">Fix bugs and issues</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.code.list.2">Implement new features</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.code.list.3">Optimize performance</Translate>
                  </li>
                  <li>
                    <Translate id="contributing.code.list.4">Add test coverage</Translate>
                  </li>
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
                    <h4><Translate id="contributing.process.step1.title">Create an Issue</Translate></h4>
                    <p><Translate id="contributing.process.step1.desc">Discuss your idea or bug report with the community first.</Translate></p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4><Translate id="contributing.process.step2.title">Fork & Branch</Translate></h4>
                    <p><Translate id="contributing.process.step2.desc">Create a new branch for your changes from the main branch.</Translate></p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>3</span>
                  <div>
                    <h4><Translate id="contributing.process.step3.title">Make Changes</Translate></h4>
                    <p><Translate id="contributing.process.step3.desc">Implement your changes following our coding standards.</Translate></p>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>4</span>
                  <div>
                    <h4><Translate id="contributing.process.step4.title">Test</Translate></h4>
                    <p><Translate id="contributing.process.step4.desc">Run tests and ensure your changes don't break existing functionality.</Translate></p>
                    <div className={styles.stepNumberTips}>
                      <p><Translate id="contributing.process.step4.tip1">1. Run precommit hooks, ensure compliance with the project submission guidelines;</Translate></p>
                      <p>
                        <Translate id="contributing.process.step4.tip2.prefix">2. You can refer to</Translate>
                        {' '}
                        <Link to="/docs/installation"><Translate id="contributing.process.step4.tip2.link">Install the local</Translate></Link>
                        {' '}
                        <Translate id="contributing.process.step4.tip2.suffix">to start semantic-router locally.</Translate>
                      </p>
                    </div>
                  </div>
                </div>

                <div className={styles.step}>
                  <span className={styles.stepNumber}>5</span>
                  <div>
                    <h4><Translate id="contributing.process.step5.title">Submit PR</Translate></h4>
                    <p><Translate id="contributing.process.step5.desc">Create a pull request with a clear description of your changes.</Translate></p>
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
            <p><Translate id="contributing.precommit.desc1">The Semantic-router project provides a precommit hook to standardize the entire project, including Go, Python, Rust, Markdown, and spelling error checking.</Translate></p>
            <p><Translate id="contributing.precommit.desc2">Although these measures may increase the difficulty of contributions, they are necessary. We are currently building a portable Docker precommit environment to reduce the difficulty of contributions, allowing you to focus on functional pull requests.</Translate></p>

            <div className={styles.card}>
              <h3><Translate id="contributing.precommit.manual.title">Manual</Translate></h3>

              <h4><Translate id="contributing.precommit.tips.title">Some Tips: </Translate></h4>
              <div className={styles.stepNumberTips}>
                <p><Translate id="contributing.precommit.tips.1">1. If the precommit check fails, don't worry. You can also get more information by executing "make help". </Translate></p>
                <p><Translate id="contributing.precommit.tips.2">2. For the pip installation tool, we recommend that you use venv for installation.</Translate></p>
                <p><Translate id="contributing.precommit.tips.3">3. We recommend installing pre-commit through Python's virtual environment.</Translate></p>
                <p><Translate id="contributing.precommit.tips.4">4. You can also directly submit the PR and let GitHub CI test it for you, but this will take a lot of time!</Translate></p>
              </div>

              <div className={styles.steps}>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4><Translate id="contributing.precommit.step1.title">Install precommit</Translate></h4>
                    <p>
                      <Translate id="contributing.precommit.step1.desc">Run</Translate>
                      <code>pip install --user pre-commit</code>
                    </p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4><Translate id="contributing.precommit.step2.title">Install check tools</Translate></h4>
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
                        <Translate id="contributing.precommit.step2.shell">5. Shell: take Mac as an example, execute</Translate>
                        <code>brew install shellcheck</code>
                      </p>
                    </div>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>3</span>
                  <div>
                    <h4><Translate id="contributing.precommit.step3.title">Install precommit to git</Translate></h4>
                    <p>
                      <Translate id="contributing.precommit.step3.desc">Run</Translate>
                      <code>pre-commit install</code>
                      ,
                      <Translate id="contributing.precommit.step3.then">then pre-commit installed at</Translate>
                      {' '}
                      <code>.git/hooks/pre-commit</code>
                    </p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>4</span>
                  <div>
                    <h4><Translate id="contributing.precommit.step4.title">Run</Translate></h4>
                    <p>
                      <Translate id="contributing.precommit.step4.desc">Run</Translate>
                      <code>make precommit-check</code>
                      {' '}
                      <Translate id="contributing.precommit.step4.check">to check.</Translate>
                    </p>
                  </div>
                </div>

                <hr />

                <h3>Docker/Podman</h3>
                <p><Translate id="contributing.docker.desc">From the above local running method, it can be seen that the process is very troublesome and complicated. Therefore, we have provided running methods based on Docker or Podman. There is no need to install various dependent software; all you need is a container runtime.</Translate></p>

                <h4><Translate id="contributing.precommit.tips.title">Some Tips: </Translate></h4>
                <div className={styles.stepNumberTips}>
                  <p>
                    <Translate id="contributing.docker.tips.1">Although Docker can help avoid installing too many detection tools locally, this does not mean that it will be automatically executed during the commit process. Therefore, when committing, you can use</Translate>
                    <code>git commit -s -m -n</code>
                    {' '}
                    <Translate id="contributing.docker.tips.2">to skip the detection.</Translate>
                  </p>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>1</span>
                  <div>
                    <h4><Translate id="contributing.docker.step1.title">Make sure Docker/Podman is installed</Translate></h4>
                    <p><code>docker --version</code></p>
                  </div>
                </div>
                <div className={styles.step}>
                  <span className={styles.stepNumber}>2</span>
                  <div>
                    <h4><Translate id="contributing.docker.step2.title">Run precommit by Docker/Podman</Translate></h4>
                    <p><code>make precommit-local</code></p>
                  </div>
                </div>
                <div>
                  <p><Translate id="contributing.docker.manual.desc">You can also manually enter the container and perform the operation:</Translate></p>
                  <div className={styles.codeBlock}>
                    <div className={styles.codeHeader}>
                      <div className={styles.windowControls}>
                        <span className={styles.controlButton}></span>
                        <span className={styles.controlButton}></span>
                        <span className={styles.controlButton}></span>
                      </div>
                      <div className={styles.title}><Translate id="contributing.docker.manual.title">Manual Docker Setup</Translate></div>
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
              <Translate id="contributing.workGroups.desc.prefix">Consider joining one of our</Translate>
              {' '}
              <Link to="/community/work-groups"><Translate id="contributing.workGroups.link">Working Groups</Translate></Link>
              {' '}
              <Translate id="contributing.workGroups.desc.suffix">to focus your contributions:</Translate>
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
              <p><Translate id="contributing.getHelp.desc">Need help with your contribution? Reach out to us:</Translate></p>
              <ul>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/discussions" target="_blank" rel="noopener noreferrer">GitHub Discussions</a>
                  {' '}
                  -
                  {' '}
                  <Translate id="contributing.getHelp.discussions">For general questions and discussions</Translate>
                </li>
                <li>
                  <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub Issues</a>
                  {' '}
                  -
                  {' '}
                  <Translate id="contributing.getHelp.issues">For bug reports and feature requests</Translate>
                </li>
                <li>
                  <Link to="/community/work-groups"><Translate id="contributing.getHelp.workGroups.link">Work Groups</Translate></Link>
                  {' '}
                  -
                  {' '}
                  <Translate id="contributing.getHelp.workGroups">Join a specific working group</Translate>
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
