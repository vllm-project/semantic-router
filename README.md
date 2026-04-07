<div align="center">

<img src="website/static/img/artworks/vllm-sr-logo.dark.png" alt="vLLM Semantic Router" width="50%"/>

<p><strong>System Level Intelligent Router for Mixture-of-Models at Cloud, Data Center and Edge</strong></p>

<p>
  <a href="https://vllm-semantic-router.com">Documentation</a> |
  <a href="https://play.vllm-semantic-router.com">Playground</a> |
  <a href="https://vllm-semantic-router.com/blog/">Blog</a> |
  <a href="https://vllm-semantic-router.com/publications/">Publications</a> |
  <a href="https://huggingface.co/LLM-Semantic-Router">Hugging Face</a>
</p>

</div>

---

## About

In the LLM era, the number of models is exploding. Different models vary across capability, scale, cost, and privacy boundaries. Choosing and connecting the right models to build semantic AI infrastructure is a system problem.

vLLM Semantic Router is a signal-driven intelligent router for that problem. It helps teams build model systems that are more efficient, safer, and more adaptive across cloud, data center, and edge environments.

It delivers three core values:

- **Token economics**: reduce wasted tokens, increase effective output, and maximize the value of every token.
- **LLM safety**: detect jailbreaks, sensitive leakage, and hallucinations so agents remain controllable, trustworthy, and auditable.
- **Fullmesh intelligence**: build personal AI at the edge and intelligent MaaS in the cloud by coordinating local, private, and frontier models across cost, privacy, and capability boundaries.

## Getting Started

### Install

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

For platform notes, detailed setup options, and troubleshooting, see the **[Installation Guide](https://vllm-semantic-router.com/docs/installation/)**.

> [!IMPORTANT]
> Online [playground](https://play.vllm-semantic-router.com) default credentials:
>
> <!-- markdownlint-disable MD004 MD032 -->
> + username: `love@vllm-sr.ai`
> + password: `vllm-sr`
> <!-- markdownlint-enable MD004 MD032 -->

## Latest News

- [2026/03/24] Vision Paper Released: [The Workload-Router-Pool Architecture for LLM Inference Optimization](https://vllm-semantic-router.com/vision-paper)
- [2026/03/10] v0.2 Released: [vLLM Semantic Router v0.2 Athena Release](https://vllm.ai/blog/v0.2-vllm-sr-athena-release)
- [2026/02/27] White Paper Released: [Signal Driven Decision Routing for Mixture-of-Modality Models](https://vllm-semantic-router.com/white-paper/)
- [2026/01/05] Iris v0.1 Released: [vLLM Semantic Router v0.1 Iris: The First Major Release](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html)
- [2025/12/16] Collaboration: [AMD × vLLM Semantic Router: Building the System Intelligence Together](https://blog.vllm.ai/2025/12/16/vllm-sr-amd.html)
- [2025/11/19] New Blog: [Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale](https://blog.vllm.ai/2025/11/19/signal-decision.html)
- [2025/11/03] Paper Published: [Category-Aware Semantic Caching for Heterogeneous LLM Workloads](https://arxiv.org/abs/2510.26835)
- [2025/10/12] Paper Accepted: [When to Reason: Semantic Router for vLLM](https://arxiv.org/abs/2510.08731)

<details>
<summary>Earlier announcements</summary>

- [2025/12/15] New Blog: [Token-Level Truth: Real-Time Hallucination Detection for Production LLMs](https://blog.vllm.ai/2025/12/14/halugate.html)
- [2025/10/27] New Blog: [Scaling Semantic Routing with Extensible LoRA](https://blog.vllm.ai/2025/10/27/semantic-router-modular.html)
- [2025/10/08] Collaboration: vLLM Semantic Router with [vLLM Production Stack](https://github.com/vllm-project/production-stack) Team.
- [2025/09/01] Released the project: [vLLM Semantic Router: Next Phase in LLM inference](https://blog.vllm.ai/2025/09/11/semantic-router.html).

</details>

More announcements are available on the **[Blog](https://vllm-semantic-router.com/blog/)** and **[Publications](https://vllm-semantic-router.com/publications/)** pages.

## Community

For questions, feedback, or to contribute, please join the `#semantic-router` channel in vLLM Slack.

### Community Meetings

We host bi-weekly community meetings to sync with contributors across different time zones:

- **First Tuesday of the month**: 9:00-10:00 AM EST (accommodates US EST, EU, and Asia Pacific contributors)
  - [Zoom Link](https://us05web.zoom.us/j/84122485631?pwd=BB88v03mMNLVHn60YzVk4PihuqBV9d.1)
  - [Google Calendar Invite](https://us05web.zoom.us/meeting/tZAsdeuspj4sGdVraOOR4UaXSstrH2jjPYFq/calendar/google/add?meetingMasterEventId=4jjzUKSLSLiBHtIKZpGc3g)
  - [ics file](https://drive.google.com/file/d/15wO8cg0ZjNxdr8OtGiZyAgkSS8_Wry0J/view?usp=sharing)
- **Third Tuesday of the month**: 1:00-2:00 PM EST (accommodates US EST and California contributors)
  - [Zoom Link](https://us06web.zoom.us/j/86871492845?pwd=LcTtXm9gtGu23JeWqXxbnLLCCvbumB.1)
  - [Google Calendar Invite](https://us05web.zoom.us/meeting/tZIlcOispzkiHtH2dlkWlLym68bEqvuf3MU5/calendar/google/add?meetingMasterEventId=PqWz2vk7TOCszPXqconGAA)
  - [ics file](https://drive.google.com/file/d/1T54mwYpXXoV9QfR76I56BFBPNbykSsTw/view?usp=sharing)
- Meeting recordings: [YouTube](https://www.youtube.com/@vLLMSemanticRouter/videos)

## Contributing

If you want to contribute, start with **[CONTRIBUTING.md](CONTRIBUTING.md)**.

For repository-native development workflow and validation commands, use **[AGENTS.md](AGENTS.md)** as the entrypoint and **[docs/agent/README.md](docs/agent/README.md)** as the canonical index.

## Citation

If you find Semantic Router helpful in your research or projects, please consider citing it:

```
@misc{semanticrouter2025,
  title={vLLM Semantic Router},
  author={vLLM Semantic Router Team},
  year={2025},
  howpublished={\url{https://github.com/vllm-project/semantic-router}},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=vllm-project/semantic-router&type=Date)](https://www.star-history.com/#vllm-project/semantic-router&Date)

## Sponsors

We are grateful to our sponsors who support us:

---

[**AMD**](https://www.amd.com) provides us with GPU resources and [ROCm™](https://www.amd.com/en/products/software/rocm.html) software for training and researching frontier router models, enhancing E2E testing, and building the online models playground.

<div align="center">
<a href="https://www.amd.com">
  <img src="website/static/img/amd-logo.svg" alt="AMD" width="40%"/>
</a>
</div>

---
