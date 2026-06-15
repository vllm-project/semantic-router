<script lang="ts">
  import Embedding from '../embedding/Embedding.svelte';
  import Packing from '../packing/Packing.svelte';
  import Tooltip from '../Tooltip.svelte';
  import { getTooltipStore } from '../../stores';
  import logoDiffusionDB from '../../imgs/logo-diffusiondb.svg?raw';
  import iconGithub from '../../imgs/icon-github.svg?raw';

  let component: HTMLElement | null = null;
  let view = 'prompt-embedding';

  // Check for url hash (/#phrase)
  if (window.location.hash) {
    const hash = window.location.hash.slice(1);

    switch (hash) {
      case 'phrase': {
        view = 'phrase';
        break;
      }

      case 'prompt-embedding': {
        view = 'prompt-embedding';
        break;
      }

      case 'image-embedding': {
        view = 'image-embedding';
        break;
      }

      default: {
        break;
      }
    }
  }

  // Initialize the stores to pass to child components
  const tooltipStore = getTooltipStore();
</script>

<style lang="scss">
  @import './DiffusiondbVis.scss';
</style>

<div class="diffusiondbvis-page">
  <Tooltip {tooltipStore} />

  <div class="app-wrapper">
    <div class="app-title">
      <div class="title-left">
        <div class="app-icon">
          {@html logoDiffusionDB}
        </div>
        <span class="app-name"> DiffusionDB Explorer </span>
      </div>

      <div class="title-right">
        <a class="title-link" href="https://github.com/anonacl/diffusiondb">
          {@html iconGithub}
        </a>
      </div>
    </div>

    <div class="main-app" bind:this={component}>
      <div
        class="main-app-container"
        class:hidden={view !== 'prompt-embedding'}
      >
        <Embedding {tooltipStore} embeddingName={'prompt'} />
      </div>

      <div class="main-app-container" class:hidden={view !== 'image-embedding'}>
        <Embedding {tooltipStore} embeddingName={'image'} />
      </div>

      <div class="main-app-container" class:hidden={view !== 'phrase'}>
        <Packing {tooltipStore} />
      </div>
    </div>

    <div class="app-tabs">
      <button
        class="tab"
        class:selected={view === 'prompt-embedding'}
        on:click={() => {
          view = 'prompt-embedding';
        }}
        data-text="Prompt Embedding">Prompt Embedding</button
      >

      <span class="splitter" />

      <button
        class="tab"
        class:selected={view === 'image-embedding'}
        on:click={() => {
          view = 'image-embedding';
        }}
        data-text="Image Embedding">Image Embedding</button
      >

      <span class="splitter" />

      <button
        class="tab"
        class:selected={view === 'phrase'}
        on:click={() => {
          view = 'phrase';
        }}
        data-text="Popular Phrases">Popular Phrases</button
      >
    </div>
  </div>
</div>
