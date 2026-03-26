<script lang="ts">
  import type { DataURLs } from '../../types/embedding-types';
  import Embedding from '../embedding/Embedding.svelte';
  import Footer from '../footer/Footer.svelte';
  import SearchPanel from '../search-panel/SearchPanel.svelte';
  import { getFooterStore, getSearchBarStore } from '../../stores';

  let component: HTMLElement | null = null;
  let view = 'prompt-embedding';
  let datasetName = 'diffusiondb';
  let mapTitle = 'Knowledge Map';
  let mapDataURLs: DataURLs | null = null;

  if (window.location.search !== '') {
    const searchParams = new URLSearchParams(window.location.search);
    const title = searchParams.get('title')?.trim();
    const point = searchParams.get('dataURL')?.trim();
    const grid = searchParams.get('gridURL')?.trim();
    const topic = searchParams.get('topicURL')?.trim();
    if (title) {
      mapTitle = title;
    }
    if (point && grid && topic) {
      mapDataURLs = {
        point,
        grid,
        topic
      };
    }
    if (searchParams.has('dataset')) {
      datasetName = searchParams.get('dataset')!;
    }
  }

  // Create stores for child components to consume
  const footerStore = getFooterStore();
  const searchBarStore = getSearchBarStore();
</script>

<style lang="scss">
  @import './MapView.scss';
</style>

<div class="mapview-page">
  <div id="popper-tooltip-top" class="popper-tooltip hidden" role="tooltip">
    <span class="popper-content"></span>
    <div class="popper-arrow"></div>
  </div>

  <div id="popper-tooltip-bottom" class="popper-tooltip hidden" role="tooltip">
    <span class="popper-content"></span>
    <div class="popper-arrow"></div>
  </div>

  <div class="app-wrapper">
    <div class="main-app" bind:this="{component}">
      <div
        class="main-app-container"
        class:hidden="{view !== 'prompt-embedding'}"
      >
        <Embedding
          datasetName="{datasetName}"
          dataURLs="{mapDataURLs}"
          footerStore="{footerStore}"
          searchBarStore="{searchBarStore}"
        />
      </div>
    </div>
  </div>

  <div class="footer-container">
    <Footer footerStore="{footerStore}" title="{mapTitle}" />
  </div>

  <div class="search-panel-container">
    <SearchPanel searchPanelStore="{searchBarStore}" />
  </div>
</div>
