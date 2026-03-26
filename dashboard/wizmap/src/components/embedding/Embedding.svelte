<script lang="ts">
  import { Embedding } from './Embedding';
  import { onMount } from 'svelte';
  import type {
    EmbeddingInitSetting,
    DataURLs
  } from '../../types/embedding-types';
  import type { Writable } from 'svelte/store';
  import type { FooterStoreValue, SearchBarStoreValue } from '../../stores';
  import iconContour2 from '../../imgs/icon-contour.svg?raw';
  import iconPoint from '../../imgs/icon-point.svg?raw';
  import iconLabel from '../../imgs/icon-label.svg?raw';
  import iconGrid from '../../imgs/icon-grid.svg?raw';
  import iconTime from '../../imgs/icon-time.svg?raw';
  import iconCaret from '../../imgs/icon-caret-down.svg?raw';
  import iconPlay from '../../imgs/icon-play-solid.svg?raw';
  import iconPause from '../../imgs/icon-pause-solid.svg?raw';
  const iconContour = iconContour2.replaceAll('black', 'currentColor');

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let myEmbedding: Embedding | null = null;
  let controlDisplayItem = '';
  let groupedNames: string[] = [];
  let hasGroupedEmbedding = false;
  $: groupedNames = myEmbedding?.groupNames ?? [];
  $: hasGroupedEmbedding = groupedNames.length > 0;

  const defaultSetting: EmbeddingInitSetting = {
    showContour: true,
    showPoint: true,
    showGrid: false,
    showLabel: true
  };

  export let datasetName = 'diffusiondb';
  export let dataURLs: DataURLs | null = null;
  export let footerStore: Writable<FooterStoreValue>;
  export let searchBarStore: Writable<SearchBarStoreValue>;

  const defaultDataURLs = (name: string): DataURLs => {
    let DATA_BASE = `${import.meta.env.BASE_URL}data`;
    if (import.meta.env.MODE === 'vercel' || import.meta.env.MODE === 'github') {
      DATA_BASE = 'https://pub-596951ee767949aba9096a18685c74bd.r2.dev';
    }

    switch (name) {
      case 'diffusiondb':
        return {
          point: DATA_BASE + '/diffusiondb/umap.ndjson',
          grid: DATA_BASE + '/diffusiondb/grid.json',
          topic: DATA_BASE + '/diffusiondb/umap-1m-topic-data.json'
        };
      case 'acl-abstracts':
        return {
          point: DATA_BASE + '/acl-abstracts/umap.ndjson',
          grid: DATA_BASE + '/acl-abstracts/grid.json',
          topic: DATA_BASE + '/acl-abstracts/topic.json'
        };
      default:
        console.error(`Unknown dataset name: ${name}`);
        return {
          point: '',
          grid: '',
          topic: ''
        };
    }
  };

  const resolveDataURLs = (): DataURLs => {
    if (dataURLs) {
      return dataURLs;
    }
    return defaultDataURLs(datasetName);
  };

  const anyTrue = (items: boolean[]) => items.reduce((a, b) => a || b);
  const allTrue = (items: boolean[]) => items.reduce((a, b) => a && b);

  onMount(() => {
    mounted = true;
  });

  const updateEmbedding = () => {
    myEmbedding = myEmbedding;
  };

  const displayCheckboxChanged = (
    e: InputEvent,
    checkbox: string,
    group: string | undefined = undefined
  ) => {
    const newValue = (e.target as HTMLInputElement).checked;
    myEmbedding?.displayCheckboxChanged(checkbox, newValue, group);
  };

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component) {
      myEmbedding = new Embedding({
        component,
        updateEmbedding,
        defaultSetting,
        dataURLs: resolveDataURLs(),
        footerStore,
        searchBarStore
      });
    }
  };

  $: mounted && !initialized && component && initView();
</script>

<style lang="scss">
  @import './Embedding.scss';
</style>

<div class="embedding-wrapper" bind:this="{component}">
  <div class="grab-blocker"></div>
  <div class="embedding">
    <svg class="top-svg"></svg>
    <canvas class="search-point-canvas hidden"></canvas>
    <canvas class="embedding-canvas"></canvas>
    <canvas class="embedding-canvas-back"></canvas>
    <canvas class="topic-grid-canvas top"></canvas>
    <canvas class="topic-grid-canvas bottom"></canvas>
    <svg class="embedding-svg"></svg>
  </div>

  <div class="control-bar">
    <button
      class="item-wrapper"
      on:click="{() => {
        if (!hasGroupedEmbedding) {
          const nextValue = myEmbedding ? !myEmbedding.showContours[0] : false;
          myEmbedding?.displayCheckboxChanged('contour', nextValue);
        } else {
          if (controlDisplayItem === 'contour') {
            controlDisplayItem = '';
          } else {
            if (controlDisplayItem === 'time') {
              myEmbedding?.displayCheckboxChanged('time', false);
            }
            controlDisplayItem = 'contour';
          }
        }
      }}"
    >
      <div
        class="item"
        class:activated="{myEmbedding
          ? anyTrue(myEmbedding.showContours)
          : false}"
      >
        <div class="svg-icon">{@html iconContour}</div>
        <div class="name">Contour</div>
        <div
          class="caret"
          class:hidden="{!hasGroupedEmbedding}"
          class:activated="{controlDisplayItem === 'contour'}"
        >
          <div class="svg-icon">
            {@html iconCaret}
          </div>
        </div>
      </div>

      {#if hasGroupedEmbedding}
        <button
          class="menu contour-menu"
          class:hidden="{controlDisplayItem !== 'contour'}"
          on:click="{e => {
            e.stopPropagation();
          }}"
        >
          {#each groupedNames as groupName, index (groupName)}
            <div class="control-row">
              <input
                type="checkbox"
                class="checkbox"
                id={"checkbox-contour-" + index}
                name={"checkbox-contour-" + index}
                checked="{index === 0 ? defaultSetting.showContour : false}"
                on:input="{e => displayCheckboxChanged(e, 'contour', groupName)}"
              />
              <label for={"checkbox-contour-" + index}>{groupName}</label>
            </div>
          {/each}
        </button>
      {/if}
    </button>
    <div class="flex-gap"></div>

    <button
      class="item-wrapper"
      on:click="{() => {
        if (!hasGroupedEmbedding) {
          const nextValue = myEmbedding ? !myEmbedding.showPoints[0] : false;
          myEmbedding?.displayCheckboxChanged('point', nextValue);
        } else {
          if (controlDisplayItem === 'point') {
            controlDisplayItem = '';
          } else {
            if (controlDisplayItem === 'time') {
              myEmbedding?.displayCheckboxChanged('time', false);
            }
            controlDisplayItem = 'point';
          }
        }
      }}"
    >
      <div
        class="item"
        class:activated="{myEmbedding
          ? anyTrue(myEmbedding.showPoints)
          : false}"
      >
        <div class="svg-icon">{@html iconPoint}</div>
        <div class="name">Point</div>
        <div
          class="caret"
          class:hidden="{!hasGroupedEmbedding}"
          class:activated="{controlDisplayItem === 'point'}"
        >
          <div class="svg-icon">
            {@html iconCaret}
          </div>
        </div>
      </div>

      {#if hasGroupedEmbedding}
        <button
          class="menu point-menu"
          class:hidden="{controlDisplayItem !== 'point'}"
          on:click="{e => {
            e.stopPropagation();
          }}"
        >
          {#each groupedNames as groupName, index (groupName)}
            <div class="control-row">
              <input
                type="checkbox"
                class="checkbox"
                id={"checkbox-point-" + index}
                name={"checkbox-point-" + index}
                checked="{index === 0 ? defaultSetting.showPoint : false}"
                on:input="{e => displayCheckboxChanged(e, 'point', groupName)}"
              />
              <label for={"checkbox-point-" + index}>{groupName}</label>
            </div>
          {/each}
        </button>
      {/if}
    </button>
    <div class="flex-gap"></div>

    <button
      class="item-wrapper"
      on:click="{() => {
        if (defaultSetting.showGrid) {
          defaultSetting.showGrid = false;
          myEmbedding?.displayCheckboxChanged('grid', false);
        } else {
          defaultSetting.showGrid = true;
          myEmbedding?.displayCheckboxChanged('grid', true);
        }
      }}"
    >
      <div class="item" class:activated="{defaultSetting.showGrid}">
        <div class="svg-icon">{@html iconGrid}</div>
        <div class="name">Grid</div>
      </div>
    </button>
    <div class="flex-gap"></div>

    <button
      class="item-wrapper"
      on:click="{() => {
        if (controlDisplayItem === 'label') {
          controlDisplayItem = '';
        } else {
          if (controlDisplayItem === 'time') {
            myEmbedding?.displayCheckboxChanged('time', false);
          }
          controlDisplayItem = 'label';
        }
      }}"
    >
      <div class="item" class:activated="{defaultSetting.showLabel}">
        <div class="svg-icon">{@html iconLabel}</div>
        <div class="name">Label</div>
        <div class="caret" class:activated="{controlDisplayItem === 'label'}">
          <div class="svg-icon">
            {@html iconCaret}
          </div>
        </div>
      </div>

      <button
        class="menu label-menu"
        class:hidden="{controlDisplayItem !== 'label'}"
        on:click="{e => {
          e.stopPropagation();
        }}"
      >
        <div class="control-item">
          <div class="item-header">Automatic Labeling</div>

          <div class="control-row">
            <input
              type="checkbox"
              class="checkbox"
              id="checkbox-label"
              name="checkbox-label"
              bind:checked="{defaultSetting.showLabel}"
              on:input="{e => displayCheckboxChanged(e, 'label')}"
            />
            <label for="checkbox-label">High Density Region</label>
          </div>
        </div>

        <div class="control-item slider-item">
          <div class="control-row">
            <label class="slider-label" for="slider-label-num"
              >Number of Labels</label
            >
            <span class="slider-count">0</span>
          </div>

          <input
            type="range"
            class="slider"
            id="slider-label-num"
            name="label-num"
            disabled="{!defaultSetting.showLabel}"
            min="0"
            max="0"
            on:input="{e =>
              myEmbedding ? myEmbedding.labelNumSliderChanged(e) : () => {}}"
          />
        </div>
      </button>
    </button>
    <div class="flex-gap"></div>

    <button
      class="item-wrapper"
      disabled="{myEmbedding ? myEmbedding.timeCountMap === null : true}"
      on:click="{() => {
        if (controlDisplayItem === 'time') {
          controlDisplayItem = '';
          myEmbedding?.displayCheckboxChanged('time', false);
        } else {
          controlDisplayItem = 'time';
          myEmbedding?.displayCheckboxChanged('time', true);
        }
      }}"
    >
      <button class="item" class:activated="{controlDisplayItem === 'time'}">
        <div class="svg-icon">{@html iconTime}</div>
        <div class="name">Time</div>
        <div class="caret" class:activated="{controlDisplayItem === 'time'}">
          <div class="svg-icon">
            {@html iconCaret}
          </div>
        </div>
      </button>

      <button
        class="menu time-menu"
        class:hidden="{controlDisplayItem !== 'time'}"
        on:click="{e => {
          e.stopPropagation();
        }}"
      >
        <div class="control-row">
          <div class="play-pause-button">
            <button
              class="svg-icon"
              class:hidden="{myEmbedding
                ? myEmbedding.playingTimeSlider
                : false}"
            >
              {@html iconPlay}
            </button>
            <button
              class="svg-icon"
              class:hidden="{myEmbedding
                ? !myEmbedding.playingTimeSlider
                : true}"
            >
              {@html iconPause}
            </button>
          </div>

          <div class="slider-container">
            <div class="back-slider"></div>

            <div class="slider">
              <div class="range-track"></div>
              <div
                class="middle-thumb"
                id="time-slider-middle-thumb"
                tabindex="-1"
              >
                <div class="thumb-label thumb-label-middle">
                  <span class="thumb-label-span"></span>
                </div>
              </div>
            </div>

            <div class="slider-svg-container">
              <svg class="slider-svg"> </svg>
            </div>
          </div>
        </div>
      </button>
    </button>
  </div>
</div>
