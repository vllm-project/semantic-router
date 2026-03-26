<script lang="ts">
  import { onMount } from 'svelte';
  import { SearchPanel } from './SearchPanel';
  import type { Writable } from 'svelte/store';
  import type { SearchBarStoreValue } from '../../stores';
  import { config } from '../../config/config';
  import d3 from '../../utils/d3-import';
  import iconTop from '../../imgs/icon-top.svg?raw';
  import iconCancel from '../../imgs/icon-cancel.svg?raw';
  import iconSearch from '../../imgs/icon-search.svg?raw';
  import iconWizmap from '../../imgs/icon-wizmap.svg?raw';

  export let searchPanelStore: Writable<SearchBarStoreValue>;

  // Components
  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let mySearchPanel: SearchPanel | null = null;
  let resultListElement: HTMLElement | null = null;
  let searchInputValue = '';

  // Component states
  let inputFocused = false;
  let searchScrolled = false;
  let showScrollTopButton = false;

  let maxListLength = 100;
  const numberFormatter = d3.format(',');

  const searchPanelUpdated = () => {
    mySearchPanel = mySearchPanel;
  };

  onMount(() => {
    mounted = true;
  });

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component && searchPanelStore) {
      mySearchPanel = new SearchPanel(
        component,
        searchPanelUpdated,
        searchPanelStore
      );
    }
  };

  $: mounted && !initialized && component && searchPanelStore && initView();
</script>

<style lang="scss">
  @import './SearchPanel.scss';
</style>

<div class="search-panel-wrapper" bind:this="{component}">
  <div
    class="search-list-container"
    class:shown="{mySearchPanel?.searchBarStoreValue.shown}"
  >
    <div class="search-list">
      <div class="header-gap" class:hidden="{!searchScrolled}"></div>

      {#if mySearchPanel !== null}
        <div
          class="result-list"
          bind:this="{resultListElement}"
          on:scroll="{e => {
            searchScrolled = e.target.scrollTop > 0;
            showScrollTopButton = e.target.scrollTop > 3000;
          }}"
        >
          <div class="count-label">
            {mySearchPanel.searchBarStoreValue.results.length ===
            config.layout.searchLimit
              ? `${config.layout.searchLimit}+`
              : numberFormatter(
                  mySearchPanel.searchBarStoreValue.results.length
                )}
            Search Results
          </div>
          {#each mySearchPanel.formattedResults.slice(0, maxListLength) as result, i}
            <div
              class="item"
              on:keypress="{() => {
                result.isSummary = !result.isSummary;
              }}"
              on:click="{() => {
                result.isSummary = !result.isSummary;
              }}"
              on:mouseenter="{() => {
                mySearchPanel?.mouseenterHandler(result.point);
              }}"
              class:clamp-line="{result.isSummary}"
            >
              {@html result.fullText}
            </div>
          {/each}

          <button
            class="add-more-button"
            class:hidden="{mySearchPanel.searchBarStoreValue.results.length <=
              maxListLength}"
            on:click="{() => {
              maxListLength += 100;
            }}"
          >
            <span>Show More</span>
          </button>
        </div>
      {/if}

      <button
        class="scroll-up-button"
        class:hidden="{!showScrollTopButton}"
        on:click="{() => {
          if (resultListElement !== null) {
            resultListElement.scrollTop = 0;
          }
        }}"
      >
        <div class="svg-icon">
          {@html iconTop}
        </div>
        Back to top
      </button>
    </div>
  </div>

  <div class="search-bar" class:focused="{inputFocused}">
    <div class="svg-icon logo-icon">{@html iconWizmap}</div>
    <input
      type="text"
      id="search-bar-input"
      name="search-query"
      bind:value="{searchInputValue}"
      placeholder="Search knowledge base exemplars"
      spellcheck="false"
      on:focus="{() => {
        inputFocused = true;
      }}"
      on:blur="{() => {
        inputFocused = false;
      }}"
      on:input="{e => mySearchPanel?.inputChanged(e)}"
    />

    <div class="end-button">
      <div
        class="svg-icon search-icon"
        class:hidden="{searchInputValue.length !== 0}"
      >
        {@html iconSearch}
      </div>

      <button
        class="svg-icon cancel-icon"
        class:hidden="{searchInputValue.length === 0}"
        on:click="{() => {
          searchInputValue = '';
          mySearchPanel?.cancelSearch();
        }}"
      >
        {@html iconCancel}
      </button>
    </div>
  </div>
</div>
