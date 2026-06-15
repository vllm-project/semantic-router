<script lang="ts">
  import { Packer } from './Packing';
  import { onMount } from 'svelte';
  import type { Writable } from 'svelte/store';
  import type { TooltipStoreValue } from '../../stores';
  import { getTooltipStoreDefaultValue } from '../../stores';

  let component: HTMLElement | null = null;
  let mounted = false;
  let initialized = false;
  let packer: Packer | null = null;

  export let tooltipStore: Writable<TooltipStoreValue> | null = null;

  onMount(() => {
    mounted = true;
  });

  const updatePacker = () => {
    packer = packer;
  };

  /**
   * Initialize the embedding view.
   */
  const initView = () => {
    initialized = true;

    if (component && tooltipStore) {
      packer = new Packer({ component, tooltipStore, updatePacker });
    }
  };

  $: mounted && !initialized && component && tooltipStore && initView();
</script>

<style lang="scss">
  @import './Packing.scss';
</style>

<div class="packing-wrapper" bind:this={component}>
  <div class="packing">
    <svg class="packing-svg" />
  </div>
</div>
