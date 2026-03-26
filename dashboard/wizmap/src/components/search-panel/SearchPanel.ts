import type { Writable } from 'svelte/store';
import d3 from '../../utils/d3-import';
import type { SearchResult, PromptPoint } from '../../types/embedding-types';
import type { SearchBarStoreValue } from '../../stores';
import { getSearchBarStoreDefaultValue } from '../../stores';

export class SearchPanel {
  component: HTMLElement;
  SearchPanelUpdated: () => void;
  inputElement: HTMLInputElement;

  searchBarStore: Writable<SearchBarStoreValue>;
  searchBarStoreValue: SearchBarStoreValue;

  formattedResults: SearchResult[] = [];
  handledQueryID = 0;

  constructor(
    component: HTMLElement,
    SearchPanelUpdated: () => void,
    searchBarStore: Writable<SearchBarStoreValue>
  ) {
    this.component = component;
    this.SearchPanelUpdated = SearchPanelUpdated;

    // Set up the store
    this.searchBarStore = searchBarStore;
    this.searchBarStoreValue = getSearchBarStoreDefaultValue();

    this.inputElement = component.querySelector(
      '#search-bar-input'
    ) as HTMLInputElement;

    this.initStore();
  }

  initStore = () => {
    this.searchBarStore.subscribe(value => {
      this.searchBarStoreValue = value;

      this.handledQueryID = this.searchBarStoreValue.queryID;
      this.formattedResults = this.formatResults(
        this.searchBarStoreValue.results
      );
      this.SearchPanelUpdated();
    });
  };

  /**
   * Format the search results to highlight matches
   * @param results Current search results
   */
  formatResults = (results: PromptPoint[]) => {
    const formattedResults: SearchResult[] = [];
    const query = this.searchBarStoreValue.query;

    for (const resultPoint of results) {
      // Try to avoid XSS attack
      const result = resultPoint.prompt;
      if (result.includes('iframe')) continue;
      if (result.includes('<script')) continue;

      const searchWords = new Set(query.split(/\s+/));
      const newResult: SearchResult = {
        fullText: result,
        shortText: result,
        isSummary: true,
        point: resultPoint
      };

      newResult.fullText = result;

      for (const word of searchWords) {
        if (word === '') continue;
        const re = new RegExp('\\b' + word + '\\b', 'ig');
        newResult.fullText = newResult.fullText.replaceAll(
          re,
          `<em>${word}</em>`
        );
      }

      // Truncate the text if it is too long
      if (newResult.fullText.length > 300) {
        newResult.shortText = newResult.fullText.slice(0, 300);
        newResult.shortText = newResult.shortText.slice(
          0,
          newResult.shortText.lastIndexOf(' ')
        );
        newResult.shortText = newResult.shortText.concat('...');
        newResult.isSummary = true;
      } else {
        newResult.shortText = newResult.fullText;
      }

      formattedResults.push(newResult);
    }

    return formattedResults;
  };

  /**
   * Event handler for event change
   * @param e Event
   */
  inputChanged = (e: InputEvent) => {
    e.preventDefault();

    const query = this.inputElement.value;

    if (query === '') {
      this.cancelSearch();
    }

    if (query !== this.searchBarStoreValue.query) {
      this.searchBarStoreValue.query = query;
      this.searchBarStoreValue.queryID += 1;
      this.searchBarStore.set(this.searchBarStoreValue);
    }
  };

  /**
   * Collapse the search list
   */
  cancelSearch = () => {
    this.formattedResults = [];
    this.searchBarStoreValue.query = '';
    this.searchBarStoreValue.shown = false;
    this.searchBarStore.set(this.searchBarStoreValue);
  };

  mouseenterHandler = (point: PromptPoint) => {
    this.searchBarStoreValue.highlightSearchPoint(point);
  };
}
