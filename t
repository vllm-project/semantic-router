name: Maintainer Board

on:
  schedule:
    - cron: 0 8 * * *

jobs:
  maintainer-board:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      - name: Set up Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '14'
      - name: Install dependencies
        run: |
          npm install
      - name: Run maintainer board
        run: |
          node -m tenacity maintainer_board.js
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: maintainer-board
          path: maintainer-board.json