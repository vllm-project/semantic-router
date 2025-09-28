# Creating Demo GIF

## Method 1: Using Browser + Screen Recorder

1. Open `terminal-demo.html` in browser
2. Use tool like LICEcap, GIMP, or ffmpeg to record:

```bash
# Using ffmpeg (if installed)
ffmpeg -f avfoundation -i "1" -t 30 -r 10 demo.gif

# Using LICEcap (GUI tool)
# Download from: https://www.cockos.com/licecap/
```

## Method 2: Using Puppeteer (Automated)

```javascript
const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch();
  const page = await page.newPage();
  await page.goto('file://' + __dirname + '/terminal-demo.html');

  // Wait for animation to complete
  await page.waitForTimeout(20000);

  // Take screenshot or record
  await page.screenshot({path: 'demo.png'});
  await browser.close();
})();
```

## Method 3: Embed as Raw HTML (Limited)

GitHub README supports some HTML, but JavaScript is stripped.
The TypeIt.js animation won't work, but we can show a static version.
