import puppeteer from 'puppeteer';

const browser = await puppeteer.launch({ headless: true });
const page = await browser.newPage();
await page.setViewport({ width: 400, height: 800, deviceScaleFactor: 2 });
await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
await page.screenshot({ path: 'screenshot.png' });
await browser.close();
console.log('Screenshot saved to screenshot.png');
