/**
 * demo.js — Puppeteer end-to-end test for ShopSmarter MVP
 */

require('dotenv').config();
const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

// Config via env or default
const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
const HEADLESS = process.env.HEADLESS === 'true';
const TEST_IMAGE_PATH = path.join(__dirname, 'test-image.jpg');
if (!fs.existsSync(TEST_IMAGE_PATH)) {
  console.error(`Test image not found at ${TEST_IMAGE_PATH}`);
  process.exit(1);
}

// Razorpay test card
const TEST_CARD = {
  number: '4111111111111111',
  expiry: '12/25',
  cvv: '123',
  name: 'Test User',
  email: 'test@example.com',
  phone: '9999999999'
};

(async () => {
  console.log('Starting ShopSmarter demo...');

  const browser = await puppeteer.launch({
    headless: HEADLESS,
    args: ['--no-sandbox', '--disable-setuid-sandbox'],
    defaultViewport: { width: 1280, height: 800 }
  });

  const page = await browser.newPage();
  try {
    // 1️⃣ Load home
    console.log('Loading home page...');
    await page.goto(BASE_URL, { waitUntil: 'networkidle0' });

    // 2️⃣ Upload image
    console.log('Waiting for file input...');
    await page.waitForSelector('input[type="file"]', { timeout: 5000 });
    const fileInput = await page.$('input[type="file"]');
    console.log('Uploading test image...');
    await fileInput.uploadFile(TEST_IMAGE_PATH);

    // 3️⃣ Wait for results
    console.log('Waiting for product cards...');
    await page.waitForSelector('[data-testid="product-card"]', { timeout: 10000 });

    // 4️⃣ Add first to cart
    console.log('Adding first item to cart...');
    await page.click('[data-testid="product-card"] button'); // adjust if needed

    // 5️⃣ Navigate to cart
    console.log('Opening cart...');
    await page.click('[data-testid="cart-link"]');
    await page.waitForSelector('[data-testid="checkout-button"]', { timeout: 5000 });

    // 6️⃣ Checkout
    console.log('Initiating checkout...');
    await page.click('[data-testid="checkout-button"]');

    // 7️⃣ Razorpay modal
    console.log('Waiting for Razorpay checkout frame...');
    const frameHandle = await page.waitForSelector('iframe[name^="razorpay-checkout-frame"]', { timeout: 10000 });
    const frame = await frameHandle.contentFrame();

    console.log('Filling payment details...');
    await frame.waitForSelector('input[name="contact"]', { timeout: 5000 });
    await frame.type('input[name="contact"]', TEST_CARD.phone);
    await frame.type('input[name="email"]', TEST_CARD.email);

    console.log('Switching to card payment...');
    await frame.click('div[data-method="card"]');

    console.log('Entering card details...');
    await frame.waitForSelector('input[name="card[number]"]');
    await frame.type('input[name="card[number]"]', TEST_CARD.number);
    await frame.type('input[name="card[expiry]"]', TEST_CARD.expiry);
    await frame.type('input[name="card[cvv]"]', TEST_CARD.cvv);
    await frame.type('input[name="card[name]"]', TEST_CARD.name);

    console.log('Submitting payment...');
    await frame.click('button[data-submit="true"]');

    // 8️⃣ Success
    console.log('Waiting for success page...');
    await page.waitForSelector('[data-testid="success-message"]', { timeout: 10000 });
    console.log('Demo completed successfully! 🎉');

  } catch (err) {
    console.error('Demo failed:', err);
    // Capture a screenshot for debugging
    const screenshotPath = path.join(__dirname, 'demo-failure.png');
    await page.screenshot({ path: screenshotPath });
    console.error(`Screenshot saved to ${screenshotPath}`);
  } finally {
    await browser.close();
  }
})();
