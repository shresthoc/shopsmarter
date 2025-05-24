# ShopSmarter 🛍️

An AI-powered e-commerce assistant that helps users find visually similar or complementary products using image and text queries.

## Features

- 📸 Image + text-based product search
- 🔍 AI-powered visual similarity matching using CLIP
- 🤖 LLM-driven query refinement
- 🛒 Test-mode checkout with Stripe
- 🎭 Automated demo with Puppeteer

## Prerequisites

- Python 3.8+
- Node.js 16+
- Google Custom Search API key
- Stripe test mode API keys

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ShopSmarter.git
   cd ShopSmarter
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Node.js dependencies**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

4. **Configure environment variables**
   Create a `.env` file in the root directory:
   ```
   # API Keys
   GOOGLE_CUSTOM_SEARCH_API_KEY=your_key_here
   GOOGLE_CUSTOM_SEARCH_ENGINE_ID=your_engine_id_here
   STRIPE_SECRET_KEY=your_stripe_test_key
   STRIPE_PUBLISHABLE_KEY=your_stripe_test_publishable_key
   
   # Configuration
   FLASK_ENV=development
   FLASK_APP=backend/app.py
   ```

5. **Set up Google Custom Search API**
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project
   - Enable Custom Search API
   - Create credentials and copy your API key
   - Go to [Google Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Create a new search engine
   - Add the sites you want to search (e.g., amazon.com, ebay.com)
   - Copy your Search Engine ID

6. **Run the application**
   
   Terminal 1 (Backend):
   ```bash
   source venv/bin/activate
   flask run
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd frontend
   npm start
   ```

7. **Run the Puppeteer demo**
   ```bash
   cd scripts
   node demo.js
   ```

## Project Structure

```
ShopSmarter/
├── backend/
│   ├── api/
│   ├── services/
│   └── utils/
├── frontend/
│   └── src/
│       ├── components/
│       ├── hooks/
│       └── utils/
├── scripts/
├── requirements.txt
└── package.json
```

## Core Components

1. **Search & Scrape Service**
   - Uses Google Custom Search API
   - Scrapes product details using BeautifulSoup/Selenium
   - Caches thumbnails locally

2. **Embedding & Similarity**
   - CLIP for image embeddings
   - FAISS for fast similarity search
   - Filters and ranks results

3. **LLM Refinement**
   - LangChain for query understanding
   - Re-ranks and filters results
   - Handles follow-up queries

4. **Frontend**
   - React components for image upload and results display
   - Stripe integration for test-mode checkout
   - Responsive product carousel

## Extending the Project

1. **Adding More E-commerce Sites**
   - Add site-specific scraping rules in `backend/services/scrapers/`
   - Update the search engine configuration

2. **Caching & Performance**
   - Implement Redis for embedding cache
   - Add rate limiting for API calls
   - Enable browser caching for thumbnails

3. **Styling & UI**
   - Customize the theme in `frontend/src/styles/`
   - Add more interactive features
   - Implement responsive design

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 