# ShopSmarter System Architecture

```mermaid
flowchart LR
    %% User Interface
    UI[React Frontend] --> |Image + Text| API[Flask API]
    UI --> |Checkout| RAZORPAY[Razorpay API]
    
    %% Backend Services
    API --> SEARCH[Search & Scrape Service]
    API --> EMBED[Embedding Service]
    API --> LLM[LLM Service]
    API --> RAZORPAY

    %% Search & Scrape Pipeline
    subgraph "Data Pipeline"
      SCRAPER[Web Scraper] --> PARSER[Product Parser]
      PARSER --> CACHE[Local Cache]
      CACHE --> EMBED
    end
    SEARCH --> |Query + Tags| GSEARCH[Google Custom Search]
    SEARCH --> |Scrape URLs| SCRAPER
    SCRAPER --> |Static Pages| BS4[BeautifulSoup4]
    SCRAPER --> |Dynamic Pages| SELENIUM[Selenium]

    %% Embedding Pipeline
    EMBED --> CLIP[CLIP Model]
    CLIP --> |512-D Vectors| FAISS[FAISS Index]
    FAISS --> FILTER[Filter Service]

    %% Refinement Pipeline
    FILTER --> |Initial Filtered Results| LLM
    LLM --> |Refined Results| API

    %% Payment Flow
    RAZORPAY --> |Checkout Session| UI
    RAZORPAY --> |Webhook| API

    %% Storage & Retraining
    subgraph "Batch Re-index Job"
      CACHE --> TRAIN[Batch Re-index]
      TRAIN --> FAISS
    end

    %% Retry & Error Handling Annotations
    GSEARCH -. retry on failure .-> GSEARCH
    SCRAPER -. retry stale cache .-> PARSER
    FAISS -. fallback to rebuild .-> FAISS
    RAZORPAY -. retry idempotent .-> API

    classDef primary fill:#93c5fd,stroke:#1d4ed8,stroke-width:2px
    classDef secondary fill:#bfdbfe,stroke:#1d4ed8,stroke-width:1px
    classDef external fill:#ddd6fe,stroke:#5b21b6,stroke-width:1px

    class UI,API primary
    class SEARCH,EMBED,LLM,FILTER,PARSER,CACHE,TRAIN secondary
    class GSEARCH,RAZORPAY external
```

## Component Details

1. **Frontend (React)**
   - Image upload via drag-and-drop
   - Text query input
   - Product carousel display
   - Cart management
   - Razorpay checkout integration

2. **Backend (Flask)**
   - RESTful API endpoints under `/api`
   - Service orchestration and error handling
   - Webhook handling with signature verification and retry logic

3. **Search & Scrape Service**
   - Combines CLIP-predicted tags and user prompt for Google Custom Search
   - Multi-strategy scraping (requests + BS4, Selenium fallback)
   - Product parsing with retries for stale cache

4. **Embedding Service**
   - CLIP model inference on images
   - FAISS index build, search, and fallback re-index

5. **Filter Service**
   - Applies price, style, color, and exclusion filters before LLM refinement

6. **LLM Service**
   - Extracts filter criteria and re-ranks via two `LLMChain` passes
   - Runs on CPU with batch retries for improved accuracy

7. **External Services**
   - Razorpay for payments (checkout sessions & webhooks)
   - Google Custom Search for product discovery
   - E-commerce websites as data sources

This diagram shows data flow, retry/error annotations, and a scheduled batch re-index job to keep the FAISS index up-to-date.
