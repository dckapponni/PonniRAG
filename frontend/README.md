# Ponni RAG React Frontend

Modern React.js frontend for the Ponni Tamil Literary Archive RAG System.

## Features

- AI-powered search interface with Tamil/English support
- Digital library browser with volume and issue navigation
- PDF viewer for historical magazine issues
- Responsive design matching the Streamlit UI
- Language toggle (Tamil/English)

## Prerequisites

- Node.js 18+
- npm or yarn
- FastAPI backend running on port 8000

## Quick Start

### Local Development

1. Install dependencies:
```bash
npm install
```

2. Create environment file:
```bash
cp .env.example .env
```

3. Start development server:
```bash
npm start
```

The app will be available at http://localhost:3000

### Docker

Build and run with Docker:

```bash
# Build image
docker build -t ponni-frontend .

# Run container
docker run -p 3000:3000 ponni-frontend
```

Or use Docker Compose from the project root:

```bash
docker-compose up frontend
```

## Project Structure

```
frontend/
├── public/
│   ├── index.html
│   └── images/          # Volume and about images
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── Navigation.js
│   │   ├── ChatMessage.js
│   │   └── ChatInput.js
│   ├── pages/           # Page components
│   │   ├── Home.js      # AI Search page
│   │   ├── Library.js   # Volume browser
│   │   ├── Issues.js    # Issue browser
│   │   ├── PDFViewer.js # PDF viewer
│   │   └── About.js     # About page
│   ├── services/        # API and utilities
│   │   ├── api.js       # FastAPI client
│   │   └── translations.js
│   ├── styles/
│   │   └── App.css      # Global styles
│   ├── App.js
│   └── index.js
├── Dockerfile
├── nginx.conf
└── package.json
```

## API Endpoints Used

| Endpoint | Description |
|----------|-------------|
| `POST /api/ask` | AI question answering |
| `GET /api/library/volumes` | List all volumes |
| `GET /api/library/volumes/:id/issues` | Get volume issues |
| `GET /api/library/volumes/:id/issues/:id/pdf` | Get PDF link |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REACT_APP_API_URL` | FastAPI backend URL | `http://localhost:8000` |

## Adding Images

Place volume cover images in `public/images/`:
- `Volume1.jpg` through `Volume8.jpg` - Volume covers
- `about1.png` through `about5.png` - About page images
- `volume X cover images/issueN.jpg` - Issue covers

## Building for Production

```bash
npm run build
```

The built files will be in the `build/` directory.
