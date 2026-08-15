import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './pages/Home';
import Library from './pages/Library';
import Issues from './pages/Issues';
import PDFViewer from './pages/PDFViewer';
import About from './pages/About';
import Dataset from './pages/Dataset';
import History from './pages/History';
import TagBrowse from './pages/TagBrowse';
import './styles/App.css';

function App() {
  const [language, setLanguage] = useState('ta');

  const toggleLanguage = () => {
    setLanguage((prev) => (prev === 'ta' ? 'en' : 'ta'));
  };

  return (
    <Router>
      <div className="App">
        <Navigation language={language} onToggleLanguage={toggleLanguage} />
        <Routes>
          <Route path="/" element={<Home language={language} />} />
          <Route path="/library" element={<Library language={language} />} />
          <Route path="/library/volume/:volumeId" element={<Issues language={language} />} />
          <Route path="/library/volume/:volumeId/issue/:issueId" element={<PDFViewer language={language} />} />
          <Route path="/tags" element={<TagBrowse language={language} />} />
          <Route path="/history" element={<History language={language} />} />
          <Route path="/dataset" element={<Dataset language={language} />} />
          <Route path="/about" element={<About language={language} />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
