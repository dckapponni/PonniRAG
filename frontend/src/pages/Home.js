import React, { useState } from 'react';
import { getTranslation } from '../services/translations';
import { askQuestion, askQuestionStream } from '../services/api';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';

const Home = ({ language }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const t = (key) => getTranslation(language, key);

  const handleSuggestionClick = (prompt) => {
    handleSubmit(prompt);
  };

  const handleSubmit = async (question) => {
    const userMessage = { role: 'user', content: question };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    // Add placeholder assistant message for streaming
    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: '', sources: [] },
    ]);

    const streamController = askQuestionStream(question, {
      onToken: (token) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = {
              ...last,
              content: last.content + token,
            };
          }
          return updated;
        });
      },
      onSources: (sources) => {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last && last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, sources };
          }
          return updated;
        });
      },
      onDone: () => {
        setIsLoading(false);
      },
      onError: (error) => {
        console.error('Streaming error, falling back:', error);
        streamController.abort();
        // Fall back to non-streaming
        askQuestion(question)
          .then((result) => {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: 'assistant',
                content: result.answer || '',
                sources: result.sources || [],
              };
              return updated;
            });
          })
          .catch((fallbackError) => {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                role: 'assistant',
                content: `${t('error_message')}: ${fallbackError.message}`,
                sources: [],
              };
              return updated;
            });
          })
          .finally(() => setIsLoading(false));
      },
    });
  };

  const suggestions = [
    {
      key: 'sugg_founder',
      prompt: language === 'ta'
        ? 'பொன்னி இதழில் எழுதிய ஆசிரியர்கள் யார்?'
        : 'Who are the authors in ponni magazine?',
    },
    {
      key: 'sugg_poets',
      prompt: language === 'ta'
        ? 'பாரதிதாசன் எழுதிய கட்டுரைகள் பட்டியலிடுக'
        : 'List the articles written by Bharathidasan',
    },
    {
      key: 'sugg_dravidian',
      prompt: language === 'ta'
        ? 'பொன்னி திராவிட இயக்கத்திற்கு எவ்வாறு பங்களித்தது?'
        : 'How did Ponni contribute to the Dravidian movement?',
    },
    {
      key: 'sugg_archive',
      prompt: language === 'ta'
        ? 'வேண்டாத ஆசை ஆசிரியர் யார்?'
        : 'Who is the founder of Ponni magazine?',
    },
  ];

  return (
    <div className="chat-container">
      {messages.length === 0 ? (
        <>
          <div className="hero-container">
            <h1 className="hero-title">{t('app_title')}</h1>
          </div>
          <div className="suggestions-container">
            {suggestions.map((suggestion) => (
              <button
                key={suggestion.key}
                className="suggestion-btn"
                onClick={() => handleSuggestionClick(suggestion.prompt)}
              >
                {t(suggestion.key)}
              </button>
            ))}
          </div>
        </>
      ) : (
        <div className="chat-messages">
          {messages.map((message, index) => (
            <ChatMessage key={index} message={message} language={language} />
          ))}
          {isLoading && (
            <div className="loading-spinner">
              <div className="spinner"></div>
              <span>{t('searching')}</span>
            </div>
          )}
        </div>
      )}
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} language={language} />
    </div>
  );
};

export default Home;
