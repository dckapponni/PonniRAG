import React, { useState, useRef, useEffect } from 'react';
import { getTranslation } from '../services/translations';
import { askQuestion, askQuestionStream } from '../services/api';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';

const Home = ({ language }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const t = (key) => getTranslation(language, key);

  // Track active stream to abort on new submit and guard stale callbacks
  const activeStreamRef = useRef(null);
  const requestIdRef = useRef(0);
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSuggestionClick = (prompt) => {
    handleSubmit(prompt);
  };

  const handleSubmit = async (question) => {
    // Abort any in-flight stream before starting a new one
    if (activeStreamRef.current) {
      activeStreamRef.current.abort();
      activeStreamRef.current = null;
    }

    // Increment request ID — stale callbacks from old streams will be ignored
    const thisRequestId = ++requestIdRef.current;

    const userMessage = { role: 'user', content: question };

    // Build history from last 3 Q&A turns (before adding the new messages)
    const historyMessages = messages.slice(-6); // last 3 turns = 6 messages max
    const history = historyMessages
      .filter((m) => m.content)
      .map((m) => ({ role: m.role, content: m.content }));

    setMessages((prev) => [
      ...prev,
      userMessage,
      { role: 'assistant', content: '', sources: [] },
    ]);
    setIsLoading(true);

    const streamController = askQuestionStream(question, {
      onToken: (token) => {
        if (requestIdRef.current !== thisRequestId) return;
        if (!token || typeof token !== 'string') return;
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
        if (requestIdRef.current !== thisRequestId) return;
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
        if (requestIdRef.current !== thisRequestId) return;
        activeStreamRef.current = null;
        setIsLoading(false);
      },
      onError: (error) => {
        if (requestIdRef.current !== thisRequestId) return;
        console.error('Streaming error, falling back:', error);
        activeStreamRef.current = null;
        // Fall back to non-streaming
        askQuestion(question, true, history)
          .then((result) => {
            if (requestIdRef.current !== thisRequestId) return;
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
            if (requestIdRef.current !== thisRequestId) return;
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
          .finally(() => {
            if (requestIdRef.current === thisRequestId) {
              setIsLoading(false);
            }
          });
      },
    }, history);

    activeStreamRef.current = streamController;
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
            <p className="hero-subtitle">
              {language === 'ta'
                ? '1947\u20131955 வரையிலான பொன்னி இதழின் அறிவுக் களஞ்சியத்தை ஆராயுங்கள்'
                : 'Explore the knowledge archive of Ponni magazine, 1947\u20131955'}
            </p>
            <div className="hero-rule" />
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
          {isLoading && messages.length > 0 && messages[messages.length - 1]?.content === '' && (
            <div className="loading-spinner">
              <div className="spinner"></div>
              <span>{t('searching')}</span>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}
      <ChatInput onSubmit={handleSubmit} isLoading={isLoading} language={language} />
    </div>
  );
};

export default Home;
