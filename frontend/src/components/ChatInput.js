import React, { useState } from 'react';
import { getTranslation } from '../services/translations';

const ChatInput = ({ onSubmit, isLoading, language }) => {
  const [input, setInput] = useState('');
  const t = (key) => getTranslation(language, key);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSubmit(input.trim());
      setInput('');
    }
  };

  return (
    <form className="chat-input-container" onSubmit={handleSubmit}>
      <input
        type="text"
        className="chat-input"
        placeholder={t('hero_input_placeholder')}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        disabled={isLoading}
      />
      <button
        type="submit"
        className="chat-submit-btn"
        disabled={isLoading || !input.trim()}
      >
        {isLoading ? '...' : '→'}
      </button>
    </form>
  );
};

export default ChatInput;
