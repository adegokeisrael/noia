import React, { useState } from 'react';
import LoginPage    from './components/LoginPage';
import ChatInterface from './components/ChatInterface';

export default function App() {
  const [token, setToken] = useState(localStorage.getItem('noia_token') || '');

  const handleLogin = (t) => {
    localStorage.setItem('noia_token', t);
    setToken(t);
  };

  const handleLogout = () => {
    localStorage.removeItem('noia_token');
    setToken('');
  };

  return token
    ? <ChatInterface token={token} onLogout={handleLogout} />
    : <LoginPage onLogin={handleLogin} />;
}
