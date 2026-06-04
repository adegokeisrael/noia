import React, { useState } from 'react';
import axios from 'axios';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export default function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('noc_engineer_demo');
  const [password, setPassword] = useState('noia_demo_2025');
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true); setError('');
    try {
      const form = new URLSearchParams();
      form.append('username', username);
      form.append('password', password);
      const { data } = await axios.post(`${API}/api/v1/auth/token`, form);
      onLogin(data.access_token);
    } catch {
      setError('Invalid credentials. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-slate-950">
      <div className="w-full max-w-md bg-slate-900 rounded-2xl shadow-2xl border border-slate-700 p-8">

        {/* Logo / Title */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full
                          bg-blue-900 border-2 border-blue-500 mb-4">
            <span className="text-2xl font-black text-blue-300">N</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">NOIA</h1>
          <p className="text-slate-400 text-sm mt-1">Network Operations Intelligence Assistant</p>
          <p className="text-slate-500 text-xs mt-1">ITU-T FG-AINN Build-a-thon 2025</p>
        </div>

        {/* Form */}
        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2.5
                         text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-400 mb-1 uppercase tracking-wider">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              className="w-full bg-slate-800 border border-slate-600 rounded-lg px-4 py-2.5
                         text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {error && (
            <div className="bg-red-900/40 border border-red-700 rounded-lg px-4 py-2 text-red-300 text-sm">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700
                       text-white font-semibold rounded-lg py-2.5 text-sm
                       transition-colors duration-150"
          >
            {loading ? 'Signing in…' : 'Sign In'}
          </button>
        </form>

        <p className="text-slate-600 text-xs text-center mt-6">
          Demo credentials pre-filled · Apache 2.0 Licensed
        </p>
      </div>
    </div>
  );
}
