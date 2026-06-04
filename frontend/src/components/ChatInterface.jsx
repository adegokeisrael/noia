import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import CitationPanel    from './CitationPanel';
import ConfidenceGauge  from './ConfidenceGauge';
import KBArticleExport  from './KBArticleExport';
import { Send, LogOut, Zap, Search, FileText, Wrench, BookOpen } from 'lucide-react';

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const MODES = [
  { id: 'query',            label: 'Query',      icon: Search,   endpoint: '/api/v1/query',            field: 'query',               color: 'blue'   },
  { id: 'rca',              label: 'RCA',        icon: Zap,      endpoint: '/api/v1/rca',              field: 'incident_description', color: 'red'    },
  { id: 'summarise',        label: 'Summarise',  icon: FileText, endpoint: '/api/v1/summarise',        field: 'incident_thread',      color: 'purple' },
  { id: 'recommend',        label: 'Recommend',  icon: Wrench,   endpoint: '/api/v1/recommend',        field: 'fault_description',    color: 'amber'  },
  { id: 'generate_article', label: 'KB Article', icon: BookOpen, endpoint: '/api/v1/generate-article', field: 'resolved_incident',    color: 'green'  },
];

const COLOR_MAP = {
  blue:   'bg-blue-600 hover:bg-blue-500 ring-blue-500',
  red:    'bg-red-700 hover:bg-red-600 ring-red-500',
  purple: 'bg-purple-700 hover:bg-purple-600 ring-purple-500',
  amber:  'bg-amber-600 hover:bg-amber-500 ring-amber-500',
  green:  'bg-emerald-700 hover:bg-emerald-600 ring-emerald-500',
};

const PLACEHOLDERS = {
  query:            'e.g. What are the steps to restore a BGP session after a route flap on Nokia 7750?',
  rca:              'Describe the outage: affected nodes, alarms, timeline, recent maintenance…',
  summarise:        'Paste the full incident ticket thread here…',
  recommend:        'Describe the current fault or anomaly…',
  generate_article: 'Paste the resolved incident thread with resolution notes…',
};

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser
        ? 'bg-blue-700 text-white rounded-2xl rounded-tr-sm px-4 py-3'
        : 'bg-slate-800 text-slate-100 rounded-2xl rounded-tl-sm px-4 py-3 border border-slate-700'}`}>

        {/* Mode badge */}
        {msg.mode && !isUser && (
          <span className="inline-block text-xs font-semibold bg-slate-700 text-slate-300
                           rounded-full px-2 py-0.5 mb-2 uppercase tracking-wider">
            {msg.mode}
          </span>
        )}

        {/* Main content */}
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{msg.text}</p>
        ) : (
          <div className="text-sm prose prose-invert prose-sm max-w-none">
            {msg.loading
              ? <span className="animate-pulse text-slate-400">NOIA is thinking…</span>
              : <ReactMarkdown>{typeof msg.text === 'string'
                  ? msg.text
                  : JSON.stringify(msg.text, null, 2)}</ReactMarkdown>}
          </div>
        )}

        {/* Groundedness + citations */}
        {msg.groundedness && !msg.loading && (
          <div className="mt-3 pt-3 border-t border-slate-700 space-y-2">
            <ConfidenceGauge
              score={msg.groundedness.groundedness_score}
              reviewRequired={msg.groundedness.review_required}
            />
            {msg.sources?.length > 0 && <CitationPanel sources={msg.sources} />}
            {msg.mode === 'generate_article' && msg.rawResponse && (
              <KBArticleExport article={msg.rawResponse} />
            )}
          </div>
        )}

        {/* Latency */}
        {msg.latency && !msg.loading && (
          <p className="text-xs text-slate-500 mt-2">{msg.latency.toFixed(0)} ms</p>
        )}
      </div>
    </div>
  );
}

export default function ChatInterface({ token, onLogout }) {
  const [messages,  setMessages]  = useState([
    { role: 'assistant', text: '👋 Welcome to **NOIA**. Select a task mode, then type your query.', mode: null }
  ]);
  const [mode,      setMode]      = useState(MODES[0]);
  const [input,     setInput]     = useState('');
  const [loading,   setLoading]   = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || loading) return;

    setInput('');
    const userMsg = { role: 'user', text, mode: mode.label };
    const pendingId = Date.now();
    setMessages(prev => [
      ...prev,
      userMsg,
      { role: 'assistant', text: '', mode: mode.label, loading: true, id: pendingId }
    ]);
    setLoading(true);

    try {
      const { data } = await axios.post(
        `${API}${mode.endpoint}`,
        { [mode.field]: text },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Format response text from the structured JSON
      const resp     = data.response || {};
      const mainText = resp.answer
        || resp.narrative_summary
        || resp.root_cause_explanation
        || (resp.probable_causes ? `**Primary cause:** ${resp.probable_causes[0]?.cause || ''}` : null)
        || (resp.recommendations  ? `**Top action:** ${resp.recommendations[0]?.action || ''}` : null)
        || JSON.stringify(resp, null, 2);

      setMessages(prev => prev.map(m => m.id === pendingId ? {
        ...m,
        loading:     false,
        text:        mainText,
        groundedness:data.groundedness,
        sources:     data.context_sources,
        latency:     data.response_time_ms,
        rawResponse: resp,
      } : m));
    } catch (err) {
      const errMsg = err.response?.status === 403
        ? 'Your role does not have permission for this operation.'
        : err.response?.data?.detail || 'Request failed. Please check the API server.';
      setMessages(prev => prev.map(m => m.id === pendingId ? {
        ...m, loading: false,
        text: `⚠️ Error: ${errMsg}`,
      } : m));
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const activeColor = COLOR_MAP[mode.color];

  return (
    <div className="flex flex-col h-screen bg-slate-950">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="flex items-center justify-between px-5 py-3
                         bg-slate-900 border-b border-slate-800 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-blue-900 border border-blue-500
                          flex items-center justify-center">
            <span className="text-blue-300 font-black text-sm">N</span>
          </div>
          <div>
            <h1 className="text-white font-bold text-sm leading-none">NOIA</h1>
            <p className="text-slate-500 text-xs">NOC Intelligence Assistant</p>
          </div>
        </div>

        {/* Mode selector */}
        <div className="flex gap-1.5">
          {MODES.map(m => {
            const Icon = m.icon;
            const isActive = m.id === mode.id;
            return (
              <button
                key={m.id}
                onClick={() => setMode(m)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold
                            transition-all duration-150
                            ${isActive
                              ? `${COLOR_MAP[m.color].split(' ')[0]} text-white ring-2 ${COLOR_MAP[m.color].split(' ')[2]}`
                              : 'bg-slate-800 text-slate-400 hover:bg-slate-700'}`}
              >
                <Icon size={12} />
                {m.label}
              </button>
            );
          })}
        </div>

        <button
          onClick={onLogout}
          className="flex items-center gap-1.5 text-slate-400 hover:text-white text-xs
                     transition-colors duration-150"
        >
          <LogOut size={14} /> Logout
        </button>
      </header>

      {/* ── Message list ────────────────────────────────────────────────────── */}
      <main className="flex-1 overflow-y-auto px-4 py-4 scrollbar-thin">
        {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
        <div ref={bottomRef} />
      </main>

      {/* ── Input bar ───────────────────────────────────────────────────────── */}
      <footer className="px-4 pb-4 pt-2 bg-slate-950 border-t border-slate-800 shrink-0">
        <div className={`flex gap-2 items-end bg-slate-900 border border-slate-700
                         rounded-2xl px-4 py-3 focus-within:ring-2 ${activeColor.split(' ')[2]}`}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKey}
            placeholder={PLACEHOLDERS[mode.id]}
            rows={2}
            className="flex-1 bg-transparent text-slate-100 text-sm resize-none
                       focus:outline-none placeholder:text-slate-500"
            disabled={loading}
          />
          <button
            onClick={sendMessage}
            disabled={loading || !input.trim()}
            className={`shrink-0 p-2 rounded-xl text-white transition-all duration-150
                        disabled:opacity-40 disabled:cursor-not-allowed
                        ${activeColor.split(' ').slice(0, 2).join(' ')}`}
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-slate-600 text-xs text-center mt-2">
          Mode: <span className="text-slate-400 font-medium">{mode.label}</span>
          &nbsp;·&nbsp;Press Enter to send · Shift+Enter for new line
        </p>
      </footer>
    </div>
  );
}
