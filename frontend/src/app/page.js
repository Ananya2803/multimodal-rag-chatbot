"use client";

import React, { useState, useEffect, useRef } from "react";
import { marked } from "marked";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

// Configure marked to render tables properly
marked.setOptions({
  breaks: true,
  gfm: true
});

export default function Home() {
  // State variables
  const [model, setModel] = useState("Qwen/Qwen2.5-7B-Instruct");
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [thinking, setThinking] = useState(false);
  const [activeCitations, setActiveCitations] = useState([]);
  const [activeMessageIndex, setActiveMessageIndex] = useState(null);
  
  // RAG tuning variables
  const [topK, setTopK] = useState(7);
  const [answerStyle, setAnswerStyle] = useState("detailed");
  
  // File upload state
  const fileInputRef = useRef(null);
  const feedEndRef = useRef(null);

  // Fetch documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Scroll to bottom of chat feed when new messages arrive
  useEffect(() => {
    feedEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, thinking]);

  // Fetch lists of documents from backend
  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data);
      }
    } catch (err) {
      console.error("Error fetching documents:", err);
    }
  };

  // Handle file drop/upload
  const handleUpload = async (e) => {
    const files = e.target.files || e.dataTransfer.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }
    try {
      const res = await fetch(`${BACKEND_URL}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        await fetchDocuments();
        alert("PDFs uploaded and parsed successfully!");
      } else {
        const errData = await res.json();
        alert(`Upload failed: ${errData.detail || "Unknown error"}`);
      }
    } catch (err) {
      console.error("Error uploading:", err);
      alert("Error uploading files to server.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  // Change Document Priority
  const handlePriorityChange = async (docName, newPriority) => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/documents/priority`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          doc_name: docName,
          priority: newPriority,
        }),
      });

      if (res.ok) {
        // Update local state quickly
        setDocuments(prev => prev.map(doc => 
          doc.doc_name === docName ? { ...doc, priority: newPriority } : doc
        ));
      } else {
        alert("Failed to update priority.");
      }
    } catch (err) {
      console.error("Error updating priority:", err);
    }
  };

  // Clear Database and uploaded files
  const handleClearAll = async () => {
    if (!confirm("Are you sure you want to clear all documents and vectors?")) return;

    try {
      const res = await fetch(`${BACKEND_URL}/api/documents/clear`, {
        method: "POST",
      });

      if (res.ok) {
        setDocuments([]);
        setChatHistory([]);
        alert("Cleared everything successfully.");
      }
    } catch (err) {
      console.error("Error clearing documents:", err);
    }
  };

  // Handle Query Submission
  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = {
      sender: "user",
      text: query,
    };

    setChatHistory((prev) => [...prev, userMessage]);
    const currentQuery = query;
    setQuery("");
    setThinking(true);

    try {
      const res = await fetch(`${BACKEND_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: currentQuery,
          model: model,
          top_k: topK,
          retrieval_mode: "tiered",
          answer_style: answerStyle,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const assistantMessage = {
          sender: "assistant",
          text: data.answer,
          citations: data.citations || [],
          error: data.error || null
        };
        setActiveCitations(data.citations || []);
        setChatHistory((prev) => {
          const updated = [...prev, assistantMessage];
          setActiveMessageIndex(updated.length - 1);
          return updated;
        });
      } else {
        const errData = await res.json();
        setChatHistory((prev) => [
          ...prev,
          {
            sender: "assistant",
            text: `Failed to fetch response: ${errData.detail || "Unknown error"}`,
            error: true
          },
        ]);
      }
    } catch (err) {
      console.error("Error querying RAG:", err);
      setChatHistory((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "Connection error: Failed to reach backend server.",
          error: true
        },
      ]);
    } finally {
      setThinking(false);
    }
  };

  // Render markdown with citations mapped to badges
  const renderMarkdown = (text) => {
    if (!text) return { __html: "" };
    // Replace relative static image paths with full backend URLs
    let processedText = text;
    processedText = processedText.replace(/\(\/static\/images\//g, `(${BACKEND_URL}/static/images/`);
    const htmlContent = marked.parse(processedText);
    return { __html: htmlContent };
  };

  return (
    <div className="app-container">
      {/* Background radial glow */}
      <div className="bg-glow-container">
        <div className="bg-glow-orb-1"></div>
        <div className="bg-glow-orb-2"></div>
      </div>

      {/* Sidebar for configs and uploads */}
      <aside className="sidebar">
        <div className="logo-section">
          <div className="logo-icon">Q</div>
          <h1 className="logo-text">DocuQuery RAG</h1>
        </div>

        {/* PDF Ingestion */}
        <section className="sidebar-panel">
          <h2 className="panel-title">1. Ingest Documents</h2>
          
          <div 
            className="dropzone"
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleUpload(e);
            }}
          >
            <span className="dropzone-icon">📥</span>
            <span className="form-label" style={{ fontWeight: 600 }}>Click or drag PDF files</span>
            <span style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Supports multiple PDFs</span>
            <input
              type="file"
              multiple
              accept=".pdf"
              style={{ display: "none" }}
              ref={fileInputRef}
              onChange={handleUpload}
            />
          </div>

          {uploading && (
            <div style={{ fontSize: "0.8rem", color: "var(--accent-secondary)", textAlign: "center" }}>
              ⏳ Parsing and indexing PDF files...
            </div>
          )}
        </section>

        {/* RAG Fine-Tuning */}
        <section className="sidebar-panel">
          <h2 className="panel-title">2. RAG Fine-Tuning</h2>
          
          <div className="form-group">
            <label className="form-label" htmlFor="modelSelect">Language Model</label>
            <select
              id="modelSelect"
              className="select-input"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="meta-llama/Meta-Llama-3-8B-Instruct">Llama 3 8B (Instruct - Gated)</option>
              <option value="Qwen/Qwen2.5-7B-Instruct">Qwen 2.5 7B (Instruct - Open/Recommended)</option>
              <option value="mistralai/Mistral-7B-Instruct-v0.3">Mistral 7B v0.3 (Instruct - Open)</option>
              <option value="google/gemma-2-9b-it">Gemma 2 9B IT (Gated)</option>
              <option value="microsoft/Phi-3-mini-4k-instruct">Phi-3 Mini (Instruct - Open/Fast)</option>
            </select>
          </div>



          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label className="form-label" htmlFor="topKRange">Context Chunk Limit (Top K)</label>
              <span style={{ fontSize: "0.85rem", color: "var(--accent-secondary)", fontWeight: "bold" }}>{topK}</span>
            </div>
            <input
              id="topKRange"
              type="range"
              min="3"
              max="12"
              step="1"
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value))}
              style={{
                width: "100%",
                accentColor: "var(--accent-primary)",
                cursor: "pointer",
                margin: "0.25rem 0"
              }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.65rem", color: "var(--text-muted)" }}>
              <span>3 chunks</span>
              <span>12 chunks</span>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="answerStyleSelect">Answer Style</label>
            <select
              id="answerStyleSelect"
              className="select-input"
              value={answerStyle}
              onChange={(e) => setAnswerStyle(e.target.value)}
            >
              <option value="detailed">🗣️ Detailed & Comprehensive</option>
              <option value="balanced">⚖️ Balanced & Structured</option>
              <option value="concise">⚡ Concise & Direct</option>
            </select>
          </div>
        </section>

        {/* PDF Source Prioritization */}
        <section className="sidebar-panel" style={{ flexGrow: 1, display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h2 className="panel-title">3. Source Priorities</h2>
            {documents.length > 0 && (
              <button 
                onClick={handleClearAll} 
                className="btn btn-danger" 
                style={{ padding: "0.25rem 0.5rem", fontSize: "0.75rem" }}
              >
                Clear All
              </button>
            )}
          </div>

          {documents.length === 0 ? (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flexGrow: 1, padding: "2rem 0", color: "var(--text-muted)", fontSize: "0.85rem", gap: "0.5rem" }}>
              <span>No documents uploaded.</span>
            </div>
          ) : (
            <div className="doc-list">
              {documents.map((doc) => (
                <div key={doc.doc_name} className="doc-item">
                  <div className="doc-header">
                    <span className="doc-title" title={doc.doc_name}>
                      {doc.doc_name}
                    </span>
                    <span className="doc-size">
                      {(doc.size_bytes / 1024 / 1024).toFixed(2)} MB
                    </span>
                  </div>
                  
                  <div className="doc-stats">
                    <span className="doc-stat-tag">📄 {doc.page_count} pages</span>
                    {doc.tables > 0 && <span className="doc-stat-tag">📊 {doc.tables} tables</span>}
                    {doc.images > 0 && <span className="doc-stat-tag">🖼️ {doc.images} images</span>}
                  </div>

                  <div className="doc-priority-selector">
                    <span className="form-label" style={{ fontSize: "0.75rem" }}>Priority:</span>
                    <select
                      className="select-input"
                      style={{ padding: "0.2rem 1.5rem 0.2rem 0.5rem", width: "auto", fontSize: "0.75rem", backgroundPosition: "right 0.4rem center" }}
                      value={doc.priority}
                      onChange={(e) => handlePriorityChange(doc.doc_name, e.target.value)}
                    >
                      <option value="High">🔴 High (1.5x)</option>
                      <option value="Medium">🔵 Medium (1.0x)</option>
                      <option value="Low">🟢 Low (0.5x)</option>
                    </select>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      </aside>

      {/* Main Chat Interface */}
      <main className="main-chat-area">
        <header className="chat-header">
          <div className="chat-header-title">
            <div className="status-dot"></div>
            <span>RAG Chatbot Assistant</span>
            {documents.length > 0 && (
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: "normal" }}>
                ({documents.length} source{documents.length > 1 ? "s" : ""} active)
              </span>
            )}
          </div>
          
          <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
            Powered by Hugging Face Inference
          </div>
        </header>

        {/* Message Feed Area */}
        {chatHistory.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">🤖</div>
            <h3 className="empty-state-title">Multi-Format PDF Chatbot</h3>
            <p className="empty-state-desc">
              Upload multiple PDF files containing text, tables, and images. 
              Configure source document priorities in the sidebar, and start asking questions!
            </p>
          </div>
        ) : (
          <div className="message-feed">
            {chatHistory.map((msg, idx) => (
              <div 
                key={idx} 
                className={`message-wrapper ${msg.sender} ${activeMessageIndex === idx ? "active-message" : ""}`}
                onClick={() => {
                  if (msg.sender === "assistant" && msg.citations) {
                    setActiveCitations(msg.citations);
                    setActiveMessageIndex(idx);
                  }
                }}
                style={{ 
                  cursor: msg.sender === "assistant" ? "pointer" : "default"
                }}
              >
                <div className={`message-avatar ${msg.sender}`}>
                  {msg.sender === "user" ? "U" : "AI"}
                </div>
                <div>
                  <div 
                    className="message-bubble"
                    dangerouslySetInnerHTML={renderMarkdown(msg.text)}
                    style={{
                      border: activeMessageIndex === idx ? "1px solid var(--accent-primary)" : "",
                      boxShadow: activeMessageIndex === idx ? "0 0 12px rgba(99, 102, 241, 0.25)" : ""
                    }}
                  />
                  
                  {/* Render citations badges for assistant responses */}
                  {msg.sender === "assistant" && msg.citations && msg.citations.length > 0 && (
                    <div className="citations-container">
                      <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", alignItems: "center" }}>
                        Sources:
                      </span>
                      {msg.citations.map((c) => (
                        <button
                          key={c.index}
                          className={`citation-badge ${
                            c.priority === "High" ? "priority-tag-high" : 
                            c.priority === "Low" ? "priority-tag-low" : "priority-tag-medium"
                          }`}
                          onClick={(e) => {
                            e.stopPropagation(); // prevent message-wrapper click re-trigger
                            setActiveCitations(msg.citations || []);
                            setActiveMessageIndex(idx);
                          }}
                          title={`Priority: ${c.priority} | Boosted Score: ${c.boosted_score.toFixed(3)}`}
                        >
                          <span className="citation-number">{c.index}</span>
                          <span className="doc-name-text" title={c.doc_name}>{c.doc_name}</span>
                          <span className="doc-page-text">(p.{c.page})</span>
                          <span style={{ fontSize: "0.65rem", opacity: 0.7 }} className="doc-type-text">
                            [{c.type}]
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {thinking && (
              <div className="thinking-container">
                <span className="dot"></span>
                <span className="dot"></span>
                <span className="dot"></span>
                <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: "0.25rem" }}>
                  AI is thinking...
                </span>
              </div>
            )}
            
            <div ref={feedEndRef} />
          </div>
        )}

        {/* Input Area */}
        <div className="chat-input-container">
          <form onSubmit={handleQuerySubmit} className="chat-input-wrapper">
            <input
              type="text"
              placeholder={documents.length === 0 ? "Upload PDFs to start chatting..." : "Ask something about the uploaded PDFs..."}
              className="chat-input"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={documents.length === 0 || thinking}
            />
            <button 
              type="submit" 
              className="send-btn" 
              disabled={!query.trim() || documents.length === 0 || thinking}
            >
              ➔
            </button>
          </form>
          <div className="input-hint-text">
            Documents marked with High priority are favored. Document citations are shown in the right sidebar.
          </div>
        </div>
      </main>

      {/* Right Citation Panel */}
      <aside className="right-panel">
        {activeCitations.length === 0 ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", flexGrow: 1, padding: "2rem", color: "var(--text-muted)", fontSize: "0.85rem", textAlign: "center", gap: "0.75rem" }}>
            <span style={{ fontSize: "2.5rem" }}>📖</span>
            <span style={{ fontWeight: 600, color: "var(--text-secondary)" }}>No active citations</span>
            <span>Citations for the selected chat message will appear here. Ask a query or select a message to see sources.</span>
          </div>
        ) : (
          <>
            <div className="right-panel-header">
              <h3 className="right-panel-title">Sources & Citations</h3>
              <span className="right-panel-subtitle">Showing {activeCitations.length} matched fragments</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", overflowY: "auto", flexGrow: 1, paddingRight: "0.25rem" }}>
              {activeCitations.map((c) => (
                <div key={c.index} className="citation-card">
                  <div className="citation-card-header">
                    <span className="citation-number">{c.index}</span>
                    <span className="citation-card-docname" title={c.doc_name}>{c.doc_name}</span>
                    <span className={`doc-stat-tag`} style={{ fontSize: "0.7rem", padding: "0.1rem 0.3rem" }}>
                      {c.priority}
                    </span>
                  </div>
                  <div className="citation-card-meta">
                    <span>Page {c.page} ({c.type})</span>
                    <span>Score: <strong style={{ color: "var(--accent-secondary)" }}>{c.boosted_score.toFixed(3)}</strong></span>
                  </div>
                  
                  {c.type === "image" && c.image_path && (
                    <img 
                      src={`${BACKEND_URL}${c.image_path}`}
                      alt="Extracted asset"
                      className="citation-card-image"
                    />
                  )}
                  
                  <div className="citation-card-snippet">
                    {c.content}
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </aside>
    </div>
  );
}
