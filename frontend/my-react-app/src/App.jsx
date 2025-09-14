import React, { useState, useRef, useEffect } from 'react';
import { Upload, Play, FileText, Brain, Users, Zap, Download, MessageCircle, Video, PenTool, BookOpen, Award, ChevronRight, Star, Clock, Target } from 'lucide-react';
import ReactMarkdown from "react-markdown";
import './styles.css';
import "katex/dist/katex.min.css";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
console.log("Frontend API_BASE =", API_BASE);
const WS_BASE = API_BASE.replace(/^http/, "ws");

const LearnWise = () => {
  const [videoUrl, setVideoUrl] = useState('');
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState('transcript');
  const [chatMessage, setChatMessage] = useState('');
  const [activeUploadType, setActiveUploadType] = useState('video');
  const [selectedLLM, setSelectedLLM] = useState('llama-3-8b');
  const fileInputRef = useRef(null);
  const pdfInputRef = useRef(null);
  const chatEndRef = useRef(null);
  const [transcript, setTranscript] = useState([]);
  const [quiz, setQuiz] = useState([]);
  const [sid, setSid] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [summary, setSummary] = useState("");
  const [werSummary, setWerSummary] = useState(null);
  const [evalSid, setEvalSid] = useState("");
  const [isEvalRunning, setIsEvalRunning] = useState(false);
  const [evalLimit, setEvalLimit] = useState(50);
  const [werRows, setWerRows] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState({
    transcript: 'disconnected',
    summary: 'disconnected',
    quiz: 'disconnected'
  });
  // in the Chat tab:
const sendChat = async () => {
  if (!sid || !chatMessage.trim()) return;

  const userMsg = { role: "user", text: chatMessage };
  setChatHistory(prev => [...prev, userMsg]);  // show user question immediately
  const q = chatMessage;  // capture
  setChatMessage("");     // clear input

  try {
    const res = await fetch(`${API_BASE}/qa`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sid, query: q })
    });

    if (!res.ok) throw new Error(`QA failed with status ${res.status}`);
    const data = await res.json();

    const aiMsg = { role: "ai", text: data.answer || "(no answer)" };
    setChatHistory(prev => [...prev, aiMsg]);  // append AI reply
  } catch (e) {
    console.error("QA failed:", e);
    setChatHistory(prev => [...prev, { role: "ai", text: "⚠️ Error getting answer" }]);
  }
};

  const startTedliumEval = async () => {
  try {
    setIsEvalRunning(true);
    const res = await fetch(`${API_BASE}/eval/tedlium`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: evalLimit }) // or omit to run full set
    });
    if (!res.ok) throw new Error(`Eval start failed: ${res.status}`);
    const data = await res.json();
    setEvalSid(data.session_id);     // IMPORTANT: use evalSid, not sid
    setWerSummary(null);
    setWerRows([]);
    setActiveTab("wer");             // jump to WER tab
  } catch (e) {
    console.error("startTedliumEval error:", e);
    alert("Failed to start TEDLIUM eval. Check backend logs.");
  } finally {
    setIsEvalRunning(false);
  }
};
  const cleanSummary = (raw) => {
  if (!raw) return "";
  let s = String(raw);

  // 1) Strip preamble noise
  const firstHeading = s.search(/(^|\n)#{1,6}\s/m);
  if (firstHeading > -1) s = s.slice(firstHeading);

  // 2) Remove known junk blocks
  s = s.replace(/Transcript window:[\s\S]*?(?=\n#|$)/m, "");
  s = s.replace(/Retrieved references:[\s\S]*?(?=\n#|$)/m, "");
  s = s.replace(/^LaTeX equations? will be rendered.*$/gim, "");

  // 3) Turn ```math fences (or bare ``` around TeX) into $$ ... $$
  // Only if the body looks like TeX (has \, ^, _, \frac, \begin, $, etc.)
  s = s.replace(/```(?:\s*math)?\s*([\s\S]*?)```/g, (_, body) => {
    const trimmed = body.trim();
    const looksLikeTex = /\\[a-zA-Z]+|\\begin|\\frac|[_^]|^\$|\$/.test(trimmed);
    return looksLikeTex ? `\n$$\n${trimmed}\n$$\n` : `\n\`\`\`\n${trimmed}\n\`\`\`\n`;
  });

  // 4) Convert common TeX envs → $$ ... $$
  // \[ ... \], \( ... \), and \begin{equation*?}/align*?/gather*?/multline*?
  s = s
    .replace(/\\\[\s*([\s\S]*?)\s*\\\]/g, (_, m) => `\n$$\n${m.trim()}\n$$\n`)
    .replace(/\\\(\s*([\s\S]*?)\s*\\\)/g, (_, m) => `$${m.trim()}$`)
    // equation / align / gather / multline with optional *
    .replace(/\\begin\{(equation|align|gather|multline)\*?\}\s*([\s\S]*?)\\end\{\1\*?\}/g,
      (_, _env, body) => `\n$$\n${body.trim()}\n$$\n`
    );

  // 5) Auto-headings: promote likely section titles to markdown headings
  const lines = s.split(/\r?\n/);
  let seenTitle = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    if (/^#{1,6}\s/.test(line)) { seenTitle = true; continue; }

    // Short capitalized line with no trailing punctuation → heading
    if (/^[A-Z][A-Za-z0-9\s&'().-]{2,80}$/.test(line) && !/[.:;!?]$/.test(line)) {
      lines[i] = (seenTitle ? "## " : "# ") + line;
      seenTitle = true;
    }
  }
  s = lines.join("\n");

  // 6) Normalize whitespace
  s = s.replace(/[ \t]+\n/g, "\n");
  s = s.replace(/\n{3,}/g, "\n\n").trim();

  return s;
};


const fmtTime = (t) => {
  const s = Math.max(0, Math.floor(Number(t || 0)));
  const m = Math.floor(s / 60);
  const ss = (s % 60).toString().padStart(2, "0");
  return `${m}:${ss}`;
};

const splitSentences = (txt) =>
  (txt || "")
    .replace(/\s+/g, " ")
    .trim()
    .split(/(?<=[.?!])\s+(?=[A-Z0-9(])/);

const groupBySection = (chunks) => {
  const buckets = {};
  for (const c of chunks) {
    const sec = Math.floor((Number(c.start || 0)) / 120);
    (buckets[sec] ||= []).push(c);
  }
  // sort by time
  Object.values(buckets).forEach(arr => arr.sort((a,b)=>Number(a.start||0)-Number(b.start||0)));
  return Object.entries(buckets).sort((a,b)=>Number(a[0])-Number(b[0]));
};
  useEffect(() => {
  if (chatEndRef.current) {
    chatEndRef.current.scrollTop = chatEndRef.current.scrollHeight;
  }
}, [chatHistory]);

useEffect(() => {
  if (!evalSid) return;
  const ws = new WebSocket(`${WS_BASE}/ws/wer/${evalSid}`);

  ws.onmessage = (event) => {
    try {
      const raw = JSON.parse(event.data);
      const type = raw?.type || null;
      const payload = raw?.payload ?? raw;

      if (type === "WER_SNAPSHOT") {
        setWerSummary(payload.summary || null);
        setWerRows(Array.isArray(payload.rows) ? payload.rows : []);
      } else if (type === "WER_PROGRESS") {
        setWerRows(prev => [payload, ...prev].slice(0, 500)); // newest first
      } else if (type === "WER_DONE") {
        setWerSummary(payload);
      }
    } catch (e) {
      console.error("WER WS parse error:", e, event.data);
    }
  };

  ws.onerror = (err) => console.error("WER WS error:", err);
  return () => ws.close();
}, [evalSid]);


  useEffect(() => {
    if (!sid) return;

    console.log("WebSocket URLs:", {
      transcript: `${WS_BASE}/ws/transcript/${sid}`,
      summary: `${WS_BASE}/ws/summary/${sid}`,
      quiz: `${WS_BASE}/ws/quiz/${sid}`
    });
  

        // TRANSCRIPT
    const wsTranscript = new WebSocket(`${WS_BASE}/ws/transcript/${sid}`);
    wsTranscript.onopen = () => {
      console.log("Transcript WS connected");
      setConnectionStatus(prev => ({...prev, transcript: 'connected'}));
    };
    wsTranscript.onclose = () => {
      console.log("Transcript WS closed");
      setConnectionStatus(prev => ({...prev, transcript: 'disconnected'}));
    };
    wsTranscript.onmessage = (event) => {
  try {
    const msg = JSON.parse(event.data);
    console.log("Transcript message:", msg);

    if (msg.transcript) {
      // full transcript
      setTranscript(msg.transcript);
    } else if (msg.text) {
      // streamed chunk
      setTranscript(prev => [...prev, {
        text: msg.text,
        start: msg.start,
        end: msg.end,
        section: msg.section ?? Math.floor((msg.start || 0) / 120),
        i: msg.i ?? prev.length
      }]);
    }
  } catch (e) {
    console.error("Transcript parse error:", e, event.data);
  }
};

    wsTranscript.onerror = (error) => {
      console.error("Transcript WebSocket error:", error);
      setConnectionStatus(prev => ({...prev, transcript: 'error'}));
    };

    // SUMMARY
    const wsSummary = new WebSocket(`${WS_BASE}/ws/summary/${sid}`);
    wsSummary.onopen = () => {
      console.log("Summary WS connected");
      setConnectionStatus(prev => ({...prev, summary: 'connected'}));
    };
    wsSummary.onclose = () => {
      console.log("Summary WS closed");
      setConnectionStatus(prev => ({...prev, summary: 'disconnected'}));
    };
    wsSummary.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log("Summary message:", msg);
        
        if (msg.notes_md) {
          setSummary(msg.notes_md);
        } else if (msg.summary) {
          setSummary(msg.summary);
        } else if (msg.text) {
          setSummary(msg.text);
        }
      } catch (e) {
        console.error("Summary parse error:", e, event.data);
      }
    };
    wsSummary.onerror = (error) => {
      console.error("Summary WebSocket error:", error);
      setConnectionStatus(prev => ({...prev, summary: 'error'}));
    };

    // QUIZ - FIXED: No need for JSON extraction anymore
    const wsQuiz = new WebSocket(`${WS_BASE}/ws/quiz/${sid}`);
    wsQuiz.onopen = () => {
      console.log("Quiz WS connected");
      setConnectionStatus(prev => ({...prev, quiz: 'connected'}));
    };
    wsQuiz.onclose = () => {
      console.log("Quiz WS closed");
      setConnectionStatus(prev => ({...prev, quiz: 'disconnected'}));
    };
   wsQuiz.onmessage = (event) => {
    try {
      // most of the time this is JSON text already
      let data = event.data;
      if (typeof data === "string") {
        try { data = JSON.parse(data); } catch {}
      }

      // shape can be: {questions:[...]}, [...] or a single object
      let questions = [];
      if (data && Array.isArray(data.questions)) {
        questions = data.questions;
      } else if (Array.isArray(data)) {
        questions = data;
      } else if (data && typeof data === "object") {
        questions = [data];
      }

      // normalize (choices -> options)
      const normalized = questions.map(q => {
        if (!q || typeof q !== "object") return null;
        const options = q.options || q.choices || null;
        return { ...q, options };
      }).filter(Boolean);

      console.log("Quiz (normalized):", normalized);
      setQuiz(normalized);
    } catch (e) {
      console.error("Quiz parse error:", e, event.data);
      setQuiz([]);
    }
  };

    wsQuiz.onerror = (error) => {
      console.error("Quiz WebSocket error:", error);
      setConnectionStatus(prev => ({...prev, quiz: 'error'}));
    };

    // Cleanup
    return () => {
      wsTranscript.close();
      wsSummary.close();
      wsQuiz.close();
    };
  }, [sid]);

  
  const startIngest = async () => {
    setIsProcessing(true);
    try {
      const res = await fetch(`${API_BASE}/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: videoUrl, selected_model: selectedLLM })
      });

      if (!res.ok) {
        throw new Error(`Ingest failed with status ${res.status}`);
      }

      const { session_id } = await res.json();
      console.log("✅ New session started:", session_id);
      setSid(session_id);
      setTranscript([]);
      setSummary("");
      setQuiz([]);

    } catch (err) {
      console.error("❌ Ingest error:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (event, type) => {
    const files = Array.from(event.target.files);
    if (files.length > 0) {
      if (type === "document") {
        setIsProcessing(true);
        const formData = new FormData();
        formData.append("selected_model", selectedLLM);
        formData.append("file", files[0]);

        try {
          const res = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
          });
          const data = await res.json();
          setSid(data.session_id);
          setTranscript([]);
          setSummary("");
          setQuiz([]);
        } catch (err) {
          console.error("Error uploading file:", err);
        } finally {
          setIsProcessing(false);
        }
      }

      const newFiles = files.map(file => ({
        name: file.name,
        type: type,
        size: (file.size / 1024 / 1024).toFixed(2) + ' MB',
        file: file
      }));
      setUploadedFiles(prev => [...prev, ...newFiles]);
      if (type === 'video') {
        setVideoUrl(`Uploaded: ${files[0].name}`);
      }
    }
  };

  const handleRemoveFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI-Powered Transcription",
      description: "Advanced Whisper model extracts clear, accurate transcripts from any lecture video"
    },
    {
      icon: <FileText className="w-8 h-8" />,
      title: "Smart Note Generation",
      description: "Fine-tuned LLaMA generates structured, comprehensive study notes automatically"
    },
    {
      icon: <PenTool className="w-8 h-8" />,
      title: "Document Analysis",
      description: "Upload PDFs, slides, and documents for enhanced context and comprehensive analysis"
    },
    {
      icon: <Users className="w-8 h-8" />,
      title: "Multi-Agent Architecture",
      description: "Specialized AI agents collaborate for transcription, summarization, and evaluation"
    },
    {
      icon: <BookOpen className="w-8 h-8" />,
      title: "RAG Integration",
      description: "Enriched explanations using lecture slides, PDFs, and textbook references"
    },
    {
      icon: <Award className="w-8 h-8" />,
      title: "Interactive Learning",
      description: "Real-time Q&A, difficulty-tagged quizzes, and personalized explanations"
    }
  ];

  const agents = [
    { name: "Transcriber Agent", role: "Video → Transcript", color: "from-blue-500 to-cyan-500" },
    { name: "Summarizer Agent", role: "Transcript → Notes", color: "from-purple-500 to-pink-500" },
    { name: "Explainer Agent", role: "Interactive Clarifications", color: "from-green-500 to-emerald-500" },
    { name: "Quiz Agent", role: "Question Generation", color: "from-orange-500 to-red-500" },
    { name: "Evaluator Agent", role: "Quality Assessment", color: "from-indigo-500 to-purple-500" }
  ];

  const llmModels = [
    { 
      id: "llama-3-8b", 
      name: "LLaMA-3 8B", 
      speciality: "General Purpose", 
      description: "Fast and efficient for most educational use-cases.", 
      performance: "Fast", 
      quality: "High" 
    },
    { 
      id: "llama-3-70b", 
      name: "LLaMA-3 70B", 
      speciality: "Complex Reasoning", 
      description: "Best for deep reasoning, nuanced explanations, and advanced topics.", 
      performance: "Slow", 
      quality: "Highest" 
    },
    { 
      id: "mistral-7b", 
      name: "Mistral 7B", 
      speciality: "Code & Math", 
      description: "Optimized for structured reasoning and code understanding.", 
      performance: "Fast", 
      quality: "High" 
    },
    { 
      id: "codellama-7b", 
      name: "CodeLLaMA 7B", 
      speciality: "Programming", 
      description: "Great for code generation, explanations, and debugging.", 
      performance: "Fast", 
      quality: "High" 
    },
    { 
      id: "openchat-7b", 
      name: "OpenChat 7B", 
      speciality: "Conversational", 
      description: "Engages in conversational learning and clarifications.", 
      performance: "Fast", 
      quality: "High" 
    }
  ];

  const selectedModel = llmModels.find(m => m.id === selectedLLM);

  return (
    <>
    {/* Header */}
    <header className="header">
      <div className="container">
        <div className="nav">
          <div className="logo">
            <Brain className="w-8 h-8" />
            <span className="logo-text">LearnWise</span>
          </div>
          <nav className="nav-links">
            <a href="#features">Features</a>
            <a href="#agents">AI Agents</a>
            <a href="#platform">Platform</a>
            <a href="#contact">Contact</a>
          </nav>
        </div>
      </div>
    </header>
  
      {/* Hero Section */}
      <section className="hero">
        <div className="container">
          <div className="hero-content">
            <div className="hero-text">
              <h1 className="hero-title">
                Transform Your Learning with
                <span className="gradient-text"> AI-Powered</span> Lecture Analysis
              </h1>
              <p className="hero-description">
                Upload lecture videos, PDFs, slides, and documents to get instant transcripts, structured notes, exam questions, and interactive explanations. 
                Powered by fine-tuned Whisper, LLaMA-3, and multi-agent architecture with advanced document analysis.
              </p>
            </div>
            <div className="hero-demo">
              <div className="demo-card">
                <div className="demo-header">
                  <div className="demo-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <span className="demo-title">LearnWise Platform</span>
                </div>
                <div className="demo-content">
                  <div className="upload-section">
                    {/* Model Selection Section */}
                    <div className="model-selection">
                      <h4>🤖 Select AI Model</h4>
                      <div className="model-dropdown">
                        <select 
                          value={selectedLLM} 
                          onChange={(e) => setSelectedLLM(e.target.value)}
                          className="model-select"
                        >
                          {llmModels.map(model => (
                            <option key={model.id} value={model.id}>
                              {model.name} - {model.speciality}
                            </option>
                          ))}
                        </select>
                        {selectedModel && (
                          <div className="model-info">
                            <div className="model-details">
                              <span className="model-desc">
                                {selectedModel.description}
                              </span>
                              <div className="model-badges">
                                <span className={`badge ${selectedModel.performance.toLowerCase()}`}>
                                  {selectedModel.performance}
                                </span>
                                <span className="badge quality">
                                  {selectedModel.quality}
                                </span>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Upload Type Selection */}
                    <div className="upload-tabs">
                      <button 
                        className={`upload-tab ${activeUploadType === 'video' ? 'active' : ''}`}
                        onClick={() => setActiveUploadType('video')}
                      >
                        <Video className="w-5 h-5" />
                        Video/YouTube
                      </button>
                      <button 
                        className={`upload-tab ${activeUploadType === 'documents' ? 'active' : ''}`}
                        onClick={() => setActiveUploadType('documents')}
                      >
                        <FileText className="w-5 h-5" />
                        PDF/Slides
                      </button>
                    </div>

                    {/* Upload Areas */}
                    {activeUploadType === 'video' ? (
                      <>
                        <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
                          <Video className="w-12 h-12" />
                          <p>Drop your lecture video here or click to upload</p>
                          <span className="upload-formats">MP4, AVI, MOV, WebM</span>
                        </div>
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="video/*"
                          onChange={(e) => handleFileUpload(e, 'video')}
                          style={{ display: 'none' }}
                        />
                        <div className="url-input">
                          <input
                            type="text"
                            value={videoUrl}
                            onChange={(e) => setVideoUrl(e.target.value)}
                            placeholder="Or paste YouTube URL here..."
                            className="input"
                          />
                          <button
                            onClick={startIngest}
                            disabled={isProcessing || (!videoUrl.trim() && uploadedFiles.filter(f => f.type === 'video').length === 0)}
                            className="btn btn-primary"
                          >
                            {isProcessing ? <div className="spinner"></div> : <Play className="w-5 h-5" />}
                            {isProcessing ? 'Processing...' : `Process with ${selectedModel?.name.split(' ')[0] || 'AI'}`}
                          </button>
                        </div>
                      </>
                    ) : (
                      <>
                        <div className="upload-area" onClick={() => pdfInputRef.current?.click()}>
                          <FileText className="w-12 h-12" />
                          <p>Upload lecture slides, PDFs, or documents</p>
                          <span className="upload-formats">PDF, PPTX, PPT, DOC, DOCX, TXT</span>
                        </div>
                        <input
                          ref={pdfInputRef}
                          type="file"
                          accept=".pdf,.pptx,.ppt,.doc,.docx,.txt"
                          multiple
                          onChange={(e) => handleFileUpload(e, 'document')}
                          style={{ display: 'none' }}
                        />
                        <div className="url-input">
                          <button
                            onClick={() => startIngest()}  
                            disabled={isProcessing || uploadedFiles.filter(f => f.type === 'document').length === 0}
                            className="btn btn-primary full-width"
                          >
                            {isProcessing ? <div className="spinner"></div> : <Brain className="w-5 h-5" />}
                            {isProcessing ? 'Processing...' : `Analyze with ${selectedModel?.name.split(' ')[0] || 'AI'}`}
                          </button>
                        </div>
                      </>
                    )}

                    {/* Uploaded Files Display */}
                    {uploadedFiles.length > 0 && (
                      <div className="uploaded-files">
                        <h4>Uploaded Files ({uploadedFiles.length}):</h4>
                        {uploadedFiles.map((file, index) => (
                          <div key={index} className="file-item">
                            <div className="file-info">
                              {file.type === 'video' ? <Video className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
                              <span className="file-name">{file.name}</span>
                              <span className="file-size">{file.size}</span>
                            </div>
                            <button 
                              onClick={() => handleRemoveFile(index)}
                              className="remove-file"
                              title="Remove file"
                            >
                              ×
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Platform Interface - Moved right after Hero */}
      <section id="platform" className="platform">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Interactive Learning Platform</h2>
            <p className="section-description">
              Real-time processing with WebSocket streaming and comprehensive output options
            </p>
          </div>
          <div className="platform-interface">
            <div className="platform-sidebar">
              <div className="tabs">
                {[
                  { id: 'transcript', icon: <FileText className="w-5 h-5" />, label: 'Transcript' },
                  { id: 'notes', icon: <BookOpen className="w-5 h-5" />, label: 'Smart Notes' },
                  { id: 'questions', icon: <PenTool className="w-5 h-5" />, label: 'Quiz Questions' },
                  { id: 'chat', icon: <MessageCircle className="w-5 h-5" />, label: 'AI Chat' },
                  { id: 'wer', icon: <Award className="w-5 h-5" />, label: 'WER' },
                ].map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`tab ${activeTab === tab.id ? 'tab-active' : ''}`}
                  >
                    {tab.icon}
                    {tab.label}
                  </button>
                ))}
              </div>
              <div className="export-options">
                <h4>Export Options</h4>
                <button className="export-btn">
                  <Download className="w-4 h-4" />
                  PDF Notes
                </button>
                <button className="export-btn">
                  <Download className="w-4 h-4" />
                  Quiz Set
                </button>
                <button className="export-btn">
                  <Download className="w-4 h-4" />
                  PowerPoint
                </button>
                <div className="evaluation-section">
                  <h4>Evaluate Speech Transcription</h4>
                  <div className="eval-controls">
                    <div className="eval-input-group">
                      <label htmlFor="evalLimit">Test samples:</label>
                      <input
                        id="evalLimit"
                        type="number"
                        min={1}
                        max={500}
                        value={evalLimit}
                        onChange={(e) => setEvalLimit(parseInt(e.target.value || "1", 10))}
                        className="eval-input"
                        title="How many test utterances (optional)"
                      />
                    </div>
                    <button
                      onClick={startTedliumEval}
                      disabled={isEvalRunning}
                      className="btn-eval"
                    >
                      {isEvalRunning ? (
                        <>
                          <div className="spinner-small"></div>
                          Running...
                        </>
                      ) : (
                        'Run Eval'
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </div>
            <div className="platform-content">
              {activeTab === 'transcript' && (
                <div className="content-panel">
                  <h3>Video Transcript</h3>
                  <div className="transcript-content">
                    {transcript.length === 0 ? (
                      <p>No transcript yet…</p>
                    ) : (
                      groupBySection(transcript).map(([sec, items]) => {
                          const startSec = Number(sec) * 120;
                          const endSec = startSec + 120;
                          const headerFrom = fmtTime(startSec);
                          const headerTo = fmtTime(endSec);
                        return (
                          <div key={sec} style={{marginBottom:"1rem"}}>
                            <div style={{fontWeight:700, color:"#6366f1", margin:"8px 0"}}>
                              {headerFrom}–{headerTo}
                            </div>
                            {items.map((chunk, idx) => (
                              <div key={`${sec}-${idx}`} className="transcript-chunk" style={{marginBottom:"0.5rem"}}>
                                <p className="timestamp">[{fmtTime(chunk.start)}]</p>
                                <p>
                                  {splitSentences(chunk.text).map((s, i) => (
                                    <span key={i} style={{display:"inline"}}>{s} </span>
                                  ))}
                                </p>
                              </div>
                            ))}
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              )}
              {activeTab === 'notes' && (
                <div className="content-panel">
                  <h3>AI-Generated Notes</h3>
                  <div className="notes-content">
                    {summary ? (
                        <>
                          {console.log("Rendering summary:", summary)}
                          <ReactMarkdown
                            remarkPlugins={[remarkMath, remarkGfm]}
                            rehypePlugins={[[rehypeKatex, { strict: false, displayMode: true  }]]}
                            >
                              {cleanSummary(summary)}
                          </ReactMarkdown>
                        </>
                          ) : (
                            <p>No summary yet…</p>
                    )}
                  </div>
                </div>
              )}
              {activeTab === 'questions' && (
                <div className="content-panel">
                  <h3>Practice Questions</h3>
                  <div className="questions-content">
                    {quiz.length > 0 ? (
                      quiz.map((q, i) => (
                        <div key={i} className="question">
                          <p><strong>{q.type || "Q"} {i+1}:</strong> {q.question || "No question text"}</p>
                          {q.options && Array.isArray(q.options) && (
                            <div className="options">
                              {q.options.map((opt, idx) => (
                                <label key={idx}>
                                  <input type="radio" name={`q${i}`} /> {opt}
                                </label>
                              ))}
                            </div>
                          )}
                          {q.answer && (
                            <p style={{marginTop:"0.5rem", fontStyle:"italic"}}>Answer: {q.answer}</p>
                          )}
                          {q.rationale && (
                            <p style={{opacity:0.8}}>Why: {q.rationale}</p>
                          )}
                        </div>
                      ))
                    ) : (
                      <p>No questions yet…</p>
                    )}
                  </div>
                </div>
              )}
               {activeTab === 'chat' && (
                <div className="content-panel">
                  <h3>Ask AI Anything</h3>
                  <div className="chat-content">
                    <div className="chat-messages" style={{ maxHeight: "400px", overflowY: "auto" }} ref={chatEndRef}>
                      {chatHistory.map((msg, i) => (
                        <div key={i} className={`chat-message ${msg.role}`}>
                          {msg.role === "ai" ? <Brain className="w-6 h-6" /> : null}
                          <p>{msg.text}</p>
                        </div>
                      ))}
                      <div ref={chatEndRef} />
                    </div>
                    <div className="chat-input">
                      <input
                        type="text"
                        value={chatMessage}
                        onChange={(e) => setChatMessage(e.target.value)}
                        placeholder="Ask about linear regression, cost functions..."
                        className="input"
                        onKeyDown={(e) => {
                          if (e.key === "Enter") {
                            e.preventDefault();
                            sendChat();
                          }
                        }}
                      />
                      <button className="btn btn-primary" onClick={sendChat}>Send</button>
                    </div>
                  </div>
                </div>
              )}
              {activeTab === 'wer' && (
                <div className="content-panel">
                  <h3>Evaluation:- Word Error Rate</h3>

                  <div className="wer-summary-card">
                    {werSummary ? (
                      <div className="wer-stats-grid">
                        <div className="wer-stat-main">
                          <div className="wer-percentage">
                            {(werSummary.corpus_wer * 100).toFixed(2)}%
                          </div>
                          <div className="wer-label">Corpus WER</div>
                        </div>
                        <div className="wer-stats-details">
                          <div className="wer-stat-item">
                            <span className="stat-label">Ref words</span>
                            <span className="stat-value">{werSummary.total_ref}</span>
                          </div>
                          <div className="wer-stat-item">
                            <span className="stat-label">Hyp words</span>
                            <span className="stat-value">{werSummary.total_hyp}</span>
                          </div>
                          <div className="wer-stat-item">
                            <span className="stat-label">Errors</span>
                            <span className="stat-value">{werSummary.total_err}</span>
                          </div>
                          {typeof werSummary.runtime_sec === 'number' && (
                            <div className="wer-stat-item">
                              <span className="stat-label">Runtime</span>
                              <span className="stat-value">{werSummary.runtime_sec.toFixed(1)}s</span>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="wer-empty-state">
                        <Award className="w-12 h-12" />
                        <h4>No evaluation results yet</h4>
                        <p>Click "Run Eval" in the sidebar to start speech recognition evaluation</p>
                      </div>
                    )}
                  </div>

                  <div className="wer-table-container">
                    <div className="wer-table-header">
                      <h4>Detailed Results</h4>
                      <span className="results-count">
                        {werRows.length} {werRows.length === 1 ? 'result' : 'results'}
                      </span>
                    </div>
                    
                    <div className="wer-table-wrapper">
                      <table className="wer-table">
                        <thead>
                          <tr>
                            <th>Utterance ID</th>
                            <th>WER %</th>
                            <th>Errors</th>
                            <th>Ref Words</th>
                            <th>Hyp Words</th>
                          </tr>
                        </thead>
                        <tbody>
                          {werRows.length === 0 ? (
                            <tr>
                              <td colSpan={5} className="no-data">
                                No utterance data yet...
                              </td>
                            </tr>
                          ) : (
                            werRows.map((r, i) => (
                              <tr key={i} className={r.wer > 0.5 ? 'high-error' : r.wer === 0 ? 'perfect' : ''}>
                                <td className="utterance-id" title={r.audio_id || r.utt_id}>
                                  {(r.audio_id || r.utt_id || `row-${i}`).substring(0, 30)}
                                  {(r.audio_id || r.utt_id || '').length > 30 ? '...' : ''}
                                </td>
                                <td className="wer-value">
                                  <span className={`wer-badge ${r.wer > 0.5 ? 'high' : r.wer === 0 ? 'perfect' : 'good'}`}>
                                    {(r.wer * 100).toFixed(1)}%
                                  </span>
                                </td>
                                <td>{r.errors}</td>
                                <td>{r.ref_len}</td>
                                <td>{r.hyp_len}</td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                    
                    <div className="wer-table-footer">
                      <p>Rows stream live during evaluation. Lower WER percentages indicate better accuracy.</p>
                    </div>
                  </div>
                </div>
              )}

            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Powerful AI Features</h2>
            <p className="section-description">
              Built with state-of-the-art machine learning models and fine-tuned for educational content
            </p>
          </div>
          <div className="features-grid">
            {features.map((feature, index) => (
              <div key={index} className="feature-card">
                <div className="feature-icon">
                  {feature.icon}
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* AI Agents Section */}
      <section id="agents" className="agents">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Multi-Agent Architecture</h2>
            <p className="section-description">
              Specialized AI agents work together to deliver comprehensive learning assistance
            </p>
          </div>
          <div className="agents-flow">
            {agents.map((agent, index) => (
              <div key={index} className="agent-card">
                <div className={`agent-avatar bg-gradient-to-br ${agent.color}`}>
                  <Users className="w-6 h-6" />
                </div>
                <div className="agent-info">
                  <h3 className="agent-name">{agent.name}</h3>
                  <p className="agent-role">{agent.role}</p>
                </div>
                {index < agents.length - 1 && (
                  <ChevronRight className="agent-arrow w-6 h-6" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technical Specs */}
      <section className="tech-specs">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title" style={{ color: 'white' }}>Choose Your AI Model</h2>
            <p className="section-description" style={{ color: '#d1d5db' }}>
              Select from our fine-tuned models optimized for different learning scenarios
            </p>
          </div>
          <div className="model-comparison">
            {llmModels.map((model, index) => (
              <div key={model.id} className="model-card">
                <div className="model-header">
                  <h3>{model.name}</h3>
                  <div className="model-specialty">{model.speciality}</div>
                </div>
                <p className="model-description">{model.description}</p>
                <div className="model-stats">
                  <div className="stat-item">
                    <Clock className="w-4 h-4" />
                    <span>Speed: {model.performance}</span>
                  </div>
                  <div className="stat-item">
                    <Star className="w-4 h-4" />
                    <span>Quality: {model.quality}</span>
                  </div>
                </div>
                <div className="model-features">
                  {model.id === 'llama-3-8b' && (
                    <>
                      <div className="feature">✓ General-purpose fine-tuning</div>
                      <div className="feature">✓ Fast inference</div>
                      <div className="feature">✓ Balanced performance</div>
                    </>
                  )}
                  {model.id === 'llama-3-70b' && (
                    <>
                      <div className="feature">✓ Advanced reasoning</div>
                      <div className="feature">✓ Complex problem solving</div>
                      <div className="feature">✓ Highest accuracy</div>
                    </>
                  )}
                  {model.id === 'mistral-7b' && (
                    <>
                      <div className="feature">✓ Math & code optimization</div>
                      <div className="feature">✓ Structured reasoning</div>
                      <div className="feature">✓ Technical content</div>
                    </>
                  )}
                  {model.id === 'codellama-7b' && (
                    <>
                      <div className="feature">✓ Programming lectures</div>
                      <div className="feature">✓ Technical documentation</div>
                      <div className="feature">✓ Code explanation</div>
                    </>
                  )}
                  {model.id === 'openchat-7b' && (
                    <>
                      <div className="feature">✓ Interactive Q&A</div>
                      <div className="feature">✓ Conversational learning</div>
                      <div className="feature">✓ Student engagement</div>
                    </>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="specs-grid" style={{ marginTop: '4rem' }}>
            <div className="spec-card">
              <Zap className="w-8 h-8" />
              <h3>Local Processing</h3>
              <p>No cloud dependencies. Everything runs on your machine with llama.cpp and faster-whisper.</p>
            </div>
            <div className="spec-card">
              <Target className="w-8 h-8" />
              <h3>Fine-tuned Models</h3>
              <p>Custom LoRA adapters trained specifically for educational content and question generation.</p>
            </div>
            <div className="spec-card">
              <Clock className="w-8 h-8" />
              <h3>Real-time Streaming</h3>
              <p>WebSocket-powered interface with live transcript and summary updates as processing happens.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <Brain className="w-8 h-8" />
              <span className="logo-text">LearnWise</span>
            </div>
            <div className="footer-text">
              <p>Transforming education with AI-powered lecture analysis and interactive learning.</p>
              <p>Built with ❤️ for students and educators worldwide.</p>
            </div>
          </div>
        </div>
      </footer>
      </>
  );
};

export default LearnWise;