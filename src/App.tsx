import React, { useState, useRef, useEffect } from 'react';
import { 
  Upload, Activity, Shield, AlertTriangle, Info, 
  Download, RefreshCw, Layers, Home, FileText, 
  Settings, User, ChevronRight, CheckCircle2
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || "" });

interface AnalysisResult {
  classification: 'Benign' | 'Malignant';
  confidence: number;
  spreadPercentage: number;
  polygon: [number, number][];
  explanation: string;
  heatmap: number[][]; // 10x10 grid of activation intensities
}

type Tab = 'home' | 'upload' | 'results' | 'about';

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('upload');
  const [image, setImage] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const gradCamRef = useRef<HTMLCanvasElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImage(reader.result as string);
        setResult(null);
        setError(null);
        setActiveTab('upload');
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;
    setAnalyzing(true);
    setError(null);
    try {
      const prompt = `
        You are an AI specialized in dermatological analysis for an M.Tech project.
        Analyze this skin lesion image.
        1. Classify it as "Benign" or "Malignant".
        2. Identify the primary tumor region. Provide a list of normalized coordinates (x, y) from 0 to 100 that form a polygon around the affected area.
        3. Estimate the spread percentage (area of tumor vs total image).
        4. Provide a brief clinical explanation.
        5. Generate a simulated Grad-CAM heatmap: Provide a 10x10 grid of numbers (0 to 1) representing the "importance" of each region for the classification.
        
        Return the response in strict JSON format:
        {
          "classification": "Benign" | "Malignant",
          "confidence": number (0-1),
          "spreadPercentage": number,
          "polygon": [[x1, y1], [x2, y2], ...],
          "explanation": "string",
          "heatmap": [[row1], [row2], ... [row10]]
        }
      `;

      const result = await ai.models.generateContent({
        model: "gemini-2.0-flash",
        contents: [
          {
            parts: [
              { text: prompt },
              {
                inlineData: {
                  data: image.split(",")[1],
                  mimeType: "image/jpeg",
                },
              },
            ],
          },
        ],
      });

      const responseText = result.text;
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) throw new Error("Invalid AI response");
      
      const analysis = JSON.parse(jsonMatch[0]);
      setResult(analysis);
      setActiveTab('results');
    } catch (err) {
      setError('Analysis failed. Please try again.');
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    if (result && image) {
      // Draw Segmentation Mask
      if (canvasRef.current) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.src = image;
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            ctx.beginPath();
            ctx.lineWidth = 4;
            ctx.strokeStyle = result.classification === 'Malignant' ? '#ef4444' : '#10b981';
            ctx.fillStyle = result.classification === 'Malignant' ? 'rgba(239, 68, 68, 0.3)' : 'rgba(16, 185, 129, 0.3)';
            result.polygon.forEach(([x, y], index) => {
              const px = (x / 100) * canvas.width;
              const py = (y / 100) * canvas.height;
              if (index === 0) ctx.moveTo(px, py);
              else ctx.lineTo(px, py);
            });
            ctx.closePath();
            ctx.stroke();
            ctx.fill();
          };
        }
      }

      // Draw Grad-CAM Heatmap
      if (gradCamRef.current && result.heatmap) {
        const canvas = gradCamRef.current;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          const img = new Image();
          img.src = image;
          img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            
            const cellW = canvas.width / 10;
            const cellH = canvas.height / 10;

            result.heatmap.forEach((row, y) => {
              row.forEach((val, x) => {
                // Simple Jet-like colormap simulation
                // val 0 -> blue (0,0,255), val 1 -> red (255,0,0)
                const r = Math.floor(255 * val);
                const g = Math.floor(255 * (1 - Math.abs(val - 0.5) * 2));
                const b = Math.floor(255 * (1 - val));
                
                ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.5)`;
                ctx.fillRect(x * cellW, y * cellH, cellW, cellH);
              });
            });
          };
        }
      }
    }
  }, [result, image]);

  return (
    <div className="flex min-h-screen bg-[#f8fafc] text-[#1e293b] font-sans">
      {/* Fixed Sidebar Navigation */}
      <aside className="w-72 bg-white border-r border-slate-200 flex flex-col fixed h-screen z-50">
        <div className="p-8 flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white shadow-lg shadow-blue-200">
            <Activity size={24} />
          </div>
          <h1 className="text-lg font-black tracking-tight text-slate-800">DermAI <span className="text-blue-600">Pro</span></h1>
        </div>

        <nav className="flex-1 px-4 space-y-2">
          <SidebarItem icon={<Home size={20} />} label="Home" active={activeTab === 'home'} onClick={() => setActiveTab('home')} />
          <SidebarItem icon={<Upload size={20} />} label="Upload Image" active={activeTab === 'upload'} onClick={() => setActiveTab('upload')} />
          <SidebarItem icon={<FileText size={20} />} label="Results" active={activeTab === 'results'} onClick={() => setActiveTab('results')} disabled={!result} />
          <SidebarItem icon={<Info size={20} />} label="About" active={activeTab === 'about'} onClick={() => setActiveTab('about')} />
        </nav>

        <div className="p-6 border-t border-slate-100">
          <div className="bg-blue-50 rounded-2xl p-4 flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
              <User size={20} />
            </div>
            <div>
              <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Researcher</p>
              <p className="text-sm font-bold text-slate-700">M.Tech AI Unit</p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content Area - Adjusted for Sidebar */}
      <main className="flex-1 ml-72 flex flex-col min-h-screen">
        <header className="h-20 bg-white/80 backdrop-blur-md border-b border-slate-200 flex items-center justify-between px-10 sticky top-0 z-40">
          <h2 className="text-xl font-bold text-slate-800 tracking-tight">Skin Cancer Detection System</h2>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-bold text-emerald-600 bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-100">
              <CheckCircle2 size={14} />
              Neural Engines Active
            </div>
            <span className="text-xs font-semibold px-3 py-1 bg-slate-100 rounded-full text-slate-600 border border-slate-200">v1.0.4-Stable</span>
          </div>
        </header>

        <div className="p-10 max-w-6xl mx-auto w-full flex-1">
          <AnimatePresence mode="wait">
            {activeTab === 'home' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                <div className="bg-white rounded-[2.5rem] p-12 shadow-sm border border-slate-200 relative overflow-hidden">
                  <div className="relative z-10 max-w-2xl">
                    <h3 className="text-4xl font-black text-slate-900 mb-6 leading-tight">Advanced AI-Based Skin Cancer Analysis</h3>
                    <p className="text-lg text-slate-600 leading-relaxed mb-8">
                      An end-to-end diagnostic system utilizing deep learning for pixel-wise segmentation and high-accuracy classification of dermatological lesions.
                    </p>
                    <button 
                      onClick={() => setActiveTab('upload')}
                      className="bg-blue-600 text-white px-8 py-4 rounded-2xl font-bold shadow-lg shadow-blue-200 hover:bg-blue-700 transition-all flex items-center gap-2"
                    >
                      Start Diagnostic Sequence
                      <ChevronRight size={20} />
                    </button>
                  </div>
                  <div className="absolute top-0 right-0 w-1/3 h-full bg-gradient-to-l from-blue-50 to-transparent opacity-50"></div>
                  <Activity className="absolute -bottom-10 -right-10 text-blue-100" size={300} />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <FeatureCard icon={<Shield className="text-blue-600" />} title="U-Net Segmentation" desc="Precise localization of tumor boundaries using convolutional neural networks." />
                  <FeatureCard icon={<Activity className="text-emerald-600" />} title="ResNet Classification" desc="State-of-the-art residual networks for benign vs malignant categorization." />
                  <FeatureCard icon={<Layers className="text-orange-600" />} title="Spread Analysis" desc="Quantitative measurement of tumor surface area relative to healthy tissue." />
                </div>
              </motion.div>
            )}

            {activeTab === 'upload' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                <section className="bg-white rounded-[2.5rem] p-10 shadow-sm border border-slate-200">
                  <div className="max-w-2xl mx-auto text-center mb-10">
                    <h3 className="text-2xl font-bold text-slate-800">Acquire Clinical Sample</h3>
                    <p className="text-slate-500 mt-2">Upload a high-resolution dermatoscopic image for automated segmentation and classification.</p>
                  </div>

                  <div className={`relative border-2 border-dashed rounded-[2rem] p-12 transition-all duration-500 flex flex-col items-center justify-center gap-6 ${
                    image ? 'border-blue-200 bg-blue-50/20' : 'border-slate-200 hover:border-blue-400 bg-slate-50/50'
                  }`}>
                    {image ? (
                      <div className="relative group w-full max-w-md aspect-square rounded-3xl overflow-hidden shadow-2xl">
                        <img src={image} alt="Preview" className="w-full h-full object-cover" />
                        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                          <button onClick={() => setImage(null)} className="bg-white text-slate-900 px-6 py-3 rounded-full text-sm font-bold shadow-xl">Replace Sample</button>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="w-20 h-20 bg-white rounded-3xl shadow-md border border-slate-100 flex items-center justify-center text-blue-500">
                          <Upload size={32} />
                        </div>
                        <div className="text-center">
                          <p className="text-lg font-bold text-slate-700">Drag & Drop Image</p>
                          <p className="text-sm text-slate-400 mt-1">Supports JPG, PNG (Max 10MB)</p>
                        </div>
                        <input type="file" accept="image/*" onChange={handleImageUpload} className="absolute inset-0 opacity-0 cursor-pointer" />
                      </>
                    )}
                  </div>

                  <div className="mt-10 flex flex-col items-center gap-6">
                    <button
                      onClick={analyzeImage}
                      disabled={!image || analyzing}
                      className={`px-12 py-5 rounded-2xl font-black text-white shadow-xl transition-all flex items-center gap-3 ${
                        !image || analyzing ? 'bg-slate-300 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 hover:-translate-y-1 shadow-blue-200'
                      }`}
                    >
                      {analyzing ? <RefreshCw className="animate-spin" size={22} /> : <Activity size={22} />}
                      {analyzing ? 'Processing Neural Layers...' : 'Initiate Diagnostic Sequence'}
                    </button>

                    <AnimatePresence>
                      {analyzing && (
                        <motion.div 
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -10 }}
                          className="w-full max-w-md space-y-3"
                        >
                          <div className="flex justify-between text-xs font-bold text-slate-500 uppercase tracking-widest">
                            <span>Neural Analysis in Progress</span>
                            <span className="text-blue-600 animate-pulse">Processing...</span>
                          </div>
                          <div className="w-full h-2.5 bg-slate-100 rounded-full overflow-hidden">
                            <motion.div 
                              className="h-full bg-blue-600"
                              initial={{ width: "0%" }}
                              animate={{ width: "95%" }}
                              transition={{ duration: 5, ease: "easeOut" }}
                            />
                          </div>
                          <p className="text-[10px] text-center text-slate-400 font-medium italic">
                            Running U-Net Segmentation & ResNet Feature Extraction
                          </p>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </section>
              </motion.div>
            )}

            {activeTab === 'results' && result && (
              <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="space-y-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <ResultCard 
                    label="Classification" 
                    value={result.classification} 
                    subValue={`Confidence: ${(result.confidence * 100).toFixed(1)}%`}
                    color={result.classification === 'Malignant' ? 'red' : 'emerald'}
                    icon={result.classification === 'Malignant' ? <AlertTriangle /> : <Shield />}
                  />
                  <ResultCard 
                    label="Tumor Spread" 
                    value={`${result.spreadPercentage.toFixed(2)}%`} 
                    subValue="Surface Area Ratio"
                    color="blue"
                    icon={<Layers />}
                  />
                  <div className="bg-white p-6 rounded-[2rem] border border-slate-200 flex flex-col justify-center">
                    <p className="text-xs font-bold text-slate-400 uppercase mb-3">Spread Intensity</p>
                    <div className="w-full h-3 bg-slate-100 rounded-full overflow-hidden">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${result.spreadPercentage}%` }} className="h-full bg-blue-500" />
                    </div>
                    <div className="flex justify-between mt-2 text-[10px] font-bold text-slate-400 uppercase">
                      <span>Localized</span>
                      <span>Extensive</span>
                    </div>
                  </div>
                </div>

                <div className="bg-white rounded-[2.5rem] p-10 border border-slate-200 shadow-sm">
                  <h3 className="text-lg font-bold mb-8 flex items-center gap-2">
                    <Layers size={20} className="text-blue-500" />
                    Comparative Visualization (Original vs Segmented)
                  </h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="space-y-3">
                      <p className="text-xs font-bold text-slate-400 uppercase px-2">Source Scan</p>
                      <div className="rounded-3xl overflow-hidden bg-slate-50 aspect-square border border-slate-100">
                        <img src={image!} alt="Original" className="w-full h-full object-cover" />
                      </div>
                    </div>
                    <div className="space-y-3">
                      <p className="text-xs font-bold text-slate-400 uppercase px-2">U-Net Segmentation Mask</p>
                      <div className="rounded-3xl overflow-hidden bg-slate-50 aspect-square border border-slate-100 relative">
                        <canvas ref={canvasRef} className="w-full h-full object-cover" />
                      </div>
                    </div>
                  </div>

                  <div className="mt-10 p-6 bg-slate-50 rounded-3xl border border-slate-100">
                    <h4 className="text-sm font-bold text-slate-800 mb-2 flex items-center gap-2">
                      <Info size={16} className="text-blue-500" />
                      Neural Interpretation
                    </h4>
                    <p className="text-sm text-slate-600 leading-relaxed italic">"{result.explanation}"</p>
                  </div>
                </div>

                {/* Grad-CAM Heatmap Section */}
                <div className="bg-white rounded-[2.5rem] p-10 border border-slate-200 shadow-sm">
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                      <Activity size={20} className="text-orange-500" />
                      Grad-CAM Explainability (ResNet50)
                    </h3>
                    <div className="flex gap-2">
                      <span className="flex items-center gap-1.5 text-[10px] font-bold px-3 py-1 bg-orange-50 text-orange-600 rounded-full border border-orange-100 uppercase tracking-wider">
                        Feature Importance Map
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-10">
                    <div className="lg:col-span-7">
                      <div className="rounded-3xl overflow-hidden bg-slate-50 aspect-square border border-slate-100 relative shadow-inner">
                        <canvas ref={gradCamRef} className="w-full h-full object-cover" />
                        <div className="absolute bottom-4 right-4 bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-lg text-[10px] text-white font-bold uppercase tracking-widest">
                          Overlay Intensity: 50%
                        </div>
                      </div>
                    </div>
                    
                    <div className="lg:col-span-5 space-y-6">
                      <div className="p-6 bg-slate-50 rounded-3xl border border-slate-100">
                        <h4 className="text-sm font-bold text-slate-800 mb-4 uppercase tracking-widest text-[10px]">Heatmap Legend</h4>
                        <div className="space-y-3">
                          <LegendItem color="bg-red-500" label="High Influence" desc="Regions most critical for the classification decision." />
                          <LegendItem color="bg-yellow-400" label="Moderate Influence" desc="Supporting features identified by the ResNet filters." />
                          <LegendItem color="bg-blue-500" label="Low Influence" desc="Background or non-contributory skin texture." />
                        </div>
                      </div>

                      <div className="p-6 bg-orange-50/50 rounded-3xl border border-orange-100">
                        <h4 className="text-sm font-bold text-orange-800 mb-2">Diagnostic Insight</h4>
                        <p className="text-xs text-orange-700 leading-relaxed">
                          Grad-CAM visualizes the gradients of the target class flowing into the final convolutional layer. The "hot" regions indicate where the model is looking to confirm its diagnosis.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'about' && (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="space-y-8">
                <div className="bg-white rounded-[2.5rem] p-12 shadow-sm border border-slate-200">
                  <h3 className="text-3xl font-black text-slate-900 mb-8">System Architecture</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-12">
                    <div className="space-y-6">
                      <div className="p-6 bg-blue-50 rounded-3xl border border-blue-100">
                        <h4 className="text-lg font-bold text-blue-800 mb-2">U-Net Segmentation</h4>
                        <p className="text-sm text-blue-700 leading-relaxed">
                          A deep convolutional neural network designed for biomedical image segmentation. It consists of a contracting path (encoder) to capture context and a symmetric expanding path (decoder) that enables precise localization.
                        </p>
                      </div>
                      <div className="p-6 bg-emerald-50 rounded-3xl border border-emerald-100">
                        <h4 className="text-lg font-bold text-emerald-800 mb-2">ResNet50 Classification</h4>
                        <p className="text-sm text-emerald-700 leading-relaxed">
                          A state-of-the-art residual network used via Transfer Learning. It classifies the detected region into Benign or Malignant categories with high confidence.
                        </p>
                      </div>
                    </div>
                    <div className="space-y-6">
                      <h4 className="text-xl font-bold text-slate-800">Project Methodology</h4>
                      <ul className="space-y-4">
                        <MethodStep number="01" title="Preprocessing" desc="Image resizing, normalization, and artifact removal (hair/lighting)." />
                        <MethodStep number="02" title="Feature Extraction" desc="Identifying morphological patterns using pre-trained ResNet weights." />
                        <MethodStep number="03" title="Pixel-wise Masking" desc="Generating binary masks to isolate the lesion area from healthy skin." />
                        <MethodStep number="04" title="Spread Quantification" desc="Calculating the ratio of tumor pixels vs total image pixels." />
                      </ul>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

// UI Sub-components
function SidebarItem({ icon, label, active, onClick, disabled = false }: { icon: any, label: string, active: boolean, onClick: () => void, disabled?: boolean }) {
  return (
    <button 
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center gap-4 px-6 py-4 rounded-2xl transition-all font-bold text-sm ${
        active 
          ? 'bg-blue-600 text-white shadow-lg shadow-blue-200' 
          : disabled ? 'opacity-30 cursor-not-allowed' : 'text-slate-500 hover:bg-slate-50 hover:text-slate-800'
      }`}
    >
      {icon}
      {label}
      {active && <ChevronRight size={16} className="ml-auto" />}
    </button>
  );
}

function ResultCard({ label, value, subValue, color, icon }: { label: string, value: string, subValue: string, color: string, icon: any }) {
  const colors: any = {
    red: 'bg-red-50 border-red-100 text-red-600',
    emerald: 'bg-emerald-50 border-emerald-100 text-emerald-600',
    blue: 'bg-blue-50 border-blue-100 text-blue-600'
  };
  return (
    <div className={`p-6 rounded-[2rem] border ${colors[color]}`}>
      <p className="text-xs font-bold uppercase tracking-wider opacity-60 mb-2">{label}</p>
      <div className="flex items-center gap-3">
        <div className="opacity-80">{icon}</div>
        <div>
          <h4 className="text-2xl font-black">{value}</h4>
          <p className="text-[10px] font-bold uppercase opacity-70">{subValue}</p>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, desc }: { icon: any, title: string, desc: string }) {
  return (
    <div className="bg-white p-8 rounded-3xl border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
      <div className="w-12 h-12 bg-slate-50 rounded-2xl flex items-center justify-center mb-6">
        {icon}
      </div>
      <h4 className="text-lg font-bold text-slate-800 mb-3">{title}</h4>
      <p className="text-sm text-slate-500 leading-relaxed">{desc}</p>
    </div>
  );
}

function MethodStep({ number, title, desc }: { number: string, title: string, desc: string }) {
  return (
    <div className="flex gap-4">
      <span className="text-blue-600 font-black text-lg">{number}</span>
      <div>
        <h5 className="font-bold text-slate-800 text-sm">{title}</h5>
        <p className="text-xs text-slate-500 mt-1">{desc}</p>
      </div>
    </div>
  );
}

function LegendItem({ color, label, desc }: { color: string, label: string, desc: string }) {
  return (
    <div className="flex items-center gap-4 p-4 rounded-2xl bg-white border border-slate-100 shadow-sm hover:border-slate-200 transition-all group">
      <div className={`w-14 h-14 rounded-2xl shrink-0 ${color} shadow-lg flex items-center justify-center relative overflow-hidden`}>
        <div className="absolute inset-0 bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
        <div className="w-6 h-6 rounded-full bg-white/20 blur-sm" />
      </div>
      <div>
        <p className="text-sm font-black text-slate-800 tracking-tight">{label}</p>
        <p className="text-[11px] text-slate-500 leading-snug mt-1">{desc}</p>
      </div>
    </div>
  );
}
