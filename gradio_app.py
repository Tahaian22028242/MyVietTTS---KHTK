import os
import warnings
# Suppress DeprecationWarnings as early as possible so third-party libs
# (Gradio, audioread, etc.) don't spam the terminal during import.
warnings.filterwarnings("ignore", category=DeprecationWarning)
import gradio as gr
import soundfile as sf
import tempfile
import time
import io
import sys
import contextlib
import logging


# Context manager to suppress stdout/stderr, warnings and reduce logging noise during heavy imports/model init
@contextlib.contextmanager
def _suppress_output():
    new_stdout = io.StringIO()
    new_stderr = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    old_level = logging.root.level
    try:
        sys.stdout, sys.stderr = new_stdout, new_stderr
        logging.root.setLevel(logging.ERROR)
        # Save current warnings filter list and then ignore warnings while
        # performing noisy imports/initialization. We'll restore the original
        # filters on exit so global suppression (set at startup) remains.
        _old_filters = warnings.filters[:] if hasattr(warnings, 'filters') else None
        warnings.simplefilter('ignore')
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        logging.root.setLevel(old_level)
        # Restore previous warnings filters instead of resetting all filters.
        try:
            if _old_filters is not None:
                warnings.filters[:] = _old_filters
        except Exception:
            # Fallback to reset if restoring fails for any reason
            warnings.resetwarnings()
import numpy as np
import re
from typing import Generator
import queue
import threading
import yaml
from utils.core_utils import split_text_into_chunks

# Defer importing torch and the model loader until model load time to avoid
# noisy compatibility messages appearing during initial startup. This keeps
# the web UI responsive while heavy libs load only when needed.
# Globals will be set when load_model runs.
torch = None
VieNeuTTS = None
FastVieNeuTTS = None

print("⏳ Đang khởi động VieNeu-TTS...")

# --- CONSTANTS & CONFIG ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config = yaml.safe_load(f) or {}
except Exception as e:
    raise RuntimeError(f"Không thể đọc config.yaml: {e}")

BACKBONE_CONFIGS = _config.get("backbone_configs", {})
CODEC_CONFIGS = _config.get("codec_configs", {})
VOICE_SAMPLES = _config.get("voice_samples", {})

_text_settings = _config.get("text_settings", {})
MAX_CHARS_PER_CHUNK = _text_settings.get("max_chars_per_chunk", 256)
MAX_TOTAL_CHARS_STREAMING = _text_settings.get("max_total_chars_streaming", 3000)

if not BACKBONE_CONFIGS or not CODEC_CONFIGS:
    raise ValueError("config.yaml thiếu backbone_configs hoặc codec_configs")
if not VOICE_SAMPLES:
    raise ValueError("config.yaml thiếu voice_samples")

# --- 1. MODEL CONFIGURATION ---
# Global model instance
tts = None
current_backbone = None
current_codec = None
model_loaded = False
using_lmdeploy = False

def should_use_lmdeploy(backbone_choice, device_choice):
    """Determine if we should use LMDeploy backend."""
    if "GGUF" in backbone_choice:
        return False
    
    if device_choice == "Auto":
        has_gpu = torch.cuda.is_available()
    elif device_choice == "CUDA":
        has_gpu = torch.cuda.is_available()
    else:
        has_gpu = False
    
    return has_gpu

def load_model(backbone_choice, codec_choice, device_choice, enable_triton, kv_quant, max_batch_size):
    """Load model with optimizations and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    # yield (
    #     "⏳ Đang tải model với tối ưu hóa... Lưu ý: Quá trình này sẽ tốn thời gian. Vui lòng kiên nhẫn.",
    #     gr.update(interactive=False),
    #     gr.update(interactive=False)
    # )
    
    try:
        # measure load start time for reporting
        start_time = time.time()
        # Lazy-import heavy libraries to avoid startup noise; set globals so other functions can use them
        global torch, VieNeuTTS, FastVieNeuTTS
        if torch is None or VieNeuTTS is None:
            try:
                import importlib
                with _suppress_output():
                    _torch = importlib.import_module('torch')
                    mod = importlib.import_module('vieneu_tts')
                torch = _torch
                VieNeuTTS = getattr(mod, 'VieNeuTTS')
                FastVieNeuTTS = getattr(mod, 'FastVieNeuTTS', None)
            except Exception as ie:
                yield (
                    f"❌ Thiếu dependency khi import mô-đun model: {ie}\nHãy chắc chắn đã cài torch/transformers/neucodec trong môi trường của bạn.",
                    gr.update(interactive=False),
                    gr.update(interactive=True)
                )
                return
        backbone_config = BACKBONE_CONFIGS[backbone_choice]
        codec_config = CODEC_CONFIGS[codec_choice]
        
        # Skip LMDeploy entirely (lmdeploy not installed/used on this machine)
        use_lmdeploy = False
        
        if use_lmdeploy:
            print(f"🚀 Using LMDeploy backend with optimizations")
            
            backbone_device = "cuda"
            
            if "ONNX" in codec_choice:
                codec_device = "cpu"
            else:
                codec_device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"📦 Loading optimized model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            print(f"   Triton: {'Enabled' if enable_triton else 'Disabled'}")
            print(f"   KV Quant: {kv_quant}")
            print(f"   Max Batch Size: {max_batch_size}")
            try:
                with _suppress_output():
                    tts = FastVieNeuTTS(
                        backbone_repo=backbone_config["repo"],
                        backbone_device=backbone_device,
                        codec_repo=codec_config["repo"],
                        codec_device=codec_device,
                        memory_util=0.3,
                        tp=1,
                        enable_prefix_caching=True,
                        quant_policy=kv_quant,
                        enable_triton=enable_triton,
                        max_batch_size=max_batch_size,  # ✅ Pass max_batch_size
                    )
                using_lmdeploy = True

                # Pre-cache voice references
                print("📝 Pre-caching voice references...")
                for voice_name, voice_info in VOICE_SAMPLES.items():
                    audio_path = voice_info["audio"]
                    text_path = voice_info["text"]
                    if os.path.exists(audio_path) and os.path.exists(text_path):
                        with open(text_path, "r", encoding="utf-8") as f:
                            ref_text = f.read()
                        tts.get_cached_reference(voice_name, audio_path, ref_text)
                print(f"   ✅ Cached {len(VOICE_SAMPLES)} voices")

            except ImportError:
                # yield (
                #     "⚠️ LMDeploy không được cài đặt. Vui lòng cài lmdeploy (pip install lmdeploy) để tận dụng tối đa GPU. Fallback về HF Transformers...",
                #     gr.update(interactive=False),
                #     gr.update(interactive=True)
                # )
                time.sleep(2)
                use_lmdeploy = False
                using_lmdeploy = False
        
        if not use_lmdeploy:
            print(f"📦 Using original backend")
            
            if device_choice == "Auto":
                if "GGUF" in backbone_choice:
                    backbone_device = "gpu" if torch.cuda.is_available() else "cpu"
                else:
                    backbone_device = "cuda" if torch.cuda.is_available() else "cpu"
                
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
                else:
                    codec_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                backbone_device = device_choice.lower()
                codec_device = device_choice.lower()
                
                if "ONNX" in codec_choice:
                    codec_device = "cpu"
            
            if "GGUF" in backbone_choice and backbone_device == "cuda":
                backbone_device = "gpu"
            
            print(f"📦 Loading model...")
            print(f"   Backbone: {backbone_config['repo']} on {backbone_device}")
            print(f"   Codec: {codec_config['repo']} on {codec_device}")
            
            with _suppress_output():
                tts = VieNeuTTS(
                    backbone_repo=backbone_config["repo"],
                    backbone_device=backbone_device,
                    codec_repo=codec_config["repo"],
                    codec_device=codec_device
                )
            using_lmdeploy = False
        
        current_backbone = backbone_choice
        current_codec = codec_choice
        model_loaded = True
        
        # Success message with optimization info
        backend_name = "🚀 LMDeploy (Optimized)" if using_lmdeploy else "📦 Standard"
        device_info = "cuda" if use_lmdeploy else (backbone_device if not use_lmdeploy else "N/A")
        
        streaming_support = "✅ Có" if backbone_config['supports_streaming'] else "❌ Không"
        preencoded_note = "\n⚠️ Codec này cần sử dụng pre-encoded codes (.pt files)" if codec_config['use_preencoded'] else ""
        
        opt_info = ""
        if using_lmdeploy and hasattr(tts, 'get_optimization_stats'):
            stats = tts.get_optimization_stats()
            opt_info = (
                f"\n\n🔧 Tối ưu hóa:"
                f"\n  • Triton: {'✅' if stats['triton_enabled'] else '❌'}"
                f"\n  • KV Cache Quant: {stats['kv_quant']}-bit"
                f"\n  • Max Batch Size: {max_batch_size}"
                f"\n  • Reference Cache: {stats['cached_references']} voices"
                f"\n  • Prefix Caching: ✅"
            )
        
        success_msg = (
            f"✅ Model đã tải thành công!\n\n"
            f"🔧 Backend: {backend_name}\n"
            f"🦜 Model Device: {device_info.upper()}\n"
            f"🎵 Codec Device: {codec_device.upper()}{preencoded_note}\n"
            f"🌊 Streaming: {streaming_support}{opt_info}"
        )
        # report load time to terminal for easier monitoring
        try:
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s — Backend: {backend_name}, Model device: {device_info}, Backbone: {backbone_config['repo']}, Codec: {codec_config['repo']}")
        except Exception:
            pass

        yield (
            success_msg,
            gr.update(interactive=True),
            gr.update(interactive=True)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        model_loaded = False
        using_lmdeploy = False
        
        yield (
            f"❌ Lỗi khi tải model: {str(e)}",
            gr.update(interactive=False),
            gr.update(interactive=True)
        )


# --- 2. DATA & HELPERS ---
GGUF_ALLOWED_VOICES = [
    "Nam 1 (miền Nam)",
    "Nam 2 (miền Bắc)",
    "Nữ 2 (miền Bắc)",
    "Dung (nữ miền Nam)",
]

def get_voice_options(backbone_choice: str):
    """Filter voice options: GGUF only shows the 4 allowed voices."""
    if "gguf" in backbone_choice.lower():
        return [v for v in GGUF_ALLOWED_VOICES if v in VOICE_SAMPLES]
    return list(VOICE_SAMPLES.keys())

def update_voice_dropdown(backbone_choice: str, current_voice: str):
    options = get_voice_options(backbone_choice)
    new_value = current_voice if current_voice in options else (options[0] if options else None)
    return gr.update(choices=options, value=new_value)

# --- 3. CORE LOGIC FUNCTIONS ---
def load_reference_info(voice_choice):
    if voice_choice in VOICE_SAMPLES:
        audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        text_path = VOICE_SAMPLES[voice_choice]["text"]
        try:
            if os.path.exists(text_path):
                with open(text_path, "r", encoding="utf-8") as f:
                    ref_text = f.read()
                return audio_path, ref_text
            else:
                return audio_path, "⚠️ Không tìm thấy file text mẫu."
        except Exception as e:
            return None, f"❌ Lỗi: {str(e)}"
    return None, ""

def synthesize_speech(text, voice_choice, custom_audio, custom_text, mode_tab, generation_mode, use_batch, max_batch_size=None):
    """Synthesis with optimization support and max batch size control"""
    global tts, current_backbone, current_codec, model_loaded, using_lmdeploy
    
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return
    
    if not text or text.strip() == "":
        yield None, "⚠️ Vui lòng nhập văn bản!"
        return
    
    raw_text = text.strip()
    
    codec_config = CODEC_CONFIGS[current_codec]
    use_preencoded = codec_config['use_preencoded']
    
    # Setup Reference
    if mode_tab == "custom_mode":
        if custom_audio is None or not custom_text:
            yield None, "⚠️ Thiếu Audio hoặc Text mẫu custom."
            return
        ref_audio_path = custom_audio
        ref_text_raw = custom_text
        ref_codes_path = None
    else:
        if voice_choice not in VOICE_SAMPLES:
            yield None, "⚠️ Vui lòng chọn giọng mẫu."
            return
        ref_audio_path = VOICE_SAMPLES[voice_choice]["audio"]
        ref_text_path = VOICE_SAMPLES[voice_choice]["text"]
        ref_codes_path = VOICE_SAMPLES[voice_choice]["codes"]
        
        if not os.path.exists(ref_audio_path):
            yield None, "❌ Không tìm thấy file audio mẫu."
            return
        
        with open(ref_text_path, "r", encoding="utf-8") as f:
            ref_text_raw = f.read()
    
    yield None, "📄 Đang xử lý Reference..."
    
    # Encode or get cached reference
    try:
        if use_preencoded and ref_codes_path and os.path.exists(ref_codes_path):
            ref_codes = torch.load(ref_codes_path, map_location="cpu")
        else:
            # Use cached reference if available (LMDeploy only)
            if using_lmdeploy and hasattr(tts, 'get_cached_reference') and mode_tab == "preset_mode":
                ref_codes = tts.get_cached_reference(voice_choice, ref_audio_path, ref_text_raw)
            else:
                ref_codes = tts.encode_reference(ref_audio_path)
        
        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()
    except Exception as e:
        yield None, f"❌ Lỗi xử lý reference: {e}"
        return
    
    text_chunks = split_text_into_chunks(raw_text, max_chars=MAX_CHARS_PER_CHUNK)
    total_chunks = len(text_chunks)
    
    # === STANDARD MODE ===
    if generation_mode == "Standard (Một lần)":
        backend_name = "LMDeploy" if using_lmdeploy else "Standard"
        # Allow batch processing even when not using LMDeploy if the backend supports infer_batch
        batch_info = " (Batch Mode)" if use_batch and hasattr(tts, 'infer_batch') and total_chunks > 1 else ""

        # Show batch size info (prefer explicit max_batch_size from UI, otherwise tts.max_batch_size if available)
        if use_batch and hasattr(tts, 'infer_batch'):
            resolved_batch_size = max_batch_size if max_batch_size is not None else (tts.max_batch_size if hasattr(tts, 'max_batch_size') else None)
            batch_size_info = f" [Max batch: {resolved_batch_size}]" if resolved_batch_size is not None else ""
        else:
            batch_size_info = ""
        
        yield None, f"🚀 Bắt đầu tổng hợp {backend_name}{batch_info}{batch_size_info} ({total_chunks} đoạn)..."
        
        all_audio_segments = []
        sr = 24000
        silence_pad = np.zeros(int(sr * 0.15), dtype=np.float32)
        
        start_time = time.time()
        
        try:
            # Use batch processing if enabled and the backend provides infer_batch() (works for LMDeploy and standard backends)
            if use_batch and hasattr(tts, 'infer_batch') and total_chunks > 1:
                # Determine batch size: prefer UI-provided max_batch_size, else backend-provided, else default 8
                batch_size = int(max_batch_size) if (max_batch_size is not None) else (tts.max_batch_size if hasattr(tts, 'max_batch_size') else 8)
                num_batches = (total_chunks + batch_size - 1) // batch_size

                yield None, f"⚡ Xử lý {num_batches} mini-batch(es) (max {batch_size} đoạn/batch)..."

                # Process each mini-batch sequentially to avoid overwhelming GPU memory
                for b in range(num_batches):
                    start_idx = b * batch_size
                    end_idx = min(start_idx + batch_size, total_chunks)
                    mini_chunks = text_chunks[start_idx:end_idx]

                    # Expect infer_batch to return a list of wav arrays matching mini_chunks
                    mini_wavs = tts.infer_batch(mini_chunks, ref_codes, ref_text_raw)

                    for i_rel, chunk_wav in enumerate(mini_wavs):
                        global_idx = start_idx + i_rel
                        if chunk_wav is not None and len(chunk_wav) > 0:
                            all_audio_segments.append(chunk_wav)
                            if global_idx < total_chunks - 1:
                                all_audio_segments.append(silence_pad)
            else:
                # Sequential processing
                for i, chunk in enumerate(text_chunks):
                    yield None, f"⏳ Đang xử lý đoạn {i+1}/{total_chunks}..."

                    chunk_wav = tts.infer(chunk, ref_codes, ref_text_raw)

                    if chunk_wav is not None and len(chunk_wav) > 0:
                        all_audio_segments.append(chunk_wav)
                        if i < total_chunks - 1:
                            all_audio_segments.append(silence_pad)
            
            if not all_audio_segments:
                yield None, "❌ Không sinh được audio nào."
                return
            
            yield None, "💾 Đang ghép file và lưu..."
            
            final_wav = np.concatenate(all_audio_segments)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                output_path = tmp.name
            
            process_time = time.time() - start_time
            backend_info = f""
            speed_info = f", Tốc độ: {len(final_wav)/sr/process_time:.2f}x realtime" if process_time > 0 else ""
            
            yield output_path, f"✅ Hoàn tất! (Thời gian: {process_time:.2f}s{speed_info}){backend_info}"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()
            
        except torch.cuda.OutOfMemoryError as e:
            yield None, (
                f"❌ GPU hết VRAM! Hãy thử:\n"
                f"• Giảm Max Batch Size (hiện tại: {tts.max_batch_size if hasattr(tts, 'max_batch_size') else 'N/A'})\n"
                f"• Bật KV Cache Quantization (8-bit)\n"
                f"• Giảm độ dài văn bản\n\n"
                f"Chi tiết: {str(e)}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield None, f"❌ Lỗi Standard Mode: {str(e)}"
            return
    
    # === STREAMING MODE ===
    else:
        sr = 24000
        crossfade_samples = int(sr * 0.03)
        audio_queue = queue.Queue(maxsize=100)
        PRE_BUFFER_SIZE = 3
        
        end_event = threading.Event()
        error_event = threading.Event()
        error_msg = ""
        
        def producer_thread():
            nonlocal error_msg
            try:
                previous_tail = None
                
                for i, chunk_text in enumerate(text_chunks):
                    stream_gen = tts.infer_stream(chunk_text, ref_codes, ref_text_raw)
                    
                    for part_idx, audio_part in enumerate(stream_gen):
                        if audio_part is None or len(audio_part) == 0:
                            continue
                        
                        if previous_tail is not None and len(previous_tail) > 0:
                            overlap = min(len(previous_tail), len(audio_part), crossfade_samples)
                            if overlap > 0:
                                fade_out = np.linspace(1.0, 0.0, overlap, dtype=np.float32)
                                fade_in = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
                                
                                blended = (audio_part[:overlap] * fade_in + 
                                         previous_tail[-overlap:] * fade_out)
                                
                                processed = np.concatenate([
                                    previous_tail[:-overlap] if len(previous_tail) > overlap else np.array([]),
                                    blended,
                                    audio_part[overlap:]
                                ])
                            else:
                                processed = np.concatenate([previous_tail, audio_part])
                            
                            tail_size = min(crossfade_samples, len(processed))
                            previous_tail = processed[-tail_size:].copy()
                            output_chunk = processed[:-tail_size] if len(processed) > tail_size else processed
                        else:
                            tail_size = min(crossfade_samples, len(audio_part))
                            previous_tail = audio_part[-tail_size:].copy()
                            output_chunk = audio_part[:-tail_size] if len(audio_part) > tail_size else audio_part
                        
                        if len(output_chunk) > 0:
                            audio_queue.put((sr, output_chunk))
                
                if previous_tail is not None and len(previous_tail) > 0:
                    audio_queue.put((sr, previous_tail))
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = str(e)
                error_event.set()
            finally:
                end_event.set()
                audio_queue.put(None)
        
        threading.Thread(target=producer_thread, daemon=True).start()
        
        yield (sr, np.zeros(int(sr * 0.05))), "📄 Đang buffering..."
        
        pre_buffer = []
        while len(pre_buffer) < PRE_BUFFER_SIZE:
            try:
                item = audio_queue.get(timeout=5.0)
                if item is None:
                    break
                pre_buffer.append(item)
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    return
                break
        
        full_audio_buffer = []
        backend_info = "🚀 LMDeploy" if using_lmdeploy else "📦 Standard"
        for sr, audio_data in pre_buffer:
            full_audio_buffer.append(audio_data)
            yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
        
        while True:
            try:
                item = audio_queue.get(timeout=0.05)
                if item is None:
                    break
                sr, audio_data = item
                full_audio_buffer.append(audio_data)
                yield (sr, audio_data), f"🔊 Đang phát ({backend_info})..."
            except queue.Empty:
                if error_event.is_set():
                    yield None, f"❌ Lỗi: {error_msg}"
                    break
                if end_event.is_set() and audio_queue.empty():
                    break
                continue
        
        if full_audio_buffer:
            final_wav = np.concatenate(full_audio_buffer)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, final_wav, sr)
                yield tmp.name, f"✅ Hoàn tất Streaming! ({backend_info})"
            
            # Cleanup memory
            if using_lmdeploy and hasattr(tts, 'cleanup_memory'):
                tts.cleanup_memory()


# --- 4. UI SETUP ---
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Plus Jakarta Sans'), 'ui-sans-serif', 'system-ui'],
    radius_size="lg",
).set(
    # Vibrant gradient buttons
    button_primary_background_fill="linear-gradient(135deg, #A78BFA 0%, #EC4899 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #8B5CF6 0%, #DB2777 100%)",
    button_primary_text_color="#FFFFFF",
    button_secondary_background_fill="linear-gradient(135deg, #F3F4F6 0%, #E5E7EB 100%)",
    button_secondary_background_fill_hover="linear-gradient(135deg, #E5E7EB 0%, #D1D5DB 100%)",
    button_secondary_text_color="#374151",
    # Clean, bright backgrounds
    body_background_fill="#FAFBFC",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F9FAFB",
    # Subtle borders with good contrast
    border_color_primary="#E2E8F0",
    # Accent colors
    color_accent="#8B5CF6",
    color_accent_soft="#EDE9FE",
)

css = """
/* Full viewport layout */
html, body, .gradio-app, .gradio-container {
    height: 100%;
    min-height: 100vh;
}
.gradio-container > .main {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}
.gradio-container > .main > .container {
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}

/* Modern container with subtle shadow */
.container { 
    max-width: 1400px; 
    margin: auto; 
    padding: 32px 24px;
    background: transparent;
}

/* Eye-catching header with gradient background */
.header-box {
    text-align: center;
    margin-bottom: 40px;
    padding: 48px 32px;
    background: linear-gradient(135deg, #EDE9FE 0%, #FBCFE8 50%, #FEF3C7 100%);
    border-radius: 24px;
    box-shadow: 0 10px 40px rgba(139, 92, 246, 0.15);
    border: 2px solid rgba(255, 255, 255, 0.8);
    position: relative;
    overflow: hidden;
}

.header-box::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 0.8; }
}

.header-title {
    font-size: 3rem;
    font-weight: 900;
    margin-bottom: 16px;
    position: relative;
    z-index: 1;
    letter-spacing: -0.02em;
}

.gradient-text {
    background: linear-gradient(135deg, #8B5CF6 0%, #EC4899 50%, #F59E0B 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 6s ease infinite;
    background-size: 200% 200%;
}

@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.header-icon {
    display: inline-block;
    font-size: 3.5rem;
    background: linear-gradient(135deg, #A78BFA, #EC4899, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 4px 8px rgba(139, 92, 246, 0.3));
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

/* Status box with modern styling */
.status-box {
    font-weight: 600;
    text-align: center;
    border: none;
    background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
    white-space: pre-wrap;
    overflow-wrap: break-word;
    height: auto;
    width: 100%;
    box-sizing: border-box;
    padding: 16px;
    border-radius: 12px;
    color: #1F2937;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Model card with modern badges */
.model-card-content {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
    gap: 12px;
    font-size: 0.95rem;
    text-align: center;
    position: relative;
    z-index: 1;
}

.model-card-item {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(139, 92, 246, 0.2);
    color: #374151;
    font-weight: 500;
    transition: all 0.3s ease;
}

.model-card-item:hover {
    background: rgba(255, 255, 255, 0.9);
    border-color: rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
}

.model-card-item strong {
    color: #8B5CF6;
    font-weight: 700;
}

.model-card-link {
    color: #EC4899;
    text-decoration: none;
    font-weight: 600;
    transition: all 0.2s ease;
    position: relative;
}

.model-card-link::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 0;
    width: 0;
    height: 2px;
    background: linear-gradient(90deg, #EC4899, #F59E0B);
    transition: width 0.3s ease;
}

.model-card-link:hover {
    color: #DB2777;
}

.model-card-link:hover::after {
    width: 100%;
}

/* Enhanced input/output sections */
.gradio-container .gr-box {
    border-radius: 16px;
    border: 2px solid #E5E7EB;
    transition: all 0.3s ease;
}

.gradio-container .gr-box:hover {
    border-color: #A78BFA;
    box-shadow: 0 4px 20px rgba(167, 139, 250, 0.1);
}

/* Modern button styling */
.gradio-container button {
    border-radius: 12px;
    font-weight: 600;
    text-transform: none;
    transition: all 0.3s ease;
    box-shadow: 0 2px 8px rgba(139, 92, 246, 0.15);
}

.gradio-container button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.25);
}

/* Input fields */
.gradio-container input, .gradio-container textarea {
    border-radius: 12px;
    border: 2px solid #E5E7EB;
    transition: all 0.3s ease;
}

.gradio-container input:focus, .gradio-container textarea:focus {
    border-color: #A78BFA;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.1);
}

/* Dropdown styling */
.gradio-container .gr-dropdown {
    border-radius: 12px;
}

/* Tab styling */
.gradio-container .gr-button-secondary {
    background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
    border: 2px solid #E5E7EB;
}

.gradio-container .gr-button-secondary.selected {
    background: linear-gradient(135deg, #A78BFA 0%, #EC4899 100%);
    color: white;
    border-color: transparent;
}

/* Audio player styling */
.gradio-container audio {
    border-radius: 12px;
}

/* Success/Error states */
.success-state {
    background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
    border: 2px solid #10B981;
    color: #065F46;
}

.error-state {
    background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
    border: 2px solid #EF4444;
    color: #991B1B;
}

.warning-state {
    background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
    border: 2px solid #F59E0B;
    color: #92400E;
}

/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #F3F4F6;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #A78BFA, #EC4899);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #8B5CF6, #DB2777);
}

/* Loading animation */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading {
    background: linear-gradient(90deg, #F3F4F6 25%, #E5E7EB 50%, #F3F4F6 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
}
"""

EXAMPLES_LIST = [
    ["Về miền Tây không chỉ để ngắm nhìn sông nước hữu tình, mà còn để cảm nhận tấm chân tình của người dân nơi đây.", "Nam 1 (miền Nam)"],
    ["Hà Nội những ngày vào thu mang một vẻ đẹp trầm mặc và cổ kính đến lạ thường.", "Nam 2 (miền Bắc)"],
]

with gr.Blocks(theme=theme, css=css, title="MyVie-TTS") as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
<div class="header-box">
    <h1 class="header-title">
        <span class="header-icon">🦜</span>
        <span class="gradient-text">MyVie-TTS App</span>
    </h1>
    <div class="model-card-content">
        <div class="model-card-item">
            <strong>📦 Models:</strong>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS" target="_blank" class="model-card-link">VieNeu-TTS</a>
            <span style="color: #D1D5DB;">•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-q4-gguf" target="_blank" class="model-card-link">Q4-GGUF</a>
            <span style="color: #D1D5DB;">•</span>
            <a href="https://huggingface.co/pnnbao-ump/VieNeu-TTS-q8-gguf" target="_blank" class="model-card-link">Q8-GGUF</a>
        </div>
        <div class="model-card-item">
            <strong>🔗 Repository:</strong>
            <a href="https://github.com/pnnbao97/VieNeu-TTS" target="_blank" class="model-card-link">GitHub</a>
        </div>
        <div class="model-card-item">
            <strong>👤 Tác giả:</strong>
            <span style="color: #6B7280; font-weight: 600;">Phạm Nguyễn Ngọc Bảo</span>
        </div>
    </div>
</div>
        """, visible=False)
        
        # --- CONFIGURATION ---
        with gr.Group(visible=False) as config_section:
            with gr.Row():
                backbone_select = gr.Dropdown(list(BACKBONE_CONFIGS.keys()), value="VieNeu-TTS (GPU)", label="🦜 Backbone")
                codec_select = gr.Dropdown(list(CODEC_CONFIGS.keys()), value="NeuCodec (Standard)", label="🎵 Codec")
                device_choice = gr.Radio(["Auto", "CPU", "CUDA"], value="CUDA", label="🖥️ Device")
            
            with gr.Row():
                enable_triton = gr.Checkbox(value=True, label="⚡ Enable Triton Compilation")
                kv_quant = gr.Radio([0, 8], value=8, label="🔧 KV Cache Quantization", info="8=int8 (save VRAM), 0=disabled")
                max_batch_size = gr.Slider(
                    minimum=1, 
                    maximum=16, 
                    value=8, 
                    step=1, 
                    label="📊 Max Batch Size",
                    info="Giảm nếu gặp lỗi OOM. 4-6 cho GPU 8GB, 8-12 cho GPU 16GB+"
                )
            
            gr.Markdown(
                "⚠️ **Lưu ý:** Nếu máy bạn chỉ có CPU vui lòng chọn phiên bản GGUF (Q4/Q8) để có tốc độ nhanh nhất.\n\n"
                "💡 **Max Batch Size:** Số lượng đoạn văn bản được xử lý cùng lúc. "
                "Giá trị cao = nhanh hơn nhưng tốn VRAM hơn. Giảm xuống nếu gặp lỗi \"Out of Memory\"."
            )

            btn_load = gr.Button("🔄 Tải Model", variant="primary")
            model_status = gr.Markdown("⏳ Chưa tải model.")
        
        with gr.Row(elem_classes="container"):
            # --- INPUT ---
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label=f"Văn bản đầu vào",
                    lines=4,
                    value="Nhập văn bản tiếng Việt bạn muốn chuyển đổi thành giọng nói tại đây.",
                )
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("Mặc định", id="preset_mode"):
                        initial_voices = get_voice_options("VieNeu-TTS (GPU)")
                        default_voice = initial_voices[0] if initial_voices else None
                        voice_select = gr.Dropdown(initial_voices, value=default_voice, label="Giọng mẫu")
                    
                    with gr.TabItem("Tùy chỉnh", id="custom_mode"):
                        custom_audio = gr.Audio(label="Tệp ghi âm mẫu (.wav)", type="filepath")
                        custom_text = gr.Textbox(label="Lời thoại mẫu")
                
                generation_mode = gr.Radio(
                    ["Standard (Một lần)"],
                    value="Standard (Một lần)",
                    label="Chế độ sinh",
                    visible=False
                )
                use_batch = gr.Checkbox(
                    value=True, 
                    label="⚡ Batch Processing",
                    info="Xử lý nhiều đoạn cùng lúc (nếu backend hỗ trợ infer_batch)",
                    visible=False
                )
                
                current_mode = gr.Textbox(visible=False, value="preset_mode")
                
                btn_generate = gr.Button("Tạo sinh", variant="primary", size="lg", interactive=False)
            
            # --- OUTPUT ---
            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Kết quả",
                    type="filepath",
                    autoplay=True,
                    show_download_button=True
                )
                status_output = gr.Textbox(label="Trạng thái", elem_classes="status-box", lines=6)
        
        # --- EVENT HANDLERS ---
        def update_info(backbone):
            return f"Streaming: {'✅' if BACKBONE_CONFIGS[backbone]['supports_streaming'] else '❌'}"
        
        backbone_select.change(update_info, backbone_select, model_status)
        backbone_select.change(update_voice_dropdown, [backbone_select, voice_select], voice_select)
        
        tabs.children[0].select(lambda: "preset_mode", outputs=current_mode)
        tabs.children[1].select(lambda: "custom_mode", outputs=current_mode)
        
        # Auto-load default model on UI load (hidden load button kept for compatibility)
        demo.load(
            fn=load_model,
            inputs=[backbone_select, codec_select, device_choice, enable_triton, kv_quant, max_batch_size],
            outputs=[model_status, btn_generate, btn_load]
        )
        
        btn_generate.click(
            fn=synthesize_speech,
            inputs=[text_input, voice_select, custom_audio, custom_text, current_mode, generation_mode, use_batch, max_batch_size],
            outputs=[audio_output, status_output]
        )

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)
