#!/usr/bin/env python3
"""
Modular Voice Assistant with Legal Database Management

This modular voice assistant includes:
- Wake word training and detection
- Speech recognition and synthesis
- Legal database management
- Natural conversation with personality
- Production-ready error handling

Requirements:
pip install openai openai-whisper pyttsx3 pyaudio pygame numpy

Author: AI Assistant
Version: 2.0
"""

import asyncio
import sqlite3
import os
import threading
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Audio libraries
import pyaudio
import wave
import pyttsx3

# AI libraries
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION MODULE
# ============================================================================

@dataclass
class AssistantConfig:
    """Configuration settings for the voice assistant."""
    wake_word: str = "rasi"
    whisper_model: str = "base"
    openai_tts_voice: str = "nova"
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    record_duration: int = 5
    wake_word_record_duration: float = 1.5
    activation_cooldown: float = 2.0
    max_conversation_history: int = 10
    database_path: str = "legal_assistant.db"
    log_level: str = "INFO"

# ============================================================================
# AUDIO PROCESSING MODULE
# ============================================================================

class AudioProcessor:
    """Handles all audio recording and processing operations."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.format = pyaudio.paInt16
        
    def record_audio(self, duration: float, show_listening_message: bool = True) -> bytes:
        """Record audio from microphone."""
        if show_listening_message:
            print("🎤 Recording...")
        
        audio_interface = pyaudio.PyAudio()
        frames = []
        
        try:
            stream = audio_interface.open(
                format=self.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            
            frames_to_record = int(self.config.sample_rate / self.config.chunk_size * duration)
            for _ in range(frames_to_record):
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
            return b""
        finally:
            audio_interface.terminate()
        
        if show_listening_message:
            print("🔇 Recording stopped")
        
        return b''.join(frames)
    
    def save_wav_file(self, filename: str, audio_data: bytes) -> bool:
        """Save audio data as WAV file."""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.config.sample_rate)
                wav_file.writeframes(audio_data)
            return True
        except Exception as e:
            logger.error(f"Error saving WAV file: {e}")
            return False

# ============================================================================
# WAKE WORD MODULE
# ============================================================================

class WakeWordTrainer:
    """Handles wake word training and sample collection."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.audio_processor = AudioProcessor(config)
        self.samples_dir = f"wake_word_samples_{config.wake_word}"
        os.makedirs(self.samples_dir, exist_ok=True)
    
    def collect_samples(self, num_samples: int = 10) -> bool:
        """Collect wake word training samples."""
        print(f"🎓 Wake Word Training for '{self.config.wake_word}'")
        print(f"📝 Collecting {num_samples} samples")
        print("💡 Speak clearly and naturally")
        print("=" * 50)
        
        successful_samples = 0
        for i in range(num_samples):
            try:
                input(f"\nPress Enter for sample {i+1}/{num_samples}...")
                print(f"🎤 Say '{self.config.wake_word}' now!")
                
                audio_data = self.audio_processor.record_audio(2.0, show_listening_message=False)
                if audio_data:
                    sample_file = os.path.join(self.samples_dir, f"sample_{i+1:02d}.wav")
                    if self.audio_processor.save_wav_file(sample_file, audio_data):
                        print(f"✅ Sample {i+1} saved")
                        successful_samples += 1
                    else:
                        print(f"❌ Failed to save sample {i+1}")
                else:
                    print(f"❌ Failed to record sample {i+1}")
                    
            except KeyboardInterrupt:
                print("\nTraining interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error during sample collection: {e}")
        
        print(f"\n🎉 Training complete! {successful_samples}/{num_samples} samples collected")
        return successful_samples >= 5  # Minimum required samples

class WakeWordDetector:
    """Advanced wake word detection using audio features."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.samples_dir = f"wake_word_samples_{config.wake_word}"
        self.reference_features = []
        self.threshold = 0.65
        self._load_reference_samples()
    
    def _load_reference_samples(self):
        """Load and process reference samples."""
        if not os.path.exists(self.samples_dir):
            logger.warning(f"No training samples found: {self.samples_dir}")
            return
        
        sample_files = [f for f in os.listdir(self.samples_dir) if f.endswith('.wav')]
        if len(sample_files) < 5:
            logger.warning(f"Insufficient samples: {len(sample_files)}")
            return
        
        logger.info(f"Loading {len(sample_files)} reference samples...")
        
        for sample_file in sample_files:
            filepath = os.path.join(self.samples_dir, sample_file)
            try:
                with wave.open(filepath, 'rb') as wav_file:
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    features = self._extract_features(audio_data)
                    if features:
                        self.reference_features.append(features)
            except Exception as e:
                logger.error(f"Error loading {sample_file}: {e}")
        
        logger.info(f"Loaded {len(self.reference_features)} valid samples")
    
    def _extract_features(self, audio_data: bytes) -> Optional[Dict]:
        """Extract audio features for comparison."""
        try:
            import numpy as np
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            if len(audio_array) == 0:
                return None
            
            # Normalize audio
            audio_float = audio_array.astype(np.float32) / 32768.0
            audio_float = audio_float - np.mean(audio_float)
            
            # Apply simple noise gate
            energy_threshold = np.percentile(np.abs(audio_float), 10)
            audio_float = np.where(np.abs(audio_float) > energy_threshold, audio_float, 0)
            
            # Extract features
            energy = np.mean(audio_float ** 2)
            zero_crossings = self._calculate_zero_crossings(audio_float)
            spectral_centroid = self._calculate_spectral_centroid(audio_float)
            
            return {
                'energy': float(energy),
                'zero_crossings': float(zero_crossings),
                'spectral_centroid': float(spectral_centroid),
                'rms': float(np.sqrt(np.mean(audio_float ** 2)))
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def _calculate_zero_crossings(self, audio_signal) -> float:
        """Calculate zero crossing rate."""
        try:
            import numpy as np
            if len(audio_signal) < 2:
                return 0.0
            
            signs = np.sign(audio_signal)
            signs[signs == 0] = 1  # Treat zeros as positive
            zero_crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(audio_signal))
            return min(1.0, zero_crossings)  # Normalize
        except:
            return 0.0

    def _calculate_spectral_centroid(self, audio_signal) -> float:
        """Calculate spectral centroid with error handling."""
        try:
            import numpy as np
            
            if len(audio_signal) < 64:  # Minimum for meaningful FFT
                return 0.0
            
            # Window the signal to reduce spectral leakage
            window = np.hanning(len(audio_signal))
            windowed_signal = audio_signal * window
            
            # FFT
            fft = np.abs(np.fft.fft(windowed_signal))
            freqs = np.fft.fftfreq(len(fft), 1/self.config.sample_rate)
            
            # Only positive frequencies
            fft = fft[:len(fft)//2]
            freqs = freqs[:len(freqs)//2]
            
            # Spectral centroid
            total_energy = np.sum(fft)
            if total_energy > 1e-10:
                centroid = np.sum(freqs * fft) / total_energy
                return float(min(self.config.sample_rate / 2, max(0, centroid)))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_zero_crossings(self, audio_signal) -> float:
        """Calculate zero crossing rate."""
        try:
            import numpy as np
            if len(audio_signal) < 2:
                return 0.0
            
            signs = np.sign(audio_signal)
            signs[signs == 0] = 1
            zero_crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(audio_signal))
            return min(1.0, zero_crossings)
        except:
            return 0.0
    
    def _calculate_spectral_centroid(self, audio_signal) -> float:
        """Calculate spectral centroid."""
        try:
            import numpy as np
            
            if len(audio_signal) < 64:
                return 0.0
            
            window = np.hanning(len(audio_signal))
            windowed_signal = audio_signal * window
            
            fft = np.abs(np.fft.fft(windowed_signal))
            freqs = np.fft.fftfreq(len(fft), 1/self.config.sample_rate)
            
            fft = fft[:len(fft)//2]
            freqs = freqs[:len(freqs)//2]
            
            total_energy = np.sum(fft)
            if total_energy > 1e-10:
                centroid = np.sum(freqs * fft) / total_energy
                return float(min(self.config.sample_rate / 2, max(0, centroid)))
            else:
                return 0.0
        except:
            return 0.0
    
    def detect(self, audio_data: bytes) -> Tuple[bool, float]:
        """Detect wake word in audio data."""
        if not self.reference_features or not audio_data:
            return False, 0.0
        
        try:
            input_features = self._extract_features(audio_data)
            if not input_features:
                return False, 0.0
            
            similarities = []
            for ref_features in self.reference_features:
                similarity = self._calculate_similarity(input_features, ref_features)
                similarities.append(similarity)
            
            best_similarity = max(similarities) if similarities else 0.0
            detected = best_similarity > self.threshold
            
            return detected, best_similarity
            
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False, 0.0
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between feature sets."""
        try:
            # Energy similarity
            energy_sim = 1 - abs(features1['energy'] - features2['energy']) / max(features1['energy'], features2['energy'], 1e-6)
            
            # Zero crossings similarity
            zc_sim = 1 - abs(features1['zero_crossings'] - features2['zero_crossings']) / max(features1['zero_crossings'], features2['zero_crossings'], 1e-6)
            
            # Spectral centroid similarity
            sc_sim = 1 - abs(features1['spectral_centroid'] - features2['spectral_centroid']) / max(features1['spectral_centroid'], features2['spectral_centroid'], 1e-6)
            
            # RMS similarity
            rms_sim = 1 - abs(features1['rms'] - features2['rms']) / max(features1['rms'], features2['rms'], 1e-6)
            
            # Weighted combination
            weights = [0.3, 0.2, 0.3, 0.2]  # energy, zc, sc, rms
            similarities = [energy_sim, zc_sim, sc_sim, rms_sim]
            
            weighted_similarity = sum(w * s for w, s in zip(weights, similarities))
            return max(0.0, min(1.0, weighted_similarity))
            
        except Exception as e:
            logger.error(f"Similarity calculation error: {e}")
            return 0.0

# ============================================================================
# SPEECH PROCESSING MODULE
# ============================================================================

class SpeechProcessor:
    """Handles speech recognition and synthesis."""
    
    def __init__(self, config: AssistantConfig, openai_client):
        self.config = config
        self.client = openai_client
        self.whisper_model = None
        self._load_whisper_model()
    
    def _load_whisper_model(self):
        """Load Whisper model for speech recognition."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Whisper not available. Install with: pip install openai-whisper")
        
        try:
            print("🔧 Loading Whisper model...")
            self.whisper_model = whisper.load_model(self.config.whisper_model)
            print("✅ Whisper model loaded")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """Convert speech to text using Whisper."""
        if not audio_data:
            return ""
        
        try:
            print("🔄 Processing speech...")
            
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            result = self.whisper_model.transcribe(
                audio_float,
                fp16=False,
                language='en'
            )
            
            text = result["text"].strip()
            print(f"🗣️  You said: '{text}'")
            return text
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return ""
    
    def text_to_speech(self, text: str) -> Optional[threading.Thread]:
        """Convert text to speech using OpenAI TTS."""
        if not text or len(text.strip()) == 0:
            return None
        
        print(f"🤖 Assistant: {text}")
        
        def speak():
            try:
                # Try OpenAI TTS first
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=self.config.openai_tts_voice,
                    input=text,
                    speed=1.1
                )
                
                import pygame
                import io
                
                pygame.mixer.init()
                
                audio_stream = io.BytesIO()
                for chunk in response.iter_bytes():
                    audio_stream.write(chunk)
                audio_stream.seek(0)
                
                pygame.mixer.music.load(audio_stream)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                pygame.mixer.quit()
                
            except Exception as openai_error:
                logger.warning(f"OpenAI TTS failed: {openai_error}, using fallback")
                self._fallback_tts(text)
        
        tts_thread = threading.Thread(target=speak, daemon=True)
        tts_thread.start()
        return tts_thread
    
    def _fallback_tts(self, text: str):
        """Fallback TTS using system speech synthesis."""
        try:
            import subprocess
            if os.name == 'nt':  # Windows
                ps_command = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female)
                $synth.Rate = 2
                $synth.Volume = 90
                $synth.Speak("{text.replace('"', "'").replace('$', 'dollars')}")
                '''
                subprocess.run(['powershell', '-Command', ps_command], 
                             capture_output=True, timeout=20)
            else:
                subprocess.run(['espeak', '-s', '160', '-p', '50', text], 
                             capture_output=True, timeout=15)
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")

# ============================================================================
# DATABASE MODULE
# ============================================================================

class LegalDatabase:
    """Handles all legal database operations."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.db_path = config.database_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with legal tables and sample data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        self._create_tables(cursor)
        self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        logger.info(f"Legal database initialized at {self.db_path}")
    
    def _create_tables(self, cursor):
        """Create all necessary database tables."""
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                phone TEXT,
                role TEXT DEFAULT 'client',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Attorneys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attorneys (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                license_number TEXT UNIQUE,
                specialization TEXT,
                email TEXT,
                phone TEXT,
                law_firm TEXT,
                years_experience INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Cases table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY,
                case_number TEXT UNIQUE NOT NULL,
                client_id INTEGER,
                attorney_id INTEGER,
                case_type TEXT NOT NULL,
                case_status TEXT DEFAULT 'active',
                case_title TEXT NOT NULL,
                description TEXT,
                filing_date DATE,
                court_name TEXT,
                judge_name TEXT,
                estimated_value DECIMAL(12,2),
                priority_level TEXT DEFAULT 'medium',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES users (id),
                FOREIGN KEY (attorney_id) REFERENCES attorneys (id)
            )
        ''')
        
        # Hearings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hearings (
                id INTEGER PRIMARY KEY,
                case_id INTEGER NOT NULL,
                hearing_date DATETIME NOT NULL,
                hearing_type TEXT NOT NULL,
                court_room TEXT,
                judge_name TEXT,
                hearing_status TEXT DEFAULT 'scheduled',
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (id)
            )
        ''')
        
        # Tasks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY,
                case_id INTEGER,
                user_id INTEGER,
                title TEXT NOT NULL,
                description TEXT,
                due_date DATETIME,
                priority TEXT DEFAULT 'medium',
                completed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                case_id INTEGER NOT NULL,
                document_type TEXT NOT NULL,
                document_name TEXT NOT NULL,
                file_path TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                document_status TEXT DEFAULT 'active',
                description TEXT,
                FOREIGN KEY (case_id) REFERENCES cases (id)
            )
        ''')
        # Notes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                case_id INTEGER,
                user_id INTEGER,
                title TEXT NOT NULL,
                content TEXT,
                note_type TEXT DEFAULT 'general',
                confidential BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases (id),
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
    
    def _insert_sample_data(self, cursor):
        """Insert sample legal data."""
        cursor.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            # Sample users
            cursor.execute("INSERT INTO users (name, email, phone, role) VALUES (?, ?, ?, ?)", 
                         ("John Doe", "john@example.com", "555-0101", "client"))
            cursor.execute("INSERT INTO users (name, email, phone, role) VALUES (?, ?, ?, ?)", 
                         ("Jane Smith", "jane@example.com", "555-0102", "client"))
            
            # Sample attorneys
            cursor.execute("""INSERT INTO attorneys 
                           (name, license_number, specialization, email, phone, law_firm, years_experience) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                         ("Sarah Johnson", "ATT-2024-001", "Criminal Defense", "sarah@lawfirm.com", "555-0201", "Johnson & Associates", 15))
            cursor.execute("""INSERT INTO attorneys 
                           (name, license_number, specialization, email, phone, law_firm, years_experience) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                         ("Michael Brown", "ATT-2024-002", "Corporate Law", "michael@lawfirm.com", "555-0202", "Brown Legal Group", 12))
            
            # Sample cases
            cursor.execute("""INSERT INTO cases 
                           (case_number, client_id, attorney_id, case_type, case_title, description, 
                            filing_date, court_name, judge_name, estimated_value, priority_level) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                         ("2024-CV-001", 1, 1, "Civil", "Contract Dispute", "Commercial contract disagreement", 
                          "2024-01-15", "District Court", "Judge Wilson", 50000.00, "high"))
            cursor.execute("""INSERT INTO cases 
                           (case_number, client_id, attorney_id, case_type, case_title, description, 
                            filing_date, court_name, judge_name, estimated_value, priority_level) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                         ("2024-CR-002", 2, 1, "Criminal", "DUI Defense", "DUI charges defense", 
                          "2024-02-01", "Municipal Court", "Judge Davis", 5000.00, "medium"))
            
            # Sample hearings
            cursor.execute("""INSERT INTO hearings 
                           (case_id, hearing_date, hearing_type, court_room, judge_name, hearing_status, notes) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                         (1, "2024-03-15 10:00:00", "Pre-trial", "Courtroom A", "Judge Wilson", "scheduled", "Initial hearing"))
            cursor.execute("""INSERT INTO hearings 
                           (case_id, hearing_date, hearing_type, court_room, judge_name, hearing_status, notes) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""", 
                         (2, "2024-02-28 14:00:00", "Arraignment", "Courtroom B", "Judge Davis", "completed", "Plea entered"))
    
    def execute_query(self, query: str, params: tuple = ()) -> Dict[str, Any]:
        """Execute database query safely."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            if query.strip().upper().startswith("SELECT"):
                data = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result = [dict(zip(columns, row)) for row in data]
            else:
                conn.commit()
                result = {"affected_rows": cursor.rowcount}
            
            conn.close()
            return {"success": True, "data": result}
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return {"success": False, "error": str(e)}

# ============================================================================
# CONVERSATION MODULE
# ============================================================================

class ConversationManager:
    """Manages conversation flow and AI responses."""
    
    def __init__(self, config: AssistantConfig, openai_client, database: LegalDatabase):
        self.config = config
        self.client = openai_client
        self.database = database
        self.conversation_history = []
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Check for database operations
        db_response = await self._handle_database_request(user_input)
        
        # System message for AI personality
        system_message = """You are Rasi, a witty and helpful legal assistant AI with a warm personality. 
        You have access to a comprehensive legal database and love helping with legal matters, but you're also great at general conversation.

        Your personality:
        - Naturally conversational and humorous
        - Professional but not robotic
        - Helpful and knowledgeable about legal matters
        - Honest when you don't know something
        - Use casual language and contractions
        - Add personality to your responses

        Be conversational, helpful, and don't sound like a robot. Add humor when appropriate."""
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        if db_response:
            context_message = f"Database result: {db_response}\n\nIncorporate this naturally into your response."
            self.conversation_history.append({"role": "system", "content": context_message})
        
        try:
            messages = [{"role": "system", "content": system_message}] + self.conversation_history[-self.config.max_conversation_history:]
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.8,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            assistant_response = response.choices[0].message.content.strip()
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"AI response error: {e}")
            
            if db_response:
                return f"I found this info: {db_response}"
            
            return "Sorry, I'm having a brain fog moment. Could you try that again?"
    
    async def _handle_database_request(self, user_input: str) -> Optional[str]:
        """Handle database-specific requests."""
        input_lower = user_input.lower()
        
        # Cases
        if any(phrase in input_lower for phrase in ["show cases", "list cases", "my cases"]):
            result = self.database.execute_query("""
                SELECT c.case_number, c.case_title, c.case_status, a.name as attorney_name, 
                       c.judge_name, c.court_name, c.priority_level
                FROM cases c 
                LEFT JOIN attorneys a ON c.attorney_id = a.id 
                ORDER BY c.created_at DESC LIMIT 5
            """)
            
            if result["success"] and result["data"]:
                cases = result["data"]
                response = f"Here are your {len(cases)} most recent cases:\n\n"
                for case in cases:
                    status_emoji = "🟢" if case["case_status"] == "active" else "🔴"
                    priority_emoji = "🔥" if case["priority_level"] == "high" else "⚡"
                    response += f"{status_emoji}{priority_emoji} {case['case_number']}: {case['case_title']}\n"
                    response += f"   Attorney: {case['attorney_name'] or 'Unassigned'}\n"
                    response += f"   Judge: {case['judge_name'] or 'TBD'}\n\n"
                return response.strip()
            else:
                return "No cases found in your legal database."
        
        # Hearings
        elif any(phrase in input_lower for phrase in ["show hearings", "next hearing", "court date"]):
            result = self.database.execute_query("""
                SELECT h.hearing_date, h.hearing_type, h.court_room, h.judge_name,
                       c.case_number, c.case_title
                FROM hearings h
                JOIN cases c ON h.case_id = c.id
                WHERE h.hearing_date >= datetime('now')
                ORDER BY h.hearing_date ASC LIMIT 3
            """)
            
            if result["success"] and result["data"]:
                hearings = result["data"]
                response = f"Upcoming hearings ({len(hearings)}):\n\n"
                for hearing in hearings:
                    response += f"📅 {hearing['hearing_date']}\n"
                    response += f"   Case: {hearing['case_number']} - {hearing['case_title']}\n"
                    response += f"   Type: {hearing['hearing_type']}\n"
                    response += f"   Judge: {hearing['judge_name']}, Room: {hearing['court_room']}\n\n"
                return response.strip()
            else:
                return "No upcoming hearings scheduled."
        
        # Add case
        elif any(phrase in input_lower for phrase in ["add case", "create case", "new case"]):
            for phrase in ["add case", "create case", "new case"]:
                if phrase in input_lower:
                    title = user_input.lower().replace(phrase, "").strip()
                    break
            
            if title:
                import datetime
                case_number = f"2024-GEN-{datetime.datetime.now().strftime('%m%d%H%M')}"
                result = self.database.execute_query("""
                    INSERT INTO cases (case_number, client_id, case_type, case_title, case_status) 
                    VALUES (?, ?, ?, ?, ?)
                """, (case_number, 1, "General", title, "active"))
                
                if result["success"]:
                    return f"New case created: {case_number} - '{title}'"
                else:
                    return f"Failed to create case: {result.get('error', 'unknown error')}"
        
        # Add task
        elif any(phrase in input_lower for phrase in ["add task", "create task", "new task"]):
            for phrase in ["add task", "create task", "new task"]:
                if phrase in input_lower:
                    title = user_input.lower().replace(phrase, "").strip()
                    break
            
            if title:
                result = self.database.execute_query("""
                    INSERT INTO tasks (user_id, title, priority) 
                    VALUES (?, ?, ?)
                """, (1, title, "medium"))
                
                if result["success"]:
                    return f"Task added: '{title}'"
                else:
                    return f"Failed to add task: {result.get('error', 'unknown error')}"
        
        # Show tasks
        elif any(phrase in input_lower for phrase in ["show tasks", "list tasks", "my tasks"]):
            result = self.database.execute_query("""
                SELECT title, priority, completed, due_date
                FROM tasks 
                WHERE user_id = ? 
                ORDER BY completed ASC, priority DESC, created_at DESC
                LIMIT 10
            """, (1,))
            
            if result["success"] and result["data"]:
                tasks = result["data"]
                pending = [t for t in tasks if not t["completed"]]
                
                if pending:
                    response = f"You have {len(pending)} pending tasks:\n\n"
                    for task in pending[:5]:
                        priority_emoji = "🔥" if task["priority"] == "high" else "⚡" if task["priority"] == "medium" else "📌"
                        response += f"{priority_emoji} {task['title']}\n"
                        if task["due_date"]:
                            response += f"   Due: {task['due_date']}\n"
                    return response.strip()
                else:
                    return "All caught up! No pending tasks."
            else:
                return "No tasks found."
        
        return None

# ============================================================================
# MAIN VOICE ASSISTANT CLASS
# ============================================================================

class VoiceAssistant:
    """Main voice assistant orchestrating all modules."""
    
    def __init__(self, openai_api_key: str, config: AssistantConfig = None):
        self.config = config or AssistantConfig()
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Initialize modules
        self.audio_processor = AudioProcessor(self.config)
        self.speech_processor = SpeechProcessor(self.config, self.client)
        self.database = LegalDatabase(self.config)
        self.conversation_manager = ConversationManager(self.config, self.client, self.database)
        
        # Wake word components
        self.wake_word_detector = None
        self.listening_for_wake_word = True
        self._setup_wake_word_detection()
    
    def _setup_wake_word_detection(self):
        """Setup wake word detection system."""
        print(f"🔧 Setting up wake word detection for '{self.config.wake_word}'...")
        
        samples_dir = f"wake_word_samples_{self.config.wake_word}"
        
        if not os.path.exists(samples_dir) or not os.listdir(samples_dir):
            print(f"❌ No training samples found for '{self.config.wake_word}'")
            
            train_choice = input(f"Train wake word '{self.config.wake_word}' now? (y/n): ").lower().strip()
            
            if train_choice == 'y':
                trainer = WakeWordTrainer(self.config)
                if trainer.collect_samples(10):
                    print("✅ Wake word training completed!")
                else:
                    print("⚠️ Insufficient samples, using Whisper fallback")
                    return
            else:
                print("⚠️ Continuing without custom wake word detection")
                return
        
        try:
            self.wake_word_detector = WakeWordDetector(self.config)
            if self.wake_word_detector.reference_features:
                print(f"✅ Wake word detector initialized with {len(self.wake_word_detector.reference_features)} samples")
            else:
                print("❌ Wake word detector failed to load samples")
                self.wake_word_detector = None
        except Exception as e:
            logger.error(f"Wake word detector initialization failed: {e}")
            self.wake_word_detector = None
    
    def _detect_wake_word(self, audio_data: bytes) -> Tuple[bool, float]:
        """Detect wake word using custom detector or Whisper fallback."""
        if not audio_data:
            return False, 0.0
        
        try:
            # Try custom detector first
            if self.wake_word_detector:
                detected, confidence = self.wake_word_detector.detect(audio_data)
                if detected:
                    print(f"🎯 Wake word detected! Confidence: {confidence:.3f}")
                    return True, confidence
                
                # If close but not quite, try Whisper backup
                if confidence > 0.4:
                    return self._whisper_wake_word_detection(audio_data)
            
            # Fallback to Whisper
            return self._whisper_wake_word_detection(audio_data)
            
        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return False, 0.0
    
    def _whisper_wake_word_detection(self, audio_data: bytes) -> Tuple[bool, float]:
        """Whisper-based wake word detection."""
        try:
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(audio_array) == 0:
                return False, 0.0
            
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            result = self.speech_processor.whisper_model.transcribe(
                audio_float,
                fp16=False,
                language='en',
                no_speech_threshold=0.6
            )
            
            text = result["text"].strip().lower()
            confidence = 1.0 - float(result.get("no_speech_prob", 0.5))
            
            # Check for wake word variations
            wake_words = [self.config.wake_word, "racy", "tracy", "crazy"]
            detected = any(word in text for word in wake_words)
            
            if detected:
                print(f"🎯 Wake word detected via Whisper: '{text}'")
            
            return detected, confidence
            
        except Exception as e:
            logger.error(f"Whisper wake word detection error: {e}")
            return False, 0.0
    
    async def run(self):
        """Main assistant loop."""
        print("\n🚀 Voice Assistant Ready!")
        print(f"🎯 Say '{self.config.wake_word}' to activate")
        print("\n📋 Available commands:")
        print("   • 'Show my cases' - List legal cases")
        print("   • 'Show hearings' - Upcoming court dates")
        print("   • 'Add task [description]' - Create new task")
        print("   • 'Show tasks' - List pending tasks")
        print("   • Ask me anything!")
        print("   • 'quit' to exit")
        print("\n" + "="*50)
        
        consecutive_errors = 0
        max_errors = 5
        last_activation = 0
        
        while True:
            try:
                current_time = asyncio.get_event_loop().time()
                
                if self.listening_for_wake_word:
                    # Listen for wake word
                    print("👂 Listening for wake word...")
                    audio_data = self.audio_processor.record_audio(
                        self.config.wake_word_record_duration, 
                        show_listening_message=False
                    )
                    
                    if not audio_data:
                        consecutive_errors += 1
                        if consecutive_errors >= max_errors:
                            print("❌ Too many audio errors. Check microphone.")
                            break
                        continue
                    
                    consecutive_errors = 0
                    
                    # Check for wake word
                    detected, confidence = self._detect_wake_word(audio_data)
                    
                    if detected and (current_time - last_activation) > self.config.activation_cooldown:
                        print("✨ Rasi activated! What can I help you with?")
                        self.listening_for_wake_word = False
                        last_activation = current_time
                        
                        # Wait for user to start speaking
                        await asyncio.sleep(0.8)
                        
                        # Record command
                        print("🎤 Listening for your command...")
                        command_audio = self.audio_processor.record_audio(self.config.record_duration)
                        
                        if not command_audio:
                            print("❌ No command detected. Try again.")
                            self.listening_for_wake_word = True
                            continue
                        
                        # Process command
                        user_input = self.speech_processor.speech_to_text(command_audio)
                        
                        if user_input and len(user_input.strip()) > 0:
                            # Check for exit
                            if any(word in user_input.lower() for word in ['quit', 'exit', 'goodbye', 'stop']):
                                response = "Goodbye! It was great talking with you!"
                                tts_thread = self.speech_processor.text_to_speech(response)
                                if tts_thread:
                                    tts_thread.join(timeout=8)
                                break
                            
                            # Generate response
                            try:
                                response = await asyncio.wait_for(
                                    self.conversation_manager.process_user_input(user_input),
                                    timeout=30.0
                                )
                                
                                # Speak response
                                tts_thread = self.speech_processor.text_to_speech(response)
                                if tts_thread:
                                    tts_thread.join(timeout=15)
                                    
                            except asyncio.TimeoutError:
                                self.speech_processor.text_to_speech("Sorry, that took too long to process.")
                            except Exception as e:
                                logger.error(f"Error processing input: {e}")
                                self.speech_processor.text_to_speech("Sorry, I encountered an error.")
                        
                        # Reset to listening
                        self.listening_for_wake_word = True
                        print(f"\n👂 Ready... Say '{self.config.wake_word}' to activate again")
                    
                    elif detected:
                        print(f"⏰ Wake word detected but in cooldown")
                    
                    # Small delay
                    await asyncio.sleep(0.05)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Critical error: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= max_errors:
                    print("❌ Too many errors. Shutting down.")
                    break
                
                print("❌ Error occurred. Recovering...")
                self.listening_for_wake_word = True
                await asyncio.sleep(1.0)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def install_requirements():
    """Install required packages."""
    import subprocess
    import sys
    
    requirements = [
        "openai",
        "openai-whisper", 
        "pyttsx3",
        "pyaudio",
        "pygame",
        "numpy"
    ]
    
    print("📦 Installing requirements...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed: {package}")

def test_microphone():
    """Test microphone functionality."""
    try:
        audio_test = pyaudio.PyAudio()
        info = audio_test.get_default_input_device_info()
        print(f"✅ Microphone: {info['name']}")
        audio_test.terminate()
        return True
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main application entry point."""
    print("🎙️ Legal Voice Assistant")
    print("=" * 30)
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    # Test audio system
    print("\n🎤 Testing audio...")
    if not test_microphone():
        choice = input("Continue anyway? (y/n): ").lower()
        if choice != 'y':
            return
    
    # Initialize and run
    try:
        config = AssistantConfig()
        assistant = VoiceAssistant(api_key, config)
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        logger.error(f"Failed to start: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

# ============================================================================
# USAGE DOCUMENTATION
# ============================================================================

"""
MODULAR VOICE ASSISTANT USAGE:

INSTALLATION:
pip install openai openai-whisper pyttsx3 pyaudio pygame numpy

FEATURES:
✅ Modular architecture with clean separation
✅ Wake word training and detection
✅ Natural speech synthesis with OpenAI TTS
✅ Legal database management
✅ Conversational AI with personality
✅ Production-ready error handling
✅ Comprehensive logging
✅ Easy configuration management

MODULES:
- AudioProcessor: Audio recording and file handling
- WakeWordTrainer: Sample collection for training
- WakeWordDetector: Feature-based wake word detection
- SpeechProcessor: Speech recognition and synthesis
- LegalDatabase: SQLite database operations
- ConversationManager: AI conversation flow
- VoiceAssistant: Main orchestrator class

CONFIGURATION:
Modify AssistantConfig class to customize:
- Wake word
- Whisper model size
- TTS voice
- Audio settings
- Database path
- Conversation history length

VOICE COMMANDS:
- "Rasi" (wake word)
- "Show my cases"
- "Show hearings" 
- "Add task [description]"
- "Show tasks"
- "Add case [title]"
- General conversation

ERROR HANDLING:
- Automatic recovery from audio errors
- Fallback TTS systems
- Database error recovery
- Network timeout handling
- Graceful degradation

CUSTOMIZATION:
- Easy to extend with new modules
- Plugin architecture ready
- Configuration-driven behavior
- Modular testing capabilities
"""
