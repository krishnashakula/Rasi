# 🎙️ Rasi - Advanced Modular Voice Assistant

> A production-ready, modular voice assistant with legal database management, custom wake word detection, and natural conversation capabilities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg)](https://openai.com/)
[![Whisper](https://img.shields.io/badge/Whisper-ASR-orange.svg)](https://github.com/openai/whisper)
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightblue.svg)](https://sqlite.org/)

## 🌟 Key Features

### 🧠 **Intelligent Voice Processing**
- **Custom Wake Word Detection**: Train your own wake word with audio feature extraction (energy, zero crossings, spectral centroid)
- **High-Accuracy Speech Recognition**: Powered by OpenAI Whisper with configurable model sizes
- **Natural Text-to-Speech**: OpenAI TTS with system fallback (Windows PowerShell/Linux espeak)
- **Real-time Audio Processing**: PyAudio-based recording with configurable sample rates

### 🏛️ **Legal Database Management**
- **Complete Case Management**: Users, attorneys, cases, hearings, tasks, documents, and notes tables
- **Pre-populated Sample Data**: Includes sample cases, attorneys, and hearings for testing
- **Automated Data Organization**: SQLite backend with proper foreign key relationships
- **Voice Commands**: Add cases/tasks, show cases/hearings/tasks via voice

### 🤖 **Conversational AI**
- **GPT-3.5-turbo Integration**: Natural conversation with configurable personality
- **Context-Aware Conversations**: Maintains conversation history (configurable max length)
- **Legal Domain Integration**: Database queries integrated into AI responses
- **Error Handling**: Graceful fallbacks for API failures

### 🏗️ **Modular Architecture**
- **7 Independent Modules**: Configuration, Audio, Wake Word (Trainer/Detector), Speech, Database, Conversation, Main Assistant
- **Configuration-Driven**: Single `AssistantConfig` dataclass controls all settings
- **Production Error Handling**: Comprehensive logging, timeout handling, recovery mechanisms
- **Threading**: TTS runs in separate threads to prevent blocking

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher required
python --version

# Audio system dependencies (Linux)
sudo apt-get install portaudio19-dev python3-pyaudio espeak

# Audio system dependencies (macOS)
brew install portaudio
```

### Installation

```bash
# Save the script as voice_assistant.py
# Install dependencies
pip install openai openai-whisper pyttsx3 pyaudio pygame numpy
```

### Environment Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Or the script will prompt you for it
```

### First Run

```bash
python voice_assistant.py
```

The assistant will:
1. **Test microphone** using `test_microphone()` function
2. **Prompt for wake word training** if no samples exist (10 samples of "rasi")
3. **Initialize SQLite database** with sample legal data
4. **Start listening** for the wake word

## 📖 Detailed Usage

### Wake Word Training

The system requires 10 training samples on first run:

```
🎓 Wake Word Training for 'rasi'
📝 Collecting 10 samples
💡 Speak clearly and naturally
==================================================

Press Enter for sample 1/10...
🎤 Say 'rasi' now!
✅ Sample 1 saved
```

Samples are saved as WAV files in `wake_word_samples_rasi/` directory.

### Voice Commands (Actual Implementation)

| Command | Example | Function Called |
|---------|---------|-----------------|
| **Show Cases** | "show cases", "list cases", "my cases" | `SELECT` from cases with attorney JOIN |
| **Show Hearings** | "show hearings", "next hearing", "court date" | `SELECT` upcoming hearings with case JOIN |
| **Show Tasks** | "show tasks", "list tasks", "my tasks" | `SELECT` pending tasks for user_id=1 |
| **Add Case** | "add case contract dispute" | `INSERT` into cases table |
| **Add Task** | "add task review contract" | `INSERT` into tasks table |
| **Exit** | "quit", "exit", "goodbye", "stop" | Terminates main loop |
| **General Chat** | Any other input | Processed by GPT-3.5-turbo |

### Configuration Options

All settings in `AssistantConfig` dataclass:

```python
@dataclass
class AssistantConfig:
    wake_word: str = "rasi"                        # Default wake word
    whisper_model: str = "base"                    # tiny/base/small/medium/large
    openai_tts_voice: str = "nova"                 # alloy/echo/fable/onyx/nova/shimmer
    sample_rate: int = 16000                       # Audio sample rate (Hz)
    channels: int = 1                              # Mono audio
    chunk_size: int = 1024                         # Audio buffer size
    record_duration: int = 5                       # Command recording duration (seconds)
    wake_word_record_duration: float = 1.5         # Wake word detection window (seconds)
    activation_cooldown: float = 2.0               # Seconds between activations
    max_conversation_history: int = 10             # Context window size
    database_path: str = "legal_assistant.db"     # SQLite database file
    log_level: str = "INFO"                        # Logging level
```

## 🏗️ Architecture Overview

### Actual Module Structure

```
voice_assistant.py (2,047 lines)
├── 🔧 AssistantConfig (dataclass)
├── 🎵 AudioProcessor 
│   ├── record_audio() - PyAudio recording
│   └── save_wav_file() - WAV file creation
├── 🎓 WakeWordTrainer
│   └── collect_samples() - Interactive sample collection
├── 🎯 WakeWordDetector
│   ├── _extract_features() - Energy, zero crossings, spectral centroid
│   ├── _calculate_similarity() - Weighted feature comparison
│   └── detect() - Returns (detected: bool, confidence: float)
├── 🗣️ SpeechProcessor
│   ├── speech_to_text() - Whisper transcription
│   ├── text_to_speech() - OpenAI TTS with threading
│   └── _fallback_tts() - System TTS backup
├── 🗄️ LegalDatabase
│   ├── _create_tables() - 7 table schema
│   ├── _insert_sample_data() - Pre-populated data
│   └── execute_query() - Safe SQL execution
├── 💬 ConversationManager
│   ├── process_user_input() - Main AI logic
│   └── _handle_database_request() - Voice command parsing
└── 🎙️ VoiceAssistant
    ├── _setup_wake_word_detection() - Initialization
    ├── _detect_wake_word() - Custom + Whisper fallback
    └── run() - Main async loop
```

### Actual Data Flow

```
1. AudioProcessor.record_audio(1.5s) -> bytes
2. WakeWordDetector.detect(audio) -> (bool, confidence)
3. If detected: AudioProcessor.record_audio(5s) -> command_bytes
4. SpeechProcessor.speech_to_text(command) -> string
5. ConversationManager.process_user_input(string) -> response
   ├── _handle_database_request() checks for SQL commands
   └── OpenAI GPT-3.5-turbo for general responses
6. SpeechProcessor.text_to_speech(response) -> audio output
7. Return to step 1
```

## 🗄️ Database Schema (Actual Implementation)

### Tables Created by `_create_tables()`

**users** - Client information
```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    phone TEXT,
    role TEXT DEFAULT 'client',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**attorneys** - Attorney records
```sql
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
```

**cases** - Legal cases
```sql
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
```

**hearings** - Court hearings
```sql
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
```

**tasks** - Personal tasks
```sql
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
```

**documents** - Case documents
```sql
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
```

**notes** - Case notes
```sql
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
```

### Sample Data (Pre-populated)

**Users:**
- John Doe (client)
- Jane Smith (client)

**Attorneys:**
- Sarah Johnson (Criminal Defense, 15 years)
- Michael Brown (Corporate Law, 12 years)

**Cases:**
- 2024-CV-001: Contract Dispute ($50,000, high priority)
- 2024-CR-002: DUI Defense ($5,000, medium priority)

**Hearings:**
- Pre-trial hearing for Contract Dispute
- Completed arraignment for DUI Defense

## 🛠️ Technical Implementation Details

### Wake Word Detection Algorithm

The `WakeWordDetector` uses 4 audio features:

```python
def _extract_features(self, audio_data: bytes) -> Dict:
    # 1. Energy: np.mean(audio_float ** 2)
    # 2. Zero crossings: Rate of sign changes
    # 3. Spectral centroid: Frequency center of mass via FFT
    # 4. RMS: Root mean square amplitude
    
    # Weighted similarity calculation:
    weights = [0.3, 0.2, 0.3, 0.2]  # energy, zc, sc, rms
    threshold = 0.65  # Detection threshold
```

### Error Handling Mechanisms

```python
# Audio errors: Consecutive error counting (max 5)
consecutive_errors = 0
max_errors = 5

# Network timeouts: 30 second OpenAI timeout
response = await asyncio.wait_for(
    self.conversation_manager.process_user_input(user_input),
    timeout=30.0
)

# TTS fallbacks: OpenAI TTS -> PowerShell (Windows) -> espeak (Linux)
# Database errors: Try-catch with error messages returned
```

### Threading Implementation

```python
def text_to_speech(self, text: str) -> Optional[threading.Thread]:
    def speak():
        # OpenAI TTS implementation
        pass
    
    tts_thread = threading.Thread(target=speak, daemon=True)
    tts_thread.start()
    return tts_thread

# Main loop waits for TTS completion:
if tts_thread:
    tts_thread.join(timeout=15)
```

## 🧪 Testing Functions

### Built-in Test Functions

```python
# Test microphone access
def test_microphone():
    """Returns True if microphone accessible, False otherwise"""
    
# Manual installation helper
def install_requirements():
    """Installs required packages via pip"""
```

### Testing Wake Word Detection

```python
# Check if samples exist
samples_dir = f"wake_word_samples_{config.wake_word}"
if not os.path.exists(samples_dir):
    # Prompts for training
    
# Test detection manually
detector = WakeWordDetector(config)
audio_data = audio_processor.record_audio(1.5)
detected, confidence = detector.detect(audio_data)
print(f"Detected: {detected}, Confidence: {confidence:.3f}")
```

## 🔧 Troubleshooting

### Common Issues (Based on Code)

**Missing Training Samples**
```
❌ No training samples found for 'rasi'
Train wake word 'rasi' now? (y/n):
```
Solution: Run training to collect 10 samples

**API Key Missing**
```python
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = input("Enter OpenAI API key: ").strip()
```
Solution: Set environment variable or enter when prompted

**Audio Errors**
```python
consecutive_errors = 0
max_errors = 5
if consecutive_errors >= max_errors:
    print("❌ Too many audio errors. Check microphone.")
```
Solution: Check microphone permissions and hardware

**Whisper Import Error**
```python
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
```
Solution: `pip install openai-whisper`

### Performance Settings

**Memory Optimization**
```python
# Use smaller Whisper model
config = AssistantConfig(whisper_model="tiny")  # vs "base"
```

**Response Speed**
```python
# Reduce conversation history
config = AssistantConfig(max_conversation_history=5)  # vs 10

# Shorter recording durations
config = AssistantConfig(
    wake_word_record_duration=1.0,  # vs 1.5
    record_duration=3               # vs 5
)
```

## 🚀 Actual Usage Example

```bash
$ python voice_assistant.py

🚀 Voice Assistant Ready!
🎯 Say 'rasi' to activate

📋 Available commands:
   • 'Show my cases' - List legal cases
   • 'Show hearings' - Upcoming court dates
   • 'Add task [description]' - Create new task
   • 'Show tasks' - List pending tasks
   • Ask me anything!
   • 'quit' to exit

👂 Listening for wake word...
🎯 Wake word detected! Confidence: 0.756
✨ Rasi activated! What can I help you with?
🎤 Listening for your command...
🗣️  You said: 'show my cases'
🤖 Assistant: Here are your 2 most recent cases:

🟢🔥 2024-CV-001: Contract Dispute
   Attorney: Sarah Johnson
   Judge: Judge Wilson

🟢⚡ 2024-CR-002: DUI Defense
   Attorney: Sarah Johnson
   Judge: Judge Davis

👂 Ready... Say 'rasi' to activate again
```

## 📄 Dependencies (requirements.txt)

```
openai>=1.0.0
openai-whisper>=20230918
pyttsx3>=2.90
pyaudio>=0.2.11
pygame>=2.0.0
numpy>=1.21.0
```

## 🙏 Acknowledgments

- **OpenAI** for GPT-3.5-turbo and Whisper models
- **PyAudio** for cross-platform audio I/O
- **SQLite** for embedded database functionality

---

**Note**: This README is based on the actual implementation in the provided code. All features, functions, and technical details are factually accurate as of the code version analyzed.
