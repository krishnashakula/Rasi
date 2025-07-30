#!/usr/bin/env python3
"""
Railway-Deployable Voice Assistant with Web Interface

A voice assistant that can be deployed on Railway with a web interface
for legal database management and voice interactions.

Requirements:
- FastAPI for web interface
- WebSocket for real-time communication
- SQLite database (file-based for Railway)
- OpenAI integration
"""

import asyncio
import sqlite3
import os
import logging
import json
import base64
import io
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import uvicorn

# Web framework
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# AI libraries
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AssistantConfig:
    """Configuration settings for the voice assistant."""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_tts_voice: str = "nova"
    database_path: str = "railway_legal_assistant.db"
    max_conversation_history: int = 10
    port: int = int(os.getenv("PORT", 8000))
    host: str = "0.0.0.0"

# ============================================================================
# DATABASE MODULE
# ============================================================================

class LegalDatabase:
    """Railway-optimized legal database with SQLite."""
    
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.db_path = config.database_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database with legal tables and sample data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            self._create_tables(cursor)
            self._insert_sample_data(cursor)
            conn.commit()
            logger.info(f"Legal database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
        finally:
            conn.close()
    
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
    
    def _insert_sample_data(self, cursor):
        """Insert sample legal data if database is empty."""
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
            
            # Sample cases
            cursor.execute("""INSERT INTO cases 
                           (case_number, client_id, attorney_id, case_type, case_title, description, 
                            filing_date, court_name, judge_name, estimated_value, priority_level) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                         ("2024-CV-001", 1, 1, "Civil", "Contract Dispute", "Commercial contract disagreement", 
                          "2024-01-15", "District Court", "Judge Wilson", 50000.00, "high"))
            
            # Sample tasks
            cursor.execute("""INSERT INTO tasks 
                           (user_id, title, description, priority, due_date) 
                           VALUES (?, ?, ?, ?, ?)""", 
                         (1, "Review contract terms", "Analyze the disputed contract clauses", "high", "2024-08-15"))
            
            # Sample hearings
            cursor.execute("""INSERT INTO hearings 
                           (case_id, hearing_date, hearing_type, court_room, judge_name, hearing_status) 
                           VALUES (?, ?, ?, ?, ?, ?)""", 
                         (1, "2024-08-30 10:00:00", "Pre-trial", "Courtroom A", "Judge Wilson", "scheduled"))
    
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
    """Manages AI conversation and database interactions."""
    
    def __init__(self, config: AssistantConfig, database: LegalDatabase):
        self.config = config
        self.database = database
        self.conversation_history = []
        
        if not config.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=config.openai_api_key)
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Check for database operations
        db_response = await self._handle_database_request(user_input)
        
        # System message for AI personality
        system_message = """You are Rasi, a witty and helpful legal assistant AI with a warm personality. 
        You have access to a comprehensive legal database and love helping with legal matters.

        Your personality:
        - Naturally conversational and humorous
        - Professional but not robotic
        - Helpful and knowledgeable about legal matters
        - Use casual language and contractions
        - Be concise but informative

        You can help with:
        - Legal case management
        - Task tracking
        - Hearing schedules
        - General legal questions
        - Database queries"""
        
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
                max_tokens=300,
                temperature=0.8
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
        if any(phrase in input_lower for phrase in ["show cases", "list cases", "my cases", "cases"]):
            result = self.database.execute_query("""
                SELECT c.case_number, c.case_title, c.case_status, a.name as attorney_name, 
                       c.priority_level, c.estimated_value
                FROM cases c 
                LEFT JOIN attorneys a ON c.attorney_id = a.id 
                ORDER BY c.created_at DESC LIMIT 5
            """)
            
            if result["success"] and result["data"]:
                cases = result["data"]
                response = f"Here are your {len(cases)} most recent cases:\n\n"
                for case in cases:
                    response += f"📋 {case['case_number']}: {case['case_title']}\n"
                    response += f"   Status: {case['case_status']} | Priority: {case['priority_level']}\n"
                    if case['estimated_value']:
                        response += f"   Value: ${case['estimated_value']:,.2f}\n"
                    response += f"   Attorney: {case['attorney_name'] or 'Unassigned'}\n\n"
                return response.strip()
            else:
                return "No cases found in your legal database."
        
        # Hearings
        elif any(phrase in input_lower for phrase in ["show hearings", "hearings", "court dates"]):
            result = self.database.execute_query("""
                SELECT h.hearing_date, h.hearing_type, h.court_room, h.judge_name,
                       c.case_number, c.case_title, h.hearing_status
                FROM hearings h
                JOIN cases c ON h.case_id = c.id
                WHERE h.hearing_date >= datetime('now')
                ORDER BY h.hearing_date ASC LIMIT 5
            """)
            
            if result["success"] and result["data"]:
                hearings = result["data"]
                response = f"Upcoming hearings ({len(hearings)}):\n\n"
                for hearing in hearings:
                    response += f"📅 {hearing['hearing_date']}\n"
                    response += f"   Case: {hearing['case_number']} - {hearing['case_title']}\n"
                    response += f"   Type: {hearing['hearing_type']} | Status: {hearing['hearing_status']}\n"
                    response += f"   Judge: {hearing['judge_name']}, Room: {hearing['court_room']}\n\n"
                return response.strip()
            else:
                return "No upcoming hearings scheduled."
        
        # Tasks
        elif any(phrase in input_lower for phrase in ["show tasks", "tasks", "my tasks"]):
            result = self.database.execute_query("""
                SELECT title, priority, completed, due_date, description
                FROM tasks 
                WHERE user_id = 1 AND completed = 0
                ORDER BY priority DESC, due_date ASC
                LIMIT 10
            """)
            
            if result["success"] and result["data"]:
                tasks = result["data"]
                if tasks:
                    response = f"You have {len(tasks)} pending tasks:\n\n"
                    for task in tasks:
                        priority_emoji = "🔥" if task["priority"] == "high" else "⚡" if task["priority"] == "medium" else "📌"
                        response += f"{priority_emoji} {task['title']}\n"
                        if task["due_date"]:
                            response += f"   Due: {task['due_date']}\n"
                        if task["description"]:
                            response += f"   Note: {task['description']}\n"
                        response += "\n"
                    return response.strip()
                else:
                    return "All caught up! No pending tasks."
            else:
                return "No tasks found."
        
        return None
    
    async def generate_speech(self, text: str) -> bytes:
        """Generate speech audio from text using OpenAI TTS."""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.config.openai_tts_voice,
                input=text,
                response_format="mp3"
            )
            
            audio_data = b""
            for chunk in response.iter_bytes():
                audio_data += chunk
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return b""

# ============================================================================
# API MODELS
# ============================================================================

class ChatMessage(BaseModel):
    message: str

class AudioMessage(BaseModel):
    audio_data: str  # Base64 encoded audio
    format: str = "webm"

# ============================================================================
# WEB APPLICATION
# ============================================================================

# Initialize configuration and dependencies
config = AssistantConfig()
database = LegalDatabase(config)
conversation_manager = ConversationManager(config, database)

# Create FastAPI app
app = FastAPI(
    title="Railway Legal Voice Assistant",
    description="A voice-enabled legal assistant with database management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
connections = []

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Rasi - Legal Voice Assistant</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container { 
                background: white; 
                border-radius: 20px; 
                padding: 2rem; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                max-width: 800px;
                width: 90%;
            }
            .header { text-align: center; margin-bottom: 2rem; }
            .header h1 { color: #333; margin-bottom: 0.5rem; }
            .header p { color: #666; }
            .chat-container { 
                height: 400px; 
                border: 2px solid #eee; 
                border-radius: 10px; 
                padding: 1rem; 
                overflow-y: auto; 
                margin-bottom: 1rem;
                background: #f8f9fa;
            }
            .message { 
                margin: 0.5rem 0; 
                padding: 0.5rem 1rem; 
                border-radius: 10px; 
                max-width: 80%;
            }
            .user { 
                background: #007bff; 
                color: white; 
                margin-left: auto; 
                text-align: right;
            }
            .assistant { 
                background: #e9ecef; 
                color: #333; 
            }
            .input-group { 
                display: flex; 
                gap: 1rem; 
                margin-bottom: 1rem;
            }
            .input-group input { 
                flex: 1; 
                padding: 0.75rem; 
                border: 2px solid #ddd; 
                border-radius: 10px; 
                font-size: 1rem;
            }
            .btn { 
                padding: 0.75rem 1.5rem; 
                border: none; 
                border-radius: 10px; 
                cursor: pointer; 
                font-size: 1rem;
                transition: all 0.3s;
            }
            .btn-primary { 
                background: #007bff; 
                color: white; 
            }
            .btn-primary:hover { background: #0056b3; }
            .btn-success { 
                background: #28a745; 
                color: white; 
            }
            .btn-success:hover { background: #1e7e34; }
            .btn-danger { 
                background: #dc3545; 
                color: white; 
            }
            .btn-danger:hover { background: #c82333; }
            .voice-controls { 
                display: flex; 
                gap: 1rem; 
                justify-content: center;
            }
            .status { 
                text-align: center; 
                margin: 1rem 0; 
                font-weight: bold;
            }
            .recording { color: #dc3545; }
            .processing { color: #ffc107; }
            .ready { color: #28a745; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Rasi</h1>
                <p>Your AI Legal Assistant</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message assistant">
                    Hi! I'm Rasi, your legal assistant. I can help you manage cases, tasks, hearings, and answer legal questions. 
                    Try asking me to "show my cases" or "list my tasks"!
                </div>
            </div>
            
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Type your message..." />
                <button class="btn btn-primary" onclick="sendMessage()">Send</button>
            </div>
            
            <div class="voice-controls">
                <button class="btn btn-success" id="recordBtn" onclick="toggleRecording()">🎤 Start Recording</button>
                <button class="btn btn-danger" id="stopBtn" onclick="stopAudio()" disabled>⏹️ Stop Audio</button>
            </div>
            
            <div class="status ready" id="status">Ready</div>
        </div>

        <script>
            const ws = new WebSocket(`wss://${window.location.host}/ws`);
            const chatContainer = document.getElementById('chatContainer');
            const messageInput = document.getElementById('messageInput');
            const recordBtn = document.getElementById('recordBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            
            let mediaRecorder;
            let audioChunks = [];
            let currentAudio = null;
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'text_response') {
                    addMessage(data.message, 'assistant');
                } else if (data.type === 'audio_response') {
                    playAudio(data.audio_data);
                } else if (data.type === 'error') {
                    addMessage('Error: ' + data.message, 'assistant');
                }
                
                setStatus('ready', 'Ready');
            };
            
            function addMessage(text, sender) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}`;
                messageDiv.textContent = text;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
            
            function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage(message, 'user');
                setStatus('processing', 'Processing...');
                
                ws.send(JSON.stringify({
                    type: 'text_message',
                    message: message
                }));
                
                messageInput.value = '';
            }
            
            async function toggleRecording() {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                    recordBtn.textContent = '🎤 Start Recording';
                    recordBtn.disabled = false;
                    setStatus('processing', 'Processing audio...');
                } else {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = async () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            const reader = new FileReader();
                            reader.onload = function() {
                                const audioData = reader.result.split(',')[1]; // Remove data:audio/webm;base64,
                                
                                ws.send(JSON.stringify({
                                    type: 'audio_message',
                                    audio_data: audioData,
                                    format: 'webm'
                                }));
                            };
                            reader.readAsDataURL(audioBlob);
                            
                            stream.getTracks().forEach(track => track.stop());
                        };
                        
                        mediaRecorder.start();
                        recordBtn.textContent = '⏹️ Stop Recording';
                        setStatus('recording', 'Recording...');
                        
                    } catch (err) {
                        console.error('Error accessing microphone:', err);
                        alert('Could not access microphone. Please ensure you have granted permission.');
                    }
                }
            }
            
            function playAudio(audioData) {
                stopAudio(); // Stop any currently playing audio
                
                const audioBlob = new Blob([Uint8Array.from(atob(audioData), c => c.charCodeAt(0))], { type: 'audio/mpeg' });
                const audioUrl = URL.createObjectURL(audioBlob);
                currentAudio = new Audio(audioUrl);
                
                currentAudio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    stopBtn.disabled = true;
                    setStatus('ready', 'Ready');
                };
                
                currentAudio.play();
                stopBtn.disabled = false;
                setStatus('processing', 'Playing response...');
            }
            
            function stopAudio() {
                if (currentAudio) {
                    currentAudio.pause();
                    currentAudio.currentTime = 0;
                    currentAudio = null;
                }
                stopBtn.disabled = true;
            }
            
            function setStatus(className, text) {
                status.className = `status ${className}`;
                status.textContent = text;
            }
            
            // Enter key to send message
            messageInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // Handle WebSocket connection
            ws.onopen = function() {
                setStatus('ready', 'Connected & Ready');
            };
            
            ws.onclose = function() {
                setStatus('processing', 'Disconnected');
            };
            
            ws.onerror = function() {
                setStatus('processing', 'Connection Error');
            };
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections for real-time communication."""
    await websocket.accept()
    connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data["type"] == "text_message":
                # Process text message
                user_message = message_data["message"]
                
                try:
                    response = await conversation_manager.process_user_input(user_message)
                    
                    # Send text response
                    await websocket.send_text(json.dumps({
                        "type": "text_response",
                        "message": response
                    }))
                    
                    # Generate and send audio response
                    audio_data = await conversation_manager.generate_speech(response)
                    if audio_data:
                        audio_b64 = base64.b64encode(audio_data).decode()
                        await websocket.send_text(json.dumps({
                            "type": "audio_response",
                            "audio_data": audio_b64
                        }))
                
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Sorry, I encountered an error processing your request."
                    }))
            
            elif message_data["type"] == "audio_message":
                # Process audio message (speech-to-text would go here)
                await websocket.send_text(json.dumps({
                    "type": "text_response",
                    "message": "Audio processing is not yet implemented in this Railway deployment. Please use text messages for now."
                }))
    
    except WebSocketDisconnect:
        connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connections:
            connections.remove(websocket)

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """REST API endpoint for chat messages."""
    try:
        response = await conversation_manager.process_user_input(message.message)
        return {"response": response, "success": True}
    except Exception as e:
        logger.error(f"Chat API error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/cases")
async def get_cases():
    """Get all legal cases."""
    result = database.execute_query("""
        SELECT c.*, a.name as attorney_name 
        FROM cases c 
        LEFT JOIN attorneys a ON c.attorney_id = a.id 
        ORDER BY c.created_at DESC
    """)
    
    if result["success"]:
        return {"cases": result["data"], "success": True}
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks."""
    result = database.execute_query("""
        SELECT t.*, c.case_number, c.case_title 
        FROM tasks t 
        LEFT JOIN cases c ON t.case_id = c.id 
        WHERE t.completed = 0
        ORDER BY t.priority DESC, t.due_date ASC
    """)
    
    if result["success"]:
        return {"tasks": result["data"], "success": True}
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/api/hearings")
async def get_hearings():
    """Get upcoming hearings."""
    result = database.execute_query("""
        SELECT h.*, c.case_number, c.case_title 
        FROM hearings h 
        JOIN cases c ON h.case_id = c.id 
        WHERE h.hearing_date >= datetime('now')
        ORDER BY h.hearing_date ASC
    """)
    
    if result["success"]:
        return {"hearings": result["data"], "success": True}
    else:
        raise HTTPException(status_code=500, detail=result["error"])

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected" if os.path.exists(config.database_path) else "disconnected"
    }

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main application entry point for Railway deployment."""
    logger.info("Starting Railway Legal Voice Assistant...")
    
    # Validate configuration
    if not config.openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is required")
        raise ValueError("OpenAI API key must be set")
    
    logger.info(f"Database initialized at: {config.database_path}")
    logger.info(f"Starting server on {config.host}:{config.port}")
    
    # Run the FastAPI application
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()