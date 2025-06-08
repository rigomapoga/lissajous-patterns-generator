import math
import numpy as np
import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Initialize Flask app and SocketIO
app = Flask(__name__)

# --- IMPORTANT: Configure Secret Key ---
# In a real production environment, you would never hardcode this.
# For Render, you should set SECRET_KEY as an environment variable in the Render Dashboard.
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your_super_secret_key_for_development')

# --- Configure SocketIO for CORS ---
# For testing and development, "cors_allowed_origins='*'" allows connections from any origin.
# For production, it is HIGHLY RECOMMENDED to replace '*' with the specific domain(s)
# of your WordPress website(s) for security reasons, e.g.:
# socketio = SocketIO(app, cors_allowed_origins=["https://yourwordpressdomain.com"])
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Configure Flask-CORS for regular HTTP routes ---
CORS(app)

class WaveformGenerator:
    """
    Generates numerical data for different types of waveforms with phase shift.
    """
    def __init__(self, sample_rate: int = 2000, duration: float = 2.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_points = int(sample_rate * duration)
        self.time_points = np.linspace(0, duration, self.num_points, endpoint=False)

    def _generate_wave(self, wave_type: str, frequency: float, amplitude: float, phase_deg: float) -> list[float]:
        """
        Generates waveform data based on type, frequency, amplitude, and phase shift.
        Phase shift is applied directly to the angular argument.
        Args:
            wave_type (str): Type of wave ('sine', 'square', 'sawtooth').
            frequency (float): Frequency of the wave in Hz (now expected as integer, but treated as float for math).
            amplitude (float): Amplitude of the wave (0.0 to 1.0).
            phase_deg (float): Phase shift in degrees (-90 to +90).
        Returns:
            list[float]: A list of numerical sample values for the waveform.
        """
        phase_rad = np.deg2rad(phase_deg)
        angular_arg = 2 * np.pi * frequency * self.time_points + phase_rad

        if wave_type == 'sine':
            samples = amplitude * np.sin(angular_arg)
        elif wave_type == 'square':
            samples = amplitude * np.sign(np.sin(angular_arg))
        elif wave_type == 'sawtooth':
            phase_fraction = phase_rad / (2 * np.pi)
            samples = amplitude * (2 * (np.fmod(self.time_points * frequency + phase_fraction, 1.0)) - 1)
        else:
            samples = np.zeros_like(self.time_points)
        
        return samples.tolist()

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page for the Lissajous plotter."""
    return render_template('index.html')

# --- Socket.IO Events ---

@socketio.on('connect')
def test_connect():
    """
    Handles new client connections to the WebSocket.
    Sends initial data for the X and Y waveforms to draw a basic Lissajous pattern.
    """
    print('Client connected:', request.sid)
    
    # Initial parameters for X-axis (Buffer 1) - Frequencies now integers
    initial_params_x = {'type': 'sine', 'frequency': 1, 'amplitude': 0.8, 'phase_deg': 0.0}
    # Initial parameters for Y-axis (Buffer 2) - Frequencies now integers
    initial_params_y = {'type': 'sine', 'frequency': 2, 'amplitude': 0.8, 'phase_deg': 90.0} # 90 deg for a circle/ellipse

    generator = WaveformGenerator()
    x_data = generator._generate_wave(
        initial_params_x['type'],
        float(initial_params_x['frequency']), # Cast to float for math
        initial_params_x['amplitude'],
        initial_params_x['phase_deg']
    )
    y_data = generator._generate_wave(
        initial_params_y['type'],
        float(initial_params_y['frequency']), # Cast to float for math
        initial_params_y['amplitude'],
        initial_params_y['phase_deg']
    )
    
    emit('waveform_data', {
        'x_data': x_data,
        'y_data': y_data,
        'params_x': initial_params_x,
        'params_y': initial_params_y
    })

@socketio.on('disconnect')
def test_disconnect():
    """Handles client disconnections from the WebSocket."""
    print('Client disconnected:', request.sid)

@socketio.on('update_params')
def handle_update_params(params):
    """
    Receives updated waveform parameters for both X and Y from the client,
    generates new data, and sends it back to the client.
    """
    # Extract parameters for X-axis
    params_x = params.get('x', {})
    wave_type_x = params_x.get('type', 'sine')
    frequency_x = int(params_x.get('frequency', 1)) # Explicitly get as integer
    amplitude_x = float(params_x.get('amplitude', 0.5))
    phase_deg_x = float(params_x.get('phase_deg', 0.0))

    # Extract parameters for Y-axis
    params_y = params.get('y', {})
    wave_type_y = params_y.get('type', 'sine')
    frequency_y = int(params_y.get('frequency', 1)) # Explicitly get as integer
    amplitude_y = float(params_y.get('amplitude', 0.5))
    phase_deg_y = float(params_y.get('phase_deg', 0.0))

    # Basic input validation for both X and Y parameters - Frequencies now clamped as integers
    frequency_x = max(1, min(frequency_x, 20)) # Clamp from 1 to 20
    amplitude_x = max(0.0, min(amplitude_x, 1.0))
    phase_deg_x = max(-90.0, min(phase_deg_x, 90.0))

    frequency_y = max(1, min(frequency_y, 20)) # Clamp from 1 to 20
    amplitude_y = max(0.0, min(amplitude_y, 1.0))
    phase_deg_y = max(-90.0, min(phase_deg_y, 90.0))

    generator = WaveformGenerator()
    x_data = generator._generate_wave(wave_type_x, float(frequency_x), amplitude_x, phase_deg_x) # Cast back to float for math
    y_data = generator._generate_wave(wave_type_y, float(frequency_y), amplitude_y, phase_deg_y) # Cast back to float for math
    
    emit('waveform_data', {
        'x_data': x_data,
        'y_data': y_data,
        'params_x': {'type': wave_type_x, 'frequency': frequency_x, 'amplitude': amplitude_x, 'phase_deg': phase_deg_x},
        'params_y': {'type': wave_type_y, 'frequency': frequency_y, 'amplitude': amplitude_y, 'phase_deg': phase_deg_y}
    })
    print(f"Received params for X and Y from {request.sid}: FreqX={frequency_x}, FreqY={frequency_y}. Sent new data.")


# --- Main Execution Block (for local development) ---
if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_dir = os.path.join(script_dir, 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        print(f"Created templates directory: {template_dir}")

    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0' 
    print(f"Running Lissajous Flask-SocketIO server locally on {host}:{port}...")
    print("For production deployment, use Gunicorn with the specified start command.")
    socketio.run(app, debug=True, host=host, port=port, allow_unsafe_werkzeug=True)
