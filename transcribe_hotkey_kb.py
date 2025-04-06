# Filnamn: transcribe_hotkey_tray_v4.py 

# Importera nödvändiga bibliotek
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel 
import threading
import time
import os
import sys 
from pynput import keyboard
from pynput.keyboard import Key, Controller as KeyboardController
import signal 
import pyperclip 
import queue 
import uuid 
import traceback 

# --- Försök importera Tray Icon bibliotek ---
try:
    import pystray
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    print("VARNING: pystray eller Pillow är inte installerat. Kör 'pip install pystray Pillow' för att få systemfältsikonen.", flush=True)
    PIL_AVAILABLE = False
# ------------------------------------

# --- Konfiguration ---
MODEL_NAME = "KBLab/kb-whisper-base" 
COMPUTE_TYPE = "int8" 
DEVICE = "cpu" 
SAMPLE_RATE = 16000
CHUNK_SECONDS = 20      
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE) 
TEMP_FILE_PREFIX = "temp_base_chunk_" 
START_STOP_HOTKEY_KEY = keyboard.Key.f9 
START_STOP_HOTKEY = {START_STOP_HOTKEY_KEY} 

# --- Ikonfilnamn (Använd dina namn här) ---
ICON_IDLE = "icon_idle.png"         # Ditt filnamn för idle
ICON_RECORDING = "icon_recording.png"  # Ditt filnamn för recording
ICON_PROCESSING = "icon_processing.png" # Ditt filnamn för processing (valfri)
ICON_ERROR = "icon_error.png"  # Behåll eller byt om du har en felikon
# --------------------------

# --- Globala Variabler ---
is_recording = False
audio_buffer = [] 
chunk_queue = queue.Queue() 
results = {} 
result_lock = threading.Lock()
chunk_counter = 0 
stop_event = threading.Event() 
worker_thread = None 
stream = None
model = None 
keyboard_controller = KeyboardController()
recording_lock = threading.Lock() 
listener = None 
exit_event = threading.Event() 
tray_icon = None
idle_icon_image = None
recording_icon_image = None
processing_icon_image = None
error_icon_image = None

# --- Funktioner för att uppdatera ikonen ---
def update_tray_icon(icon_image):
    if tray_icon and icon_image:
        try:
            tray_icon.icon = icon_image 
        except Exception as e:
            # print(f"DEBUG: Kunde inte uppdatera tray icon: {e}", flush=True) 
            pass 

# --- Transkriberings-Worker Tråd ---
def transcription_worker():
    global results, model 
    if model is None: print("FATAL WORKER ERROR: Modellen ej laddad!", flush=True); stop_event.set(); return 
    while not stop_event.is_set():
        try:
            chunk_index, chunk_data = chunk_queue.get(timeout=0.5) 
            if chunk_data is None: continue 
            temp_filename = f"{TEMP_FILE_PREFIX}{uuid.uuid4()}.wav"
            try:
                wav.write(temp_filename, SAMPLE_RATE, chunk_data)
                segments, info = model.transcribe(temp_filename, language="sv", beam_size=5, condition_on_previous_text=False)
                text_segments = [segment.text.strip() for segment in segments]
                transcribed_text = " ".join(text_segments)
                with result_lock: results[chunk_index] = transcribed_text
            except Exception as e:
                print(f"ERROR worker transcribing chunk {chunk_index}: {e}", flush=True)
                with result_lock: results[chunk_index] = f"[Fel chunk {chunk_index}]"
            finally:
                if os.path.exists(temp_filename):
                    try: os.remove(temp_filename)
                    except Exception: pass 
                chunk_queue.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"ERROR in worker main loop: {e}", flush=True); print(traceback.format_exc(), flush=True); time.sleep(0.1)
    print("Worker: Stoppar.", flush=True)

# --- Ljudinspelningsfunktioner ---
def audio_callback(indata, frames, time, status):
    global audio_buffer, chunk_queue, chunk_counter
    if status: print(f"Ljudfel: {status}", flush=True)
    with recording_lock:
        if is_recording:
            audio_buffer.append(indata.copy())
            current_samples = sum(len(block) for block in audio_buffer)
            if current_samples >= CHUNK_SAMPLES:
                chunk_np = np.concatenate(audio_buffer, axis=0)
                chunk_queue.put((chunk_counter, chunk_np))
                chunk_counter += 1; audio_buffer.clear() 

def start_recording():
    global is_recording, audio_buffer, chunk_queue, results, chunk_counter, worker_thread, stop_event, stream, model
    if model is None: print("ERROR: Modellen ej laddad!", flush=True); return
    with recording_lock:
        if is_recording: return 
        print("-----------------------------------------------------")
        print(f"Startar inspelning (chunk size: {CHUNK_SECONDS}s)... (Tryck F9 igen för att stoppa)", flush=True)
        audio_buffer = []; results = {}; chunk_counter = 0
        while not chunk_queue.empty():
            try: chunk_queue.get_nowait(); chunk_queue.task_done()
            except queue.Empty: break
        stop_event.clear(); exit_event.clear(); is_recording = True
        if worker_thread is None or not worker_thread.is_alive():
            print("Startar transkriberings-worker...", flush=True); stop_event.clear() 
            worker_thread = threading.Thread(target=transcription_worker, daemon=True); worker_thread.start()
    try:
        stream = sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        stream.start()
        print(">>> Mikrofon aktiv (bearbetar löpande) <<<", flush=True)
        update_tray_icon(recording_icon_image) # Uppdatera ikon
    except Exception as e:
        print(f"ERROR: Kunde inte starta ljudström: {e}", flush=True)
        with recording_lock: is_recording = False; stop_event.set() 
        update_tray_icon(error_icon_image if error_icon_image else idle_icon_image) 

# --- Stopp- och Finaliseringsfunktion ---
def stop_recording_and_finalize():
    global is_recording, stream, audio_buffer, chunk_queue, chunk_counter, results, stop_event, worker_thread
    print("Förbereder att stoppa och finalisera...", flush=True)
    update_tray_icon(processing_icon_image if processing_icon_image else idle_icon_image) # Sätt bearbetningsikon
    final_chunk_data = None
    with recording_lock:
        if not is_recording: 
            print("Redan stoppad.", flush=True); update_tray_icon(idle_icon_image); return
        print("Stoppar ljudinsamling...", flush=True); is_recording = False 
        if audio_buffer:
            print(f"Hanterar sista chunk ({sum(len(block) for block in audio_buffer)} samples)...", flush=True)
            final_chunk_data = np.concatenate(audio_buffer, axis=0); audio_buffer.clear()
        else: print("Ingen sista ljudbit att hantera.", flush=True)
    if stream:
        try: stream.stop(); stream.close(); print("Mikrofon stängd.", flush=True)
        except Exception as e: print(f"ERROR: Fel vid stängning av ljudström: {e}", flush=True)
        stream = None 
    if final_chunk_data is not None and len(final_chunk_data) > 0:
        print(f"Skickar sista chunk {chunk_counter} ({len(final_chunk_data)} samples) till kön.", flush=True)
        chunk_queue.put((chunk_counter, final_chunk_data))
        with recording_lock: chunk_counter += 1 
    print("Väntar på att all transkribering ska slutföras (queue.join())...", flush=True)
    chunk_queue.join(); print("Transkriberingskön är tom.", flush=True)
    print("Signalerar till worker att stoppa...", flush=True); stop_event.set() 
    if worker_thread and worker_thread.is_alive():
       print("Väntar på att worker-tråden ska avslutas...", flush=True); worker_thread.join(timeout=2.0) 
       if worker_thread.is_alive(): print("VARNING: Worker-tråden avslutades inte.", flush=True)
       else: print("Worker-tråden har avslutats.", flush=True)
    print("\n--- Sammanställer Resultat ---", flush=True); final_text = ""
    with result_lock:
        sorted_indices = sorted(results.keys())
        text_pieces = [results[i] for i in sorted_indices]
        final_text = " ".join(filter(None, text_pieces)) 
    print("------------------------------", flush=True); print(f"Slutgiltigt resultat: {final_text}", flush=True)
    if final_text:
        try:
            print("Kopierar text till urklipp...", flush=True); pyperclip.copy(final_text + " ") 
            time.sleep(0.1); print("Simulerar Ctrl+V (Klistra in)...", flush=True)
            keyboard_controller.press(Key.ctrl_l); keyboard_controller.press('v')
            keyboard_controller.release('v'); keyboard_controller.release(Key.ctrl_l)
            time.sleep(0.1); print("-> Text infogad via urklipp.", flush=True)
        except Exception as e: print(f"ERROR: Urklipp/Klistra in misslyckades: {e}", flush=True)
    else: print("-> Ingen text att infoga.", flush=True)
    print("-----------------------------------------------------")
    with result_lock: 
        results.clear() # Korrigerad rad
    update_tray_icon(idle_icon_image) # Återställ ikon

# --- Kortkommandohantering ---
def on_press(key):
    global is_recording 
    if key == START_STOP_HOTKEY_KEY:
        with recording_lock: rec_status_copy = is_recording 
        if rec_status_copy: stop_recording_and_finalize()
        else: start_recording()

def on_release(key): pass

# --- Funktion för att köra ikonen ---
def run_tray_icon(icon):
    try: icon.run()
    except Exception as e: print(f"ERROR i ikontråden: {e}", flush=True)

# --- Funktion för menyvalet "Avsluta" ---
def exit_action(icon, item):
    print("Avslutar via menyval...", flush=True)
    exit_event.set(); stop_event.set()
    if listener and listener.is_alive(): listener.stop()
    if icon: icon.stop() 

# --- Signalhanterare ---
def signal_handler(sig, frame):
    print("\nCtrl+C upptäckt. Förbereder avslutning...", flush=True)
    exit_event.set(); stop_event.set() 
    if listener and listener.is_alive(): listener.stop()
    if tray_icon: tray_icon.stop()

# --- Huvudprogram ---
if __name__ == "__main__":
    print("="*40); print("   KB-Whisper Hotkey Transcriber (Tray Icon v4 - Fixed Syntax)"); print("="*40)
    print(f"Använder modell: {MODEL_NAME}")
    print(f"Laddar modellen...", flush=True)
    try:
        load_start_time = time.time(); model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE); load_end_time = time.time()
        print(f"Modell laddad på {load_end_time - load_start_time:.2f} sekunder.", flush=True)
    except Exception as e: print(f"\nFATAL ERROR: Kunde inte ladda modellen ({MODEL_NAME}): {e}", flush=True); print(traceback.format_exc(), flush=True); sys.exit(1) 

    # Ladda ikonbilder
    if PIL_AVAILABLE:
        try: 
            idle_icon_image = Image.open(ICON_IDLE)
            print(f"Laddade ikon: {ICON_IDLE}")
        except Exception as e: 
            print(f"VARNING: Kunde inte ladda {ICON_IDLE}: {e}")
            
        try: 
            recording_icon_image = Image.open(ICON_RECORDING)
            print(f"Laddade ikon: {ICON_RECORDING}")
        except Exception as e: 
            print(f"VARNING: Kunde inte ladda {ICON_RECORDING}: {e}")
            
        # ----> KORRIGERAD SYNTAX FÖR VALFRIA IKONER <----
        try: 
            processing_icon_image = Image.open(ICON_PROCESSING) # Valfri
            print(f"Laddade ikon: {ICON_PROCESSING}") # Skriv ut om den lyckas
        except FileNotFoundError: 
            pass # Ignorera tyst om filen inte finns
        except Exception as e: 
            print(f"VARNING: Kunde inte ladda {ICON_PROCESSING}: {e}")

        try: 
            error_icon_image = Image.open(ICON_ERROR) # Valfri
            print(f"Laddade ikon: {ICON_ERROR}") # Skriv ut om den lyckas
        except FileNotFoundError: 
            pass # Ignorera tyst om filen inte finns
        except Exception as e: 
            print(f"VARNING: Kunde inte ladda {ICON_ERROR}: {e}")
        # -------------------------------------------------
            
        if idle_icon_image:
             tray_icon = pystray.Icon("Transcriber", idle_icon_image, "KB Whisper (F9)", menu=pystray.Menu(pystray.MenuItem('Avsluta', exit_action)))
        else: 
             print("VARNING: Kan ej skapa ikon, idle-ikon saknas.")
    
    print(f"\nChunk-storlek: {CHUNK_SECONDS}s (Inget överlapp)") 
    print(f"Kortkommando: {START_STOP_HOTKEY_KEY} (F9)") 
    print("Instruktioner:\n 1. Använd F9 start/stopp.\n 2. Högerklicka ikon för Avsluta.\n 3. Text infogas via Ctrl+V.\n")
    print("Programmet är aktivt. Tryck Ctrl+C för att avsluta.\n" + "-" * 40)
    
    signal.signal(signal.SIGINT, signal_handler)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release) 
    listener.start()

    icon_thread = None
    if tray_icon:
        print("Startar systemfältsikon...")
        icon_thread = threading.Thread(target=run_tray_icon, args=(tray_icon,), daemon=True)
        icon_thread.start()

    # Huvudloop
    try:
        while not exit_event.is_set(): 
            if not listener.is_alive(): print("ERROR: Lyssnartråden avslutats.", flush=True); exit_event.set() 
            exit_event.wait(timeout=1.0) 
            
    except Exception as e: 
        print(f"\nERROR: Oväntat fel i huvudloopen: {e}", flush=True); print(traceback.format_exc(), flush=True); exit_event.set(); stop_event.set() 
        
    finally:
        # ----- Städning vid avslut -----
        print("Påbörjar städning...", flush=True); stop_event.set() 
        
        # Stoppa Ikon FÖRST
        if tray_icon:
            print("Stoppar systemfältsikon...", flush=True)
            tray_icon.stop() # Be ikonen stoppa
            if icon_thread and icon_thread.is_alive():
                 icon_thread.join(timeout=1.0) 
                 if icon_thread.is_alive(): print("VARNING: Ikontråden avslutades ej korrekt.", flush=True)

        # Vänta på worker
        if worker_thread and worker_thread.is_alive():
           print("Väntar på worker...", flush=True)
           # Försök väcka workern om den väntar på kön
           try: 
               chunk_queue.put_nowait((None, None)) # Skicka None för att låsa upp get()
           except queue.Full: 
               pass # Om kön är full är det ok
           # Vänta på att tråden avslutas
           worker_thread.join(timeout=2.0) 
           if worker_thread.is_alive(): 
               print("VARNING: Worker avslutades ej korrekt.", flush=True)
               
        # Vänta på listener
        if listener and listener.is_alive():
           listener.stop()
           listener.join(timeout=1.0) 
           if listener.is_alive(): 
               print("VARNING: Lyssnaren avslutades ej korrekt.", flush=True)
               
        # Stäng ljudström
        if stream and not stream.closed:
           print("Stoppar stream...", flush=True) 
           try: 
               stream.stop()
               stream.close()
           except Exception as e: 
               print(f"ERROR stängning stream: {e}", flush=True)
               
        # Städa temporära filer
        print("Städar temp filer...", flush=True)
        cleaned_count = 0
        try:
            current_dir = '.' 
            for filename in os.listdir(current_dir): 
                if filename.startswith(TEMP_FILE_PREFIX) and filename.endswith(".wav"):
                    file_path = os.path.join(current_dir, filename)
                    try: 
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e: 
                        print(f"ERROR borttagning '{filename}': {e}", flush=True)
            if cleaned_count > 0: 
                print(f"Tog bort {cleaned_count} temp filer.")
        except Exception as e: 
            print(f"ERROR listning/städning filer: {e}", flush=True)
            
        # Slutmeddelande
        print("Programmet har avslutats.")
        sys.stdout.flush() # Sista flush