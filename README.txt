# KB-Whisper Hotkey Transcriber (Svenska)

Ett Python-skript för Windows som låter dig snabbt transkribera tal till text lokalt via ett kortkommando (F9) med hjälp av KBLab:s svenska Whisper-modell.

> **OBS!** Detta projekt släpps "as is" (som det är) utan aktivt underhåll eller support. Du är välkommen att använda, modifiera och distribuera koden enligt MIT-licensen, men inga garantier ges för funktionalitet eller säkerhet.

## Funktioner

*   **Globalt Kortkommando:** Använd **F9** för att starta/stoppa inspelning.
*   **Lokal & Privat:** Transkribering sker 100% lokalt med [KBLab/kb-whisper-base](https://huggingface.co/KBLab/kb-whisper-base) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Ingen data lämnar din dator.
*   **Chunking:** Långa inspelningar delas upp (20s chunks) för snabbare respons vid stopp.
*   **Systemfältsikon:** Visar aktuell status (väntar, spelar in, bearbetar).
*   **Automatisk Inklistring:** Resultatet kopieras till urklipp och klistras in (Ctrl+V).

## Förutsättningar

*   Windows (10/11)
*   [Python 3.8+](https://www.python.org/downloads/) (med PATH-tillgång)
*   [FFmpeg](https://ffmpeg.org/download.html) (installerat och i systemets PATH)
    *   *Tips: `winget install Gyan.FFmpeg` eller `choco install ffmpeg` (kör som admin)*
    *   *Verifiera med `ffmpeg -version` i terminalen efter installation/omstart.*
*   Mikrofon

## Installation

1.  **Klona eller ladda ner** detta repo.
2.  **Öppna terminal** i projektmappen.
3.  **Installera beroenden:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Du kan behöva uppgradera pip först: `python -m pip install --upgrade pip`)*

## Användning

1.  **Kör skriptet (som administratör):**
    ```bash
    python transcribe_hotkey_kb.py 
    ```
    *(Första körningen laddar ner modellen - ca 150MB).*
2.  **Tryck F9** för att starta inspelning (ikonen ändras).
3.  **Prata.**
4.  **Tryck F9** igen för att stoppa.
5.  Texten klistras in där markören är.
6.  **Avsluta:** Högerklicka på systemfältsikonen och välj "Avsluta", eller tryck Ctrl+C i terminalen.

## Konfiguration (Valfritt)

Redigera Python-skriptfilen för att ändra:
*   `ICON_IDLE`, `ICON_RECORDING`, `ICON_PROCESSING`: Filnamn för ikoner.
*   `START_STOP_HOTKEY_KEY`: Ändra kortkommandot (standard `keyboard.Key.f9`).
*   `CHUNK_SECONDS`: Ändra längden på ljudchunks (standard `20`).

## Licens

[MIT](LICENSE)

## Tack till

*   [KBLab](https://github.com/KBLab), [OpenAI](https://openai.com/), [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) och utvecklarna av de använda Python-biblioteken.