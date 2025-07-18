INSTALL_GUIDE
--- START OF FILE INSTALL_GUIDE.txt ---
Generated text
============================================================
 INSTALLATION AND USAGE GUIDE for GGTH EA (v1.83+)
============================================================

This guide explains how to set up and run the complete GGTH EA system,
which uses a Python backend for machine learning predictions.

---------------------------------
 SECTION 1: PREREQUISITES
---------------------------------

Before you begin, you MUST have the following software installed on your Windows machine:

1.  **MetaTrader 5 Terminal:** The trading platform itself.

2.  **Python:** Version 3.10 or newer is recommended.
    *   **IMPORTANT:** During installation, make sure to check the box that says "Add Python to PATH".

3.  **Required Python Libraries:** Open a Administrator Command Prompt (cmd) and run the following command to install all necessary libraries at once:

    pip install torch pandas numpy scikit-learn joblib pykalman

---------------------------------
 SECTION 2: FILE PLACEMENT
---------------------------------

All file paths are relative to your MetaTrader 5 Data Folder. To find it, open MetaTrader 5 and go to "File" -> "Open Data Folder".
Its easiest to get the path if you copy it directly from windows explorer example C:\Users\USERNAME\AppData\Roaming\MetaQuotes\Terminal\LONG-STRING-OF-NUMBERS-AND-LETTERS\MQL5
Create a folder named "LSTM_Trading" in the Files Folder
[Your MQL5 Data Folder]
|
|--- MQL5/
|    |
|    |--- Experts/
|    |    |
|    |    +--- GGTH.mq5             <-- The main EA source code file goes here.
|    |         (GGTH.ex5 will be created here after compiling)
|    |
|    |--- Files/
|         |
|         +--- LSTM_Trading/           <-- This is your MAIN project folder.
|              |
|              |--- data_processing.py
|              |--- train_combined_model.py
|              |--- retrain_manager.py
|              |--- daemon.py
|              |--- generate_backtest_data.py
|              |
|              |--- EURUSD60.csv        <-- Your historical data files go here.
|              |--- EURJPY60.csv
|              |--- USDJPY60.csv
|              |--- (and so on for all required pairs)
|              |
|              |--- models/             <-- This folder will be CREATED AUTOMATICALLY.
|              |
|              +--- data/               <-- This folder will be CREATED AUTOMATICALLY.
|
|--- Common/
     |
     +--- Files/
          |
          +--- backtest_predictions.csv  <-- The backtest file MUST be MANUALLY MOVED here.


---------------------------------
 SECTION 3: WORKFLOW
---------------------------------

Follow these workflows for training, backtesting, and live trading.

---
>>> PHASE 1: INITIAL SETUP & TRAINING (Do this first)
---

This process trains the AI model with your historical data.

1.  **Place Historical Data:** Ensure all your `.csv` history files (EURUSD60.csv, etc.) are inside the `MQL5\Files\LSTM_Trading\` folder.

2.  **Train the Models**
    *   Open a Command Prompt (cmd).
    *   Navigate to your project directory:
        cd C:\Path\To\Your\Data\Folder\MQL5\Files\LSTM_Trading
    *   python train_combined_model.py
    *   This will take several minutes. It will automatically:
        a) Train the new combined model.
        b) Save the model files (`.pth` and `.pkl`) into the `models` subfolder.
        c) Start the `daemon.py` script in the background.

---
>>> PHASE 2: HOW TO BACKTEST
---

The Strategy Tester requires a special data file and CANNOT have the daemon running.

1.  **IMPORTANT: STOP THE DAEMON:** If the `daemon.py` script is running from Phase 1, close its command prompt window. The backtester will conflict with it.

2.  **Generate Backtest Data:**
    *   Open a Command Prompt.
    *   Navigate to your project directory.
    *   Run the generation script, specifying the symbol:
        python generate_backtest_data.py EURUSD
    *   This will create a new `backtest_predictions.csv` file inside your `LSTM_Trading` folder.

3.  **MOVE THE FILE:**
    *   Go to your `MQL5\Files\LSTM_Trading\` folder.
    *   **CUT** the `backtest_predictions.csv` file.
    *   Navigate to the `MQL5\Common\Files\` folder.
    *   **PASTE** the file there. This is where the EA looks for it during backtests.

4.  **Run the Backtest in MetaTrader:**
    *   Open the Strategy Tester (Ctrl+R).
    *  ndbox, and paste/select your `backtest_predictions.csv` file there.
    *   Set your desired date range and click "Start".

---
>>> PHASE 3: HOW TO LIVE/DEMO TRADE
---

Live trading requires the daemon to be running constantly.

1.  **Start the Python Daemon:**
    *   Open a Command Prompt.
    *   Navigate to your project directory.
    *   Run the daemon script:
        python daemon.py
    *   You will see the message `--- Advanced Combined LSTM Daemon is running. ---`
    *   **IMPORTANT: KEEP THIS COMMAND PROMPT WINDOW OPEN!** This is the brain of the EA. If you close it, the EA will stop getting predictions.

2.  **Attach the EA to a Chart:**
    *   In MetaTrader 5, open a EURUSD, H1 chart.
    *   Drag the `GGTH` EA onto the chart.
    *   Ensure "Algo Trading" is enabled. can copy the text below and save it as `README.txt` in your project folder.

--- START OF FILE README.txt ---
