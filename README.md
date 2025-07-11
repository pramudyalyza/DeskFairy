# Desk Fairy 🧚🏻‍♀️

**Desk Fairy** is a personal, magical assistant that swoops in to transform your chaotic desktop full of research papers into beautifully organized, topic-specific folders *all while you sleep*! No more endless manual sorting. just wake up to a perfectly tidy digital workspace.

---

### How It Works
1. Scans for any new PDFs on my desktop (only proceeds if there are at least 5 PDFs found).
2. Reads the abstract from each paper to understand its content.
3. Clusters the papers into groups using a bit of NLP and machine learning.
4. Asks an LLM (either Google Gemini or a local Ollama model) to give each cluster a short, descriptive name (e.g., "Generative AI").
5. Moves the files into neatly named folders.

This entire process runs automatically every night using Windows Task Scheduler.

---

### Tech Stack

-   **Core Logic:** Python
-   **PDF Parsing:** `pypdf`
-   **Embeddings & ML Clustering:** `sentence-transformers`, `scikit-learn` (K-Means)
-   **Topic Naming Options:**
    *   **Cloud LLM:** Google Gemini API
    *   **Local LLM:** Local LLM via 'Ollama' (e.g., Llama 3.2, Mistral, Deepseek R1)
-   **Scheduling:** Windows Task Scheduler

---

### Getting Started

Desk Fairy offers flexibility with its LLM backend. You can choose to use the Google Gemini API (requiring an API key) or run a Large Language Model locally using Ollama (requiring local setup and a downloaded model).

**Choose your preferred setup below:**

---

### Option 1: Using Google Gemini (Cloud LLM)

This option utilizes the Google Gemini API for generating cluster names.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/DeskFairy.git
    ```
2.  **Navigate to the Project Folder:**
    ```bash
    cd DeskFairy
    ```
3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
4.  **Activate the Virtual Environment:**
    ```bash
    venv\Scripts\activate
    ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Set Up Your Gemini API Key:**
    Create your .env file just like .env.example and paste your Gemini API Key
    ```
    GEMINI_API_KEY=your_api_key_here
    ```
7.  **Run the Script Manually:**
    To test if everything is working, place some PDF files on your desktop and run:
    ```bash
    python main.py
    ```

    **Important:** In your `main.py` script, find this line `from extractClusters import main as extract_clusters_main` and change it to `from extractClusters-gemini import main as extract_clusters_main`

### Option 2: Using a Local LLM with Ollama

This option runs an LLM model directly on your machine using Ollama so its free.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YourUsername/DeskFairy.git
    ```
2.  **Navigate to the Project Folder:**
    ```bash
    cd DeskFairy
    ```
3.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
4.  **Activate the Virtual Environment:**
    ```bash
    venv\Scripts\activate
    ```
5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Install and Configure Ollama:**
    *   **Download & Install Ollama:** Visit [ollama.com](https://ollama.com/) and download the Ollama application for Windows. Install it on your system.
    *   **Start Ollama Server:** Ensure the Ollama application is running in your system, or manually start the server by opening a command prompt and typing:
        ```bash
        ollama serve
        ```
    *   **Pull an LLM Model:** Open a command prompt and pull the desired LLM model. For example, here i use Llama 3.2
        ```bash
        ollama pull llama3.2
        ```

        **Important:** In your 'extractClusters.py' script, find the 'LLM_MODEL' variable and set its value to the exact name of the model you pulled (e.g., "llama3.2" or "deepseek-r1:7b").

7.  **Run the Script Manually:**
    To test if everything is working, place some PDF files on your desktop and run:
    ```bash
    python main.py
    ```

## Schedule with Windows Task Scheduler:
*   Open "Task Scheduler" on Windows.
*   Go to "Action" -> "Create Task..."
*   **Name:** anything you like
*   **Trigger:** Choose "Daily" and set your preferred time (e.g., 11:59 PM)
*   **Actions:** "Start a program"
*   **Program/script:** Copy path to your python.exe (e.g., D:\Alyza\Projectz\DeskFairy\venv\Scripts\python.exe)
*   **Add arguments:** main.py
*   **Start in (optional):** Copy path to the project folder (e.g., D:\Alyza\Projectz\DeskFairy)