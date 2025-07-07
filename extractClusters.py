import os
import pypdf
import logging
import pandas as pd
import google.generativeai as genai

from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

def extract_abstract_pypdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        
        for page_num in range(min(3, len(reader.pages))):
            page = reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else "" 

        
        abstract_keywords = ["abstract", "summary", "introduction", "keywords", "1. introduction"]
        text_lower = text.lower()
        abstract_start_idx = -1

        for keyword in abstract_keywords:
            idx = text_lower.find(keyword)
            if idx != -1:
                if keyword == "abstract" or keyword == "summary":
                    abstract_start_idx = idx
                    break
                elif keyword == "introduction" or keyword == "1. introduction":
                    if text_lower.find("abstract", 0, idx) != -1:
                        abstract_start_idx = text_lower.find("abstract", 0, idx)
                    else:
                        abstract_start_idx = 0
                    break
                elif keyword == "keywords":
                    if text_lower.find("abstract", 0, idx) != -1:
                        abstract_start_idx = text_lower.find("abstract", 0, idx)
                    else:
                        abstract_start_idx = 0
                    break

        if abstract_start_idx != -1:
            potential_ends = ["keywords", "1. introduction", "i. introduction", "sections"]
            end_idx = len(text)
            for end_keyword in potential_ends:
                current_end_idx = text_lower.find(end_keyword, abstract_start_idx)
                if current_end_idx != -1:
                    end_idx = min(end_idx, current_end_idx)

            abstract = text[abstract_start_idx:end_idx].strip()

            for keyword in ["Abstract", "ABSTRACT", "Summary", "SUMMARY"]:
                if abstract.startswith(keyword):
                    abstract = abstract[len(keyword):].strip()
                    break

            word_count = len(abstract.split())
            if 50 <= word_count <= 500:
                return abstract
            else:
                logging.warning(f"Abstract for {os.path.basename(pdf_path)} seems too short or too long (word count: {word_count}). Skipping")
                return None
        else:
            logging.warning(f"Could not find clear abstract section in {os.path.basename(pdf_path)}. Attempting to extract first significant paragraph")
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            if paragraphs:
                first_paragraph_words = len(paragraphs[0].split())
                if 50 <= first_paragraph_words <= 300:
                    return paragraphs[0]
            logging.warning(f"No suitable abstract found for {os.path.basename(pdf_path)}. Skipping")
            return None

    except Exception as e:
        logging.error(f"Error extracting abstract from {os.path.basename(pdf_path)}: {e}")
        return None
    
def find_optimal_clusters(embeddings, max_k=10):

    if len(embeddings) < 2:
        return 1

    scores = []
    K = range(2, min(max_k + 1, len(embeddings)))

    if not K:
        logging.warning("Not enough samples to perform K-means clustering with K >= 2. Returning K=1")
        return 1

    for k in tqdm(K, desc="Finding optimal K"):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(embeddings)
            if len(set(labels)) > 1:
                score = silhouette_score(embeddings, labels)
                scores.append((score, k))
        except Exception as e:
            logging.warning(f"Could not compute silhouette score for k={k}: {e}")
            continue

    if scores:
        best_k = max(scores, key=lambda item: item[0])[1]
        logging.info(f"Optimal number of clusters (KMeans Silhouette): {best_k}")
        return best_k
    else:
        logging.warning("Could not determine optimal K. Defaulting to 3 clusters")
        return min(3, len(embeddings) -1) if len(embeddings) > 1 else 1
    
def generate_cluster_name(abstracts_sample, model):

    prompt = f"""
    Based on the following research paper abstracts, suggest 1  very short, descriptive topic names (e.g., "Deep Learning", "Graph Neural Networks", "Computational Fluid Dynamics").
    Return only the name.

    Abstracts:
    {'- ' + ' - '.join(abstracts_sample)}
    """
    try:
        response = model.generate_content(prompt)
        cluster_name = response.text.strip().replace('"', '')
        logging.info(f"Generated cluster name: {cluster_name}")
        return cluster_name
    except Exception as e:
        logging.error(f"Error generating cluster name with Gemini: {e}")
        return "Unknown Topic"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
    OUTPUT_CSV_PATH = "data/paper_data.csv"
    MIN_PDFS_REQUIRED = 5

    MODEL_NAME = "gemini-2.5-flash-preview-05-20"

    load_dotenv()
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)

    os.makedirs("data", exist_ok=True)

    pdf_files = [f for f in os.listdir(DESKTOP_PATH) if f.lower().endswith(".pdf")]
    full_pdf_paths = [os.path.join(DESKTOP_PATH, f) for f in pdf_files]

    if len(full_pdf_paths) < MIN_PDFS_REQUIRED:
        logging.info(f"Found only {len(full_pdf_paths)} PDFs. Skipping processing as less than {MIN_PDFS_REQUIRED} are required")
        pd.DataFrame(columns=['fileName', 'abstract', 'clusterID', 'clusterName']).to_csv(OUTPUT_CSV_PATH, index=False)
        exit()
    else:
        logging.info(f"Found {len(full_pdf_paths)} PDF files on Desktop")
        
        # Load SentenceTransformer model
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer model: {e}")
            exit()
        
        # Load Gemini model
        try:
            gemini_model = genai.GenerativeModel(MODEL_NAME)
            logging.info(f"Gemini model '{MODEL_NAME}' initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {e}")
            exit()
        
    # Extract Abstracts
    paper_data = []
    for pdf_path in tqdm(full_pdf_paths, desc="Extracting abstracts"):
        abstract = extract_abstract_pypdf(pdf_path)
        if abstract:
            paper_data.append({"fileName": os.path.basename(pdf_path), "abstract": abstract})
        else:
            logging.warning(f"Skipping {os.path.basename(pdf_path)} due to missing or unreadable abstract")

    if not paper_data:
        logging.info("No abstracts could be extracted. Exiting")
        pd.DataFrame(columns=['fileName', 'abstract', 'clusterID', 'clusterName']).to_csv(OUTPUT_CSV_PATH, index=False)
        exit()

    df = pd.DataFrame(paper_data)
    logging.info(f"Successfully extracted {len(df)} abstracts.")

    # Vectorize Abstracts
    logging.info("Vectorizing abstracts...")
    abstract_embeddings = model.encode(df['abstract'].tolist())
    logging.info("Abstracts vectorized.")

    # Cluster Abstracts
    logging.info("Clustering abstracts...")
    num_clusters = find_optimal_clusters(abstract_embeddings, max_k=min(10, len(df) -1) if len(df) > 1 else 1)
        
    if num_clusters <= 1:
        logging.info("Not enough data to form multiple clusters or optimal K is 1. All papers will be in a single cluster")
        df['clusterID'] = 0
        df['clusterName'] = generate_cluster_name(df['abstract'].sample(min(3, len(df))).tolist(), gemini_model)
    else:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
        df['clusterID'] = kmeans.fit_predict(abstract_embeddings)
        logging.info(f"Abstracts clustered into {num_clusters} clusters")

        # Generate Cluster Names
        logging.info("Generating cluster names...")
        cluster_names = {}
        for cluster_id in sorted(df['clusterID'].unique()):
            cluster_abstracts = df[df['clusterID'] == cluster_id]['abstract'].tolist()
            sample_abstracts = cluster_abstracts[:min(3, len(cluster_abstracts))]
            name = generate_cluster_name(sample_abstracts, gemini_model)
            cluster_names[cluster_id] = name
        df['clusterName'] = df['clusterID'].map(cluster_names)
        logging.info("Cluster names generated")

    # Save Results
    logging.info(f"Saving results to {OUTPUT_CSV_PATH}...")
    df[['fileName', 'abstract', 'clusterID', 'clusterName']].to_csv(OUTPUT_CSV_PATH, index=False)
    logging.info("Processing complete for 1-extractClusters.py")
    
if __name__ == "__main__":
    main()