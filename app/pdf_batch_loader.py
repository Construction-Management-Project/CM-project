import os
import shutil
from glob import glob
from time import sleep
from tqdm import tqdm
from app.pdf_loader import load_pdfs_and_create_vectorstore
from app.config import DATA_DIR, CHROMA_DIR

BATCH_SIZE = 10
ALL_PDFS = sorted(glob(os.path.join(DATA_DIR, "*.pdf")))

def reset_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def split_and_process_batches():
    total_batches = (len(ALL_PDFS) - 1) // BATCH_SIZE + 1
    original_data_dir = DATA_DIR
    original_chroma_dir = CHROMA_DIR

    for batch_num in range(total_batches):
        print(f"\nProcessing batch {batch_num + 1} / {total_batches}...")

        # Prepare folder for this batch
        temp_data_dir = os.path.join("data", f"temp_batch_{batch_num + 1}")
        reset_folder(temp_data_dir)

        # Copy PDF files for the batch
        batch_pdfs = ALL_PDFS[batch_num * BATCH_SIZE:(batch_num + 1) * BATCH_SIZE]
        for pdf in batch_pdfs:
            shutil.copy(pdf, temp_data_dir)

        # Initialize Chroma DB folder for the batch
        temp_chroma_dir = os.path.join("db", f"chroma_batch_{batch_num + 1}")
        reset_folder(temp_chroma_dir)

        # Temporarily override environment variables
        os.environ["DATA_DIR"] = temp_data_dir
        os.environ["CHROMA_DIR"] = temp_chroma_dir

        # Run vectorization
        try:
            load_pdfs_and_create_vectorstore()
        except Exception as e:
            print(f"Failed to process batch {batch_num + 1}: {e}")
        sleep(2)

if __name__ == "__main__":
    print("Starting batch vectorization of all PDFs...")
    split_and_process_batches()
    print("All batches completed!")
