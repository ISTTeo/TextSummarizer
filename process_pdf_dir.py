import os
import json
import PyPDF2
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

def extract_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """
    Extract metadata and text content from a PDF file.
    
    :param pdf_path: Path to the PDF file
    :return: Dictionary containing file path, metadata, and text content
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        
        # Extract metadata
        metadata = reader.metadata
        if metadata is not None:
            metadata = {key: str(value) for key, value in metadata.items()}
        else:
            metadata = {}
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return {
            "file_path": pdf_path,
            "metadata": metadata,
            "text": text.strip()
        }

def process_pdf_directory(directory: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Process all PDF files in the given directory and return a list of dictionaries
    containing file information.
    
    :param directory: Path to the directory containing PDF files
    :return: List of dictionaries with PDF information
    """
    pdf_info_list = []
    
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        pdf_path = os.path.join(directory, filename)
        try:
            pdf_info = extract_pdf_info(pdf_path)
            pdf_info_list.append(pdf_info)
            save_to_json(pdf_info_list, output_file)
            tqdm.write(f"Processed {filename} - OK")
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {str(e)}")
    
    return pdf_info_list

def save_to_json(data: List[Dict[str, Any]], output_file: str):
    """
    Save the list of dictionaries to a JSON file.
    
    :param data: List of dictionaries to save
    :param output_file: Path to the output JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process artifacts using GPU acceleration.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--pdf_dir", 
                        required=True, 
                        help="Path to the pdfs")
    parser.add_argument("--output_json", 
                        required=True, 
                        help="Path to the output json")

    args = parser.parse_args()
    

    # Specify the directory containing PDF files
    pdf_directory = args.pdf_dir
    
    # Specify the output JSON file path
    output_json_file = args.output_json
    
    # Process PDF files and get the list of dictionaries
    pdf_info_list = process_pdf_directory(pdf_directory, output_json_file)
        
    print(f"Processed {len(pdf_info_list)} PDF files.")
    print(f"JSON data saved to {output_json_file}")