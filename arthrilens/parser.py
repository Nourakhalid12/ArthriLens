import os
from typing import List, Dict, Any
import pandas as pd
import pypdf

class DocumentParser:
    """
    A unified document parser that extracts text and metadata from PDFs and Excel files.
    """

    @staticmethod
    def parse_pdf(file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a PDF file page by page.
        Returns a list of dicts: [{'text': str, 'metadata': dict}]
        """
        documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        try:
            reader = pypdf.PdfReader(file_path)
            file_name = os.path.basename(file_path)
            
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": file_path,
                            "file_name": file_name,
                            "file_type": "pdf",
                            "page": page_idx + 1
                        }
                    })
        except Exception as e:
            print(f"Error parsing PDF file {file_path}: {e}")
            
        return documents

    @staticmethod
    def parse_excel(file_path: str) -> List[Dict[str, Any]]:
        """
        Parses an Excel file sheet by sheet, converting each row into a text representation.
        Returns a list of dicts: [{'text': str, 'metadata': dict}]
        """
        documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")

        try:
            excel_file = pd.ExcelFile(file_path)
            file_name = os.path.basename(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                # Fill NaN with empty string to avoid formatting issues
                df = df.fillna("")
                
                for row_idx, row in df.iterrows():
                    # Format row as a descriptive text string: "ColumnName: Value | ..."
                    row_parts = []
                    for col in df.columns:
                        val = str(row[col]).strip()
                        if val:
                            row_parts.append(f"{col}: {val}")
                    
                    row_text = " | ".join(row_parts)
                    if row_text.strip():
                        documents.append({
                            "text": row_text.strip(),
                            "metadata": {
                                "source": file_path,
                                "file_name": file_name,
                                "file_type": "xlsx",
                                "sheet": sheet_name,
                                "row": int(row_idx) + 1
                            }
                        })
        except Exception as e:
            print(f"Error parsing Excel file {file_path}: {e}")
            
        return documents

    @staticmethod
    def parse_txt(file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a text file.
        Returns a list containing a single dict with file contents.
        """
        documents = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TXT file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text.strip():
                documents.append({
                    "text": text.strip(),
                    "metadata": {
                        "source": file_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": "txt"
                    }
                })
        except Exception as e:
            print(f"Error parsing TXT file {file_path}: {e}")
            
        return documents

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a file based on its extension. Supported extensions: .pdf, .xlsx, .xls, .txt
        """
        ext = os.path.splitext(file_path.lower())[1]
        if ext == ".pdf":
            return self.parse_pdf(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self.parse_excel(file_path)
        elif ext == ".txt":
            return self.parse_txt(file_path)
        else:
            print(f"Unsupported file format: {ext} for file {file_path}")
            return []

    def parse_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Recursively scans a directory and parses all supported files.
        """
        all_documents = []
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                docs = self.parse_file(file_path)
                if docs:
                    all_documents.extend(docs)
        return all_documents
