from typing import List, Dict, Any

class RecursiveCharacterTextSplitter:
    """
    Splits text into chunks recursively using a list of separators.
    Attempts to split on paragraph (\n\n), sentence (\n), word (space), and finally character-by-character.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively splits text using separators until chunks are within chunk_size.
        """
        final_chunks = []
        
        # Determine the active separator
        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else []
        
        # Split text by active separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)  # Character split
            
        good_splits = []
        for s in splits:
            if len(s) <= self.chunk_size:
                good_splits.append(s)
            else:
                # If a sub-split is still too big, recurse
                if good_splits:
                    final_chunks.extend(self._merge_splits(good_splits, separator))
                    good_splits = []
                if next_separators:
                    rec_splits = self._split_text(s, next_separators)
                    final_chunks.extend(rec_splits)
                else:
                    # No more separators, just force slice it
                    for i in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(s[i:i + self.chunk_size])
        
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, separator))
            
        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """
        Merges splits back together into chunks that are under chunk_size, respecting overlap.
        """
        chunks = []
        current_doc = []
        current_len = 0
        
        for d in splits:
            d_len = len(d)
            # Length including the separator
            sep_len = len(separator) if current_doc else 0
            
            if current_len + d_len + sep_len > self.chunk_size:
                if current_doc:
                    # Save current chunk
                    chunk_text = separator.join(current_doc)
                    chunks.append(chunk_text)
                    
                    # Prepare next chunk with overlap
                    # We keep taking elements from current_doc from the end until they fit the overlap budget
                    overlap_doc = []
                    overlap_len = 0
                    for prev_d in reversed(current_doc):
                        prev_len = len(prev_d)
                        prev_sep_len = len(separator) if overlap_doc else 0
                        if overlap_len + prev_len + prev_sep_len <= self.chunk_overlap:
                            overlap_doc.insert(0, prev_d)
                            overlap_len += prev_len + prev_sep_len
                        else:
                            break
                    current_doc = overlap_doc
                    current_len = overlap_len
                
            current_doc.append(d)
            current_len += d_len + (len(separator) if len(current_doc) > 1 else 0)
            
        if current_doc:
            chunks.append(separator.join(current_doc))
            
        return chunks

    def split_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single document dictionary into multiple chunk dictionaries.
        Keeps and updates the metadata with chunk index.
        """
        text = doc["text"]
        metadata = doc["metadata"]
        
        chunks = self._split_text(text, self.separators)
        
        split_docs = []
        for idx, chunk in enumerate(chunks):
            # Shallow copy metadata and add chunk details
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = idx
            split_docs.append({
                "text": chunk,
                "metadata": chunk_metadata
            })
            
        return split_docs

    def split_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a list of document dictionaries.
        """
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.split_document(doc))
        return all_chunks
