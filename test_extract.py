import re
from typing import List, Dict, Any

def extract_key_points(query: str, text: str, max_points: int = 4) -> List[str]:
    """
    Extracts the most salient, high-information sentences from the context
    that directly answer the query, cleaning up artifacts.
    """
    # Split by sentence boundaries
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 35]
    
    # Simple query relevance scoring based on shared keywords
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    
    scored = []
    for s in sentences:
        s_words = set(re.findall(r'\b\w{3,}\b', s.lower()))
        overlap = len(query_words.intersection(s_words))
        # Prioritize definition and feature sentences
        boost = 1.0
        if any(w in s.lower() for w in ['is a', 'provides', 'allows', 'enables', 'eliminates', 'acts as', 'serves as', 'features']):
            boost = 1.5
        score = (overlap + 1) * boost
        scored.append((score, s))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Pick top distinct points
    picked = []
    seen = set()
    for _, s in scored:
        clean_s = s.strip()
        # Avoid near-duplicates
        first_words = " ".join(clean_s.split()[:4]).lower()
        if first_words not in seen:
            seen.add(first_words)
            picked.append(clean_s)
        if len(picked) >= max_points:
            break
            
    return picked

if __name__ == '__main__':
    # Test on chunk text
    import db
    chunks = db.get_all_chunks()
    if chunks:
        c0 = chunks[0]['text']
        pts = extract_key_points('What is Hibernate and what does it provide?', c0)
        print('Extracted points:')
        for p in pts:
            print('•', p)
