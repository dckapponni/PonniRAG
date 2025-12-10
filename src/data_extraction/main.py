import json
from pathlib import Path
from collections import defaultdict

from content_extraction import count_consecutive_blanks, extract_remaining_content, extract_intro_content_phase1
from doc_utils import extract_doc_info, get_shared_authors, extract_authors_alternative
from pattern_a import extract_pattern_a_forward
from pattern_b import extract_pattern_b_forward
from pattern_c import extract_pattern_c_reverse
from shared_author import check_author_ahead, build_shared_authors_dict
from text_processing import normalize_text, is_valid_heading, fuzzy_match_author, extract_author_from_line, find_author_in_range, get_intro_keywords
from utils import count_words, count_content_lines




def parse_tamil_document(file_path, shared_authors_dict):
    """
    Parse a Tamil document file and extract intro sections, articles, and authors.
    
    Args:
        file_path (str): Path to the text file
        shared_authors_dict (dict): Dictionary of shared authors across documents
        
    Returns:
        dict: Dictionary containing intro, articles, authors_list, doc_id, and doc_issue
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    doc_id, doc_issue = extract_doc_info(lines)
    
    authors_original = []
    authors_normalized = []
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if "பொருளடக்கம்" in line:
            start_idx = i
        if "ஆகியோரின் எழுத்தோவியங்கள்" in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
      
        authors_original, authors_normalized = get_shared_authors(doc_id, doc_issue, shared_authors_dict)
        if not authors_original:
           
            authors_original, authors_normalized = extract_authors_alternative(lines)
    else:
        for i in range(start_idx + 1, end_idx):
            author_name = lines[i].strip()
            if author_name:
                authors_original.append(author_name)
                authors_normalized.append(normalize_text(author_name))
    
   
    
    parse_start_idx = end_idx + 1 if end_idx != -1 else 0
    processed_lines = [False] * len(lines)
    
    if start_idx != -1 and end_idx != -1:
        for i in range(start_idx, end_idx + 1):
            processed_lines[i] = True
    
    intro_keywords = get_intro_keywords()
    articles = []
    intro = []
    article_no = 1
    
   
    i = parse_start_idx
    phase1_count = 0
    
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        matched_keyword = None
        
        for keyword in intro_keywords:
            if keyword in line:
                if i == 0 or not lines[i - 1].strip():
                    matched_keyword = keyword
                    break
        
        if matched_keyword:
            content, content_end_idx, has_author = extract_intro_content_phase1(
                lines, i, processed_lines, authors_normalized, authors_original, intro_keywords
            )
            
            for j in range(i, content_end_idx):
                if j < len(lines):
                    processed_lines[j] = True
            
            if content.strip() and count_content_lines(content) >= 4:
                if has_author:
                    articles.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "article_no": article_no,
                        "article_heading": matched_keyword,
                        "article_author_name": has_author,
                        "article_content": content
                    })
                    article_no += 1
                   
                else:
                    intro.append({
                        "doc_id": doc_id,
                        "doc_issue": doc_issue,
                        "heading": matched_keyword,
                        "content": content
                    })
                    phase1_count += 1
                   
            
            i = content_end_idx
        else:
            i += 1
    
    
    pattern_a_articles = extract_pattern_a_forward(
        lines, parse_start_idx, len(lines), 
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_a_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    

    pattern_b_articles = extract_pattern_b_forward(
        lines, parse_start_idx, len(lines),
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_b_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
   
    pattern_c_articles = extract_pattern_c_reverse(
        lines, parse_start_idx, len(lines),
        authors_normalized, authors_original, processed_lines, intro_keywords
    )
    
    for article in pattern_c_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
   
    remaining_articles = extract_remaining_content(
        lines, parse_start_idx, processed_lines
    )
    
    for article in remaining_articles:
        articles.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "article_no": article_no,
            "article_heading": article["heading"],
            "article_author_name": article["author"],
            "article_content": article["content"]
        })
        article_no += 1
    
    authors_list = []
    for author in authors_original:
        authors_list.append({
            "doc_id": doc_id,
            "doc_issue": doc_issue,
            "author_name": author
        })
    
    
    
    return {
        "intro": intro,
        "articles": articles,
        "authors_list": authors_list,
        "doc_id": doc_id,
        "doc_issue": doc_issue
    }


def process_folder(root_folder_path, output_root_folder="output_json"):
    """
    Process all text files in a folder and extract structured content.
    
    Args:
        root_folder_path (str): Path to input folder containing text files
        output_root_folder (str): Path to output folder for JSON files
    """
    root_path = Path(root_folder_path)
    output_root = Path(output_root_folder)
    
    if not root_path.exists():
        
        return
    
    output_root.mkdir(parents=True, exist_ok=True)
    txt_files = list(root_path.rglob("*.txt"))
    
    if not txt_files:
       
        return
    
    
    
    shared_authors_dict = build_shared_authors_dict(root_path)
    folder_authors = defaultdict(lambda: defaultdict(set))
    
    for txt_file in txt_files:
        try:
           
            
            result = parse_tamil_document(str(txt_file), shared_authors_dict)
            
            relative_path = txt_file.relative_to(root_path)
            output_folder = output_root / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            base_name = txt_file.stem
            
            content_json_path = output_folder / f"{base_name}.json"
            with open(content_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "intro": result["intro"],
                    "articles": result["articles"]
                }, f, ensure_ascii=False, indent=2)
            
          
            
            folder_key = relative_path.parent
            doc_key = (result["doc_id"], result["doc_issue"])
            for author_info in result["authors_list"]:
                folder_authors[folder_key][doc_key].add(author_info["author_name"])
            
        except Exception as e:
          
            continue

   
    for folder_key, doc_authors_dict in folder_authors.items():
        output_folder = output_root / folder_key
        folder_name = folder_key.name if folder_key.name else "root"
        
        consolidated_authors = []
        for (doc_id, doc_issue), authors_set in doc_authors_dict.items():
            consolidated_authors.append({
                "doc_id": doc_id,
                "doc_issue": doc_issue,
                "authors": sorted(list(authors_set))
            })
        
        authors_json_path = output_folder / f"{folder_name}_authors.json"
        with open(authors_json_path, 'w', encoding='utf-8') as f:
            json.dump(consolidated_authors, f, ensure_ascii=False, indent=2)
        
       

if __name__ == "__main__":
   
    print("Processing....")
    root_folder = "extracted_texts"
    output_folder = "output"
    process_folder(root_folder, output_folder)
    print("completed.")