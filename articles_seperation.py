import json
import re
import os
from pathlib import Path
from difflib import SequenceMatcher

def count_words(text):
    """Count words in text."""
    if not text:
        return 0

    words = [word for word in text.split() if word.strip()]
    return len(words)

def normalize_text(text):
    """Normalize text by removing extra spaces, dots, and converting to lowercase."""
    text = text.replace('ண', 'ண').replace('ணु', 'ணु')
    text = re.sub(r'\.', '', text)
    text = re.sub(r'\s+', '', text.strip().lower())
    return text

def fuzzy_match_author(line, authors_normalized, authors_original, threshold=0.8):
    """
    Fuzzy match line against authors with 80-100% similarity.
    Returns (matched_author_name, similarity) or (None, 0) if no match.
    """
    if not line.strip():
        return (None, 0)
    
    line_normalized = normalize_text(line)
    best_match = None
    best_similarity = 0
    
    for i, author_norm in enumerate(authors_normalized):
        similarity = SequenceMatcher(None, line_normalized, author_norm).ratio()
        if similarity >= threshold and similarity > best_similarity:
            best_similarity = similarity
            best_match = authors_original[i]
    
    return (best_match, best_similarity)

def extract_author_from_brackets(line, authors_normalized, authors_original):
    """Extract author from content within brackets [...]."""
    if '[' in line and ']' in line:
        bracket_content = re.search(r'\[(.*?)\]', line)
        if bracket_content:
            content = bracket_content.group(1)
            
           
            parts = content.split('"')
            if len(parts) > 0:
                potential_author = parts[0].strip()
                matched_author, similarity = fuzzy_match_author(potential_author, authors_normalized, authors_original)
                if matched_author:
                    return matched_author
            
           
            if 'ஆசிரியர்' in content or ':' in content:
                parts = content.split(':')
                if len(parts) > 1:
                    potential_author = parts[-1].strip()
                    potential_author = re.sub(r'[,.\]]+$', '', potential_author)
                    matched_author, similarity = fuzzy_match_author(potential_author, authors_normalized, authors_original)
                    if matched_author:
                        return matched_author
    return None

def extract_author_from_pattern(line, authors_normalized, authors_original):
    """Extract author from patterns like 'ஆசிரியர் : author_name'."""
    if 'ஆசிரியர்' in line or ':' in line:
        parts = line.split(':')
        if len(parts) > 1:
            potential_author = parts[-1].strip()
            matched_author, similarity = fuzzy_match_author(potential_author, authors_normalized, authors_original)
            if matched_author:
                return matched_author
    return None

def find_author_in_range(lines, start_idx, end_idx, authors_normalized, authors_original):
    """Search for author name within a range of lines."""
    for i in range(start_idx, min(end_idx, len(lines))):
        line = lines[i].strip()
        
       
        author = extract_author_from_brackets(line, authors_normalized, authors_original)
        if author:
            return (author, i)
        
       
        author = extract_author_from_pattern(line, authors_normalized, authors_original)
        if author:
            return (author, i)
        
       
        matched_author, similarity = fuzzy_match_author(line, authors_normalized, authors_original)
        if matched_author:
            return (matched_author, i)
    
    return (None, -1)

def extract_doc_info(lines):
    """
    Extract மலர் (doc_id) and இதழ் (doc_issue) from document.
    Returns: (doc_id, doc_issue)
    """
    doc_id = "Unknown"
    doc_issue = "Unknown"
    
    for line in lines[:50]: 
        line_stripped = line.strip()
        
       
        if 'மலர்' in line_stripped:
           
            match = re.search(r'மலர்\s*[:—-]?\s*(\d+)', line_stripped)
            if match:
                doc_id = match.group(1)
        
        
        if 'இதழ்' in line_stripped:
           
            match = re.search(r'இதழ்\s*[:—-]?\s*(\d+)', line_stripped)
            if match:
                doc_issue = match.group(1)
    
    return (doc_id, doc_issue)

def extract_authors_alternative(lines):
    """
    Alternative method to extract authors when no author list section exists.
    New Logic:
    - Look for: 2+ empty lines → author name → content (30+ chars on 1st/2nd/3rd line above)
    - Search bottom-to-top for heading (≤20 chars) with empty line above it
    Returns: (authors_original, authors_normalized)
    """
    authors_original = []
    authors_normalized = []
    authors_set = set() 
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
       
        if not line:
            blank_count = 0
            blank_start = i
            while i < len(lines) and not lines[i].strip():
                blank_count += 1
                i += 1
            
            if blank_count >= 2 and i < len(lines):
                
                potential_author_idx = i
                potential_author = lines[potential_author_idx].strip()
                
               
                if potential_author and 3 <= len(potential_author) <= 30:
                   
                    has_content = False
                    for offset in [1, 2, 3]:
                        check_idx = potential_author_idx - offset
                        if check_idx >= 0:
                            check_line = lines[check_idx].strip()
                            if len(check_line) > 30:
                                has_content = True
                                break
                    
                    if has_content:
                        
                        normalized = normalize_text(potential_author)
                        if normalized not in authors_set:
                            authors_original.append(potential_author)
                            authors_normalized.append(normalized)
                            authors_set.add(normalized)
        
        i += 1
    
    return (authors_original, authors_normalized)

def parse_tamil_document(file_path):
    """
    Parse Tamil TXT document with strict rules.
    Returns: {intro: [...], articles: [...], authors_list: [...]}
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
        print(f"  ℹ No author list section found. Using alternative extraction method...")
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
    
   
    articles = []
    article_no = 1
    
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        

        author_found, author_idx = find_author_in_range(lines, i, i + 1, authors_normalized, authors_original)
        
        if author_found:
            heading = extract_heading(lines, author_idx, authors_normalized, authors_original, processed_lines)
            
           
            content, content_end_idx = extract_content_after_author(lines, author_idx, authors_normalized, authors_original, processed_lines)
            
            mark_heading_lines_processed(lines, author_idx, authors_normalized, authors_original, processed_lines)
            processed_lines[author_idx] = True
            for j in range(author_idx + 1, content_end_idx):
                processed_lines[j] = True
            
            if heading.strip() and count_words(content) >= 50:
                articles.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "article_no": article_no,
                    "article_heading": heading,
                    "article_author_name": author_found,
                    "article_content": content
                })
                article_no += 1
            
            i = content_end_idx
        else:
            i += 1
    
    intro_keywords = [
        "எங்கள் எண்ணம்",
        "கலையுலகம்",
        "வளரும் இலக்கியம்",
        "காலமும் கருத்தும்",
        "பாரதிதாசன் பரம்பரை",
        "வள்ளுவர் விருந்து",
        "பொது மேடை",
        "செய்திப் பாட்டு",
        "கலை உலகம்",
        "அட்டைப் படம்",
        "இந்தி வேண்டாம்!",
        "இந்தி வந்தது, இந்தி!",
        "உயர்திரு உல்லாசம் அவர்கட்கு"
    ]
    intro = []
    
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        line = lines[i].strip()
        
        matched_keyword = None
        for keyword in intro_keywords:
            if keyword in line:
                matched_keyword = keyword
                break
        
        if matched_keyword:
            content, content_end_idx = extract_intro_content(lines, i, processed_lines, authors_normalized, authors_original)
            
            for j in range(i, content_end_idx):
                processed_lines[j] = True
        
            content_line_count = len([line for line in content.split('\n') if line.strip()])
            if count_words(content) >= 50 and content_line_count > 2:
                intro.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "heading": matched_keyword,
                    "content": content
                })
            
            i = content_end_idx
        else:
            i += 1
    
    i = parse_start_idx
    while i < len(lines):
        if processed_lines[i]:
            i += 1
            continue
        
        
        line = lines[i].strip()
        if not line:
            blank_count = count_consecutive_blanks(lines, i, processed_lines)
            if blank_count >= 4:
               
                for j in range(i, i + blank_count):
                    if j < len(lines):
                        processed_lines[j] = True
                i += blank_count
                continue
        
      
        if line: 
            content, content_end_idx = extract_orphan_content_na(lines, i, processed_lines, authors_normalized, authors_original)
            
           
            for j in range(i, content_end_idx):
                processed_lines[j] = True
            
           
            content_line_count = len([line for line in content.split('\n') if line.strip()])
            if heading != "NA" and count_words(content) >= 50 and content_line_count > 2:
                articles.append({
                    "doc_id": doc_id,
                    "doc_issue": doc_issue,
                    "article_no": article_no,
                    "article_heading": heading,
                    "article_author_name": "NA",
                    "article_content": content
                })
                article_no += 1
            
            i = content_end_idx
        else:
            i += 1
    
    
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
        "authors_list": authors_list
    }

def count_consecutive_blanks(lines, start_idx, processed_lines):
    """Count consecutive blank lines starting from start_idx."""
    count = 0
    i = start_idx
    while i < len(lines) and not lines[i].strip() and not processed_lines[i]:
        count += 1
        i += 1
    return count

def mark_heading_lines_processed(lines, author_idx, authors_normalized, authors_original, processed_lines):
    """Mark heading lines (1 or 2 lines above author) as processed."""
   
    if author_idx - 2 >= 0:
        line2 = lines[author_idx - 2].strip()
        matched_author, _ = fuzzy_match_author(line2, authors_normalized, authors_original)
        if line2 and not matched_author and len(line2) <= 25 and '[' not in line2:
            processed_lines[author_idx - 2] = True
    
   
    if author_idx - 1 >= 0:
        line1 = lines[author_idx - 1].strip()
        matched_author, _ = fuzzy_match_author(line1, authors_normalized, authors_original)
        if line1 and not matched_author and len(line1) <= 25 and '[' not in line1:
            processed_lines[author_idx - 1] = True

def extract_heading(lines, author_idx, authors_normalized, authors_original, processed_lines):
    """Extract heading from lines above author (max 25 chars)."""
    heading_lines = []
    
   
    if author_idx - 2 >= 0:
        line2 = lines[author_idx - 2].strip()
        matched_author, _ = fuzzy_match_author(line2, authors_normalized, authors_original)
        if line2 and not matched_author and len(line2) <= 25 and '[' not in line2:
            heading_lines.append(line2)
    
    
    if author_idx - 1 >= 0:
        line1 = lines[author_idx - 1].strip()
        matched_author, _ = fuzzy_match_author(line1, authors_normalized, authors_original)
        if line1 and not matched_author and len(line1) <= 25 and '[' not in line1:
            heading_lines.append(line1)
    
    if len(heading_lines) == 2:
        return ' '.join(heading_lines)
    elif len(heading_lines) == 1:
        return heading_lines[0]
    else:
        return ""

def extract_content_after_author(lines, author_idx, authors_normalized, authors_original, processed_lines):
    """Extract content after author line. Stops at: 4+ blanks OR another author."""
    content_lines = []
    i = author_idx + 1
    
    
    if i < len(lines) and not lines[i].strip():
        i += 1
    
    consecutive_blanks = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
    
        author_check = find_author_in_range(lines, i, i + 1, authors_normalized, authors_original)
        if author_check[0] is not None:
    
            if len(content_lines) >= 2:
                content_lines = content_lines[:-2]
            elif len(content_lines) == 1:
                content_lines = content_lines[:-1]
            break
        

        if not stripped:
            consecutive_blanks += 1
            if consecutive_blanks >= 4:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            consecutive_blanks = 0
            content_lines.append(line.rstrip())
        
        i += 1
    
    return ('\n'.join(content_lines), i)

def extract_intro_content(lines, start_idx, processed_lines, authors_normalized, authors_original):
    """
    Extract intro content. 
    NEW LOGIC: First check if author name found - stop 2 lines before author.
    OLD LOGIC: If no author found, stop at 4+ consecutive blank lines.
    Content must be more than 2 lines.
    """
    content_lines = []
    i = start_idx + 1
    
    if i < len(lines) and not lines[i].strip():
        i += 1
    
    consecutive_blanks = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
      
        author_check = find_author_in_range(lines, i, i + 1, authors_normalized, authors_original)
        if author_check[0] is not None:
           
            if len(content_lines) >= 2:
                content_lines = content_lines[:-2]
            elif len(content_lines) == 1:
                content_lines = content_lines[:-1]
            break
        
       
        if not stripped:
            consecutive_blanks += 1
            if consecutive_blanks >= 4:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            consecutive_blanks = 0
            content_lines.append(line.rstrip())
        
        i += 1
    
    return ('\n'.join(content_lines), i)

def extract_orphan_article(lines, start_idx, processed_lines):
    """Extract orphan content with heading. First non-empty line ≤25 chars becomes heading."""
    heading = ""
    content_start = start_idx
    
    
    first_line = lines[start_idx].strip()
    if len(first_line) <= 25:
        heading = first_line
        content_start = start_idx + 1
    else:
        content_start = start_idx
    
   
    content_lines = []
    i = content_start
    consecutive_blanks = 0
    
    while i < len(lines):
        if processed_lines[i]:
            break
        
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            consecutive_blanks += 1
            if consecutive_blanks >= 4:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            consecutive_blanks = 0
            content_lines.append(line.rstrip())
        
        i += 1
    
    return (heading, '\n'.join(content_lines), i)

def extract_orphan_content_na(lines, start_idx, processed_lines, authors_normalized, authors_original):
    """
    Extract remaining orphan content with heading extracted from content.
    New Logic for Heading:
    - Search bottom-to-top from current position
    - Find heading (≤20 chars) with empty line above it
    - Stop searching when found
    
    Content Extraction (DUAL LOGIC):
    - NEW LOGIC: Check if author name found - stop 2 lines before author
    - OLD LOGIC: If no author found, stop at 4+ consecutive blank lines
    
    Content must be more than 2 lines.
    """
    content_lines = []
    heading = "NA"
    i = start_idx
    consecutive_blanks = 0
    
   
    while i < len(lines):
        if processed_lines[i]:
            break
        
        line = lines[i]
        stripped = line.strip()
        
       
        author_check = find_author_in_range(lines, i, i + 1, authors_normalized, authors_original)
        if author_check[0] is not None:
           
            if len(content_lines) >= 2:
                content_lines = content_lines[:-2]
            elif len(content_lines) == 1:
                content_lines = content_lines[:-1]
            break
        
       
        if not stripped:
            consecutive_blanks += 1
            if consecutive_blanks >= 4:
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()
                break
            else:
                content_lines.append(line.rstrip())
        else:
            consecutive_blanks = 0
            content_lines.append(line.rstrip())
        
        i += 1
    
   
    content_text = '\n'.join(content_lines)
    content_list = content_text.split('\n')
    
    for idx in range(len(content_list) - 1, -1, -1):
        line_text = content_list[idx].strip()
        
       
        if line_text and len(line_text) <= 20:
            if idx > 0:
                line_above = content_list[idx - 1].strip()
                if not line_above: 
                    heading = line_text
                   
                    content_list = content_list[:idx - 1]
                    break
            elif idx == 0:
               
                heading = line_text
                content_list = content_list[1:]
                break
    
    final_content = '\n'.join(content_list)
    
    return (final_content, i)

def process_folder(root_folder_path, output_root_folder="output_json"):
    """
    Process all TXT files in all subfolders of root folder.
    Creates JSON files in a separate output folder with same structure.
    """
    root_path = Path(root_folder_path)
    output_root = Path(output_root_folder)
    
    if not root_path.exists():
        print(f"Error: Folder '{root_folder_path}' does not exist!")
        return
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    txt_files = list(root_path.rglob("*.txt"))
    
    if not txt_files:
        print(f"No TXT files found in '{root_folder_path}' or its subfolders.")
        return
    
    print(f"Found {len(txt_files)} TXT file(s) to process...\n")
    
    for txt_file in txt_files:
        try:
            print(f"Processing: {txt_file}")
            
    
            result = parse_tamil_document(str(txt_file))
        
            relative_path = txt_file.relative_to(root_path)
        
            output_folder = output_root / relative_path.parent
            output_folder.mkdir(parents=True, exist_ok=True)
            
            base_name = txt_file.stem
            
            content_json_path = output_folder / f"{base_name}_parsed.json"
            with open(content_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "intro": result["intro"],
                    "articles": result["articles"]
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Created: {content_json_path}")
            
            authors_json_path = output_folder / f"{base_name}_authors.json"
            with open(authors_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "authors": result["authors_list"]
                }, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Created: {authors_json_path}")
            print(f"  ✓ Successfully processed!\n")
            
        except Exception as e:
            print(f"  ✗ Error processing {txt_file}: {str(e)}\n")
            continue
    
    print(f"\nProcessing complete! All JSON files saved in: {output_root.absolute()}")

if __name__ == "__main__":
    print(" Tamil TXT Document Parser \n")

   
    root_folder = "extracted_texts"  
    output_folder = "output_json" 

    print(f"Input folder: {root_folder}")
    print(f"Output folder: {output_folder}\n")

    process_folder(root_folder, output_folder)

    