
def count_words(text):
    """
    Count the number of words in the given text.
    
    Args:
        text (str): Input text string
        
    Returns:
        int: Number of words in the text
    """
    if not text:
        return 0
    return len([word for word in text.split() if word.strip()])


def count_content_lines(text):
    """
    Count the number of non-empty lines in the text.
    
    Args:
        text (str): Input text string
        
    Returns:
        int: Number of lines with content
    """
    if not text:
        return 0
    return len([line for line in text.split('\n') if line.strip()])

