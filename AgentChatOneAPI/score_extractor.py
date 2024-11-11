import re

def extract_final_score(text):
    """
    Extracts the final score from the responder's text using regex.
    Assumes the score is a number between 0 and 100.
    """
    match = re.search(r'Final Score[:\- ]+(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None