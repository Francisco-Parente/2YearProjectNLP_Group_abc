### Running this Takes LOTR text and makes it into txt with word for line

import re

# Open the input file
with open("LOTR_as_txt.sty", "r", encoding="utf-8") as file:
    content = file.read()

# Remove chapter headings and names
cleaned_content = re.sub(r'Chapter\s+\w+\s+.*?\n', '', content)

# Tokenize the cleaned content into words and punctuation
tokens = re.findall(r"[mM]r.|[\w’]+|[.,!?;():\[\]]", cleaned_content)

# Write each token to a new text file
with open("LOTR_tokens.txt", "w", encoding="utf-8") as output_file:
    for token in tokens:
        output_file.write(token + "\n")
