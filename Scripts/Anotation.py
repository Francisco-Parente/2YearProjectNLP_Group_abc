import re


### Running this Takes LOTR text and makes it into txt with word for line
def LOTR_word_per_line():
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
    return 0

#LOTR_word_per_line()


# Function to read the file and split into sentences
def read_file_and_split(filename):
    annotated = []
    sentences = []
    labels = []
    with open(filename, 'r') as file:
        sentence = []
        label = []
        for line in file:
            line = line.strip()
            if line == '.' or line == '?' or line == '!':
                if sentence:
                    sentence.append(line)
                    sentences.append(sentence)
                    label.append('O')
                    labels.append(label)
                sentence = []
                label = []
            else:
                words = line.split()
                if len(words) > 1:
                    sentence.append(words[0])
                    label.append(words[1])
                else:
                    sentence.append(words[0])
                    label.append('O')  # Assign label '0' for words without labels
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    annotated.append(sentences)
    annotated.append(labels)
    return annotated

# Read all files and split them into sentences
file_names = ["Data\LotR\Abel_Anot.txt", "Data\LotR\Daniil_Anot.txt", "Data\LotR\Tobi_Anot.txt", "Data\LotR\Francisco_Anot.txt"]
all_labels = []

for file_name in file_names:
    sentences = read_file_and_split(file_name)
    all_labels.append(sentences)

# Displaying the result
'''for sentence in all_labels:
    words = sentence[0]
    labels = sentence[1]
    print("Words:", words)
    print("Labels:", labels)
    print()'''

# Iterate over the sentences and labels in list1
for i, (sentence, labels) in enumerate(zip(all_labels[0][0], all_labels[0][1])):
    # Check corresponding labels in other lists
    for j, other_list in enumerate([all_labels[1], all_labels[2], all_labels[2]]):
        other_labels = other_list[1][i]
        # Compare labels
        differance = []
        labels1 = []
        labels2 = []
        labels3 = []
        labels4 = []
        if labels != other_labels:
            differance.append(" ".join(sentence))
            labels1.append(" ".join(labels))
            labels2.append(" ".join(all_labels[1][1][i]))
            labels3.append(" ".join(all_labels[2][1][i]))
            labels4.append(" ".join(all_labels[3][1][i]))

            print("Sentence: ", differance)
            print("Abel:      ",labels1)
            print("Daniil:    ",labels2)
            print("Tobi:      ",labels3)
            print("Francisco: ",labels4,"\n")
            break
