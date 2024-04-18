import re
import random
import os

### Running this Takes Hobbit text and makes it into txt with word for line
def Hobbit_word_per_line():
    # Open the input file
    with open("Scripts\Hobbit_as_txt.sty", "r", encoding="utf-8") as file:
        content = file.read()

    # Tokenize the cleaned content into words and punctuation
    tokens = re.findall(r"[mM]r.|[\w’]+|[.,!?;():\[\]]", content)

    # Write each token to a new text file
    with open("Scripts\Hobbit_tokens.txt", "w", encoding="utf-8") as output_file:
        for token in tokens:
            output_file.write(token + "\n")
    return 0


def LOTR_word_per_line(input_file):
    # Get the directory and filename without extension
    directory = os.path.dirname(input_file)
    filename = os.path.splitext(os.path.basename(input_file))[0]

    # Open the input file
    with open(input_file, "r", encoding="utf-8") as file:
        content = file.read()

    # Remove chapter headings and names
    cleaned_content = re.sub(r'_Chapter\s+\d+_\n[\s\S]*?\n\n', '', content)

    # Tokenize the cleaned content into words and punctuation
    tokens = re.findall(r"[mM]r\.|[\w’]+|[.,!?;():\[\]]", cleaned_content)

    # Write each token to a new text file
    output_filename = os.path.join(directory, f"{filename}_tokens.txt")
    with open(output_filename, "w", encoding="utf-8") as output_file:
        for token in tokens:
            output_file.write(token + "\n")
    return output_filename

def combine_text_files(input_files):
    output_file_name = "Scripts\combined_tokens.txt"
    with open(output_file_name, "w", encoding="utf-8") as output:
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as input:
                output.write(input.read())

def read_and_write_random_sentences(input_file, output_files):
    # Set random seed
    random.seed(666)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Split the content into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=(?:[^\'\"]*\'[^\'\"]*\')*[^\'\"]*$)', content)

    # Shuffle the sentences randomly
    random.shuffle(sentences)

    # Write 500 random sentences to each output file
    for i, output_file in enumerate(output_files):
        start_index = i * 500
        end_index = start_index + 500
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(sentences[start_index:end_index]))
#-----------------------------Comparing annotations----------------

# Function to read the file and split into sentences
def read_file_and_split(filename):
    annotated = []
    sentences = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
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

def all_annotations_to_list(file_name):
    for file_name in file_names:
        single_person_annotation = read_file_and_split(file_name)
        all_labels = []
        all_labels.append(single_person_annotation)
    return all_labels

def compare_labels(file_name):

    all_labels = all_annotations_to_list(file_name)

    # Iterate over the sentences and labels in lists
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
                print("Daniil:   ",labels1)
                print("Abel:     ",labels2)
                print("Tobi:     ",labels3)
                print("Francisco:",labels4,"\n")
                break

#---------------------------Using functions---------------------------------
Hobbit_word_per_line()
files = []
files.append("Scripts\The Fellowship Of The Ring.txt")
files.append("Scripts\The Return Of The King.txt")
files.append("Scripts\The Two Towers.txt")
for name in files:
    LOTR_word_per_line(name)

token_files = []
token_files.append("Scripts\Hobbit_tokens.txt")
token_files.append("Scripts\The Fellowship Of The Ring_tokens.txt")
token_files.append("Scripts\The Return Of The King_tokens.txt")
token_files.append("Scripts\The Two Towers_tokens.txt")
combine_text_files(token_files)


output_files = ['Data\LotR\LabelingFinals\Tobi.txt', 'Data\LotR\LabelingFinals\Abel.txt', 'Data\LotR\LabelingFinals\Daniila.txt']
# Call the function to read and write random sentences
read_and_write_random_sentences("Scripts\combined_tokens.txt", output_files)

file_names = ["Data\LotR\Labeling2nd\Daniil_Anot2.txt", "Data\LotR\Labeling2nd\Abel_Anot2.txt", "Data\LotR\Labeling2nd\Tobi_Anot2.txt", "Data\LotR\Labeling2nd\Francisco_Anot2.txt"]
#compare_labels(file_names)

