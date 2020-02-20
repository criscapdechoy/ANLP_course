"""
"""
# !/usr/bin/python3
from nltk.tokenize import StanfordTokenizer as Tokenizer
from xml.dom.minidom import parse
from glob import glob
import os
import sys
# Global variables to control script flow
input_default_path = "data/Devel/"
tmp_path = "data/tmp"


def parseXML(file):
    """
    Parse XML file.
    Function to parse the XML file with path given as argument.

    Args:
        - file: string with path of xml file to parse.
    Returns:
        - sentences: list of xml elements of tag name "sentence".
    """
    xml = parse(file)
    sentences = xml.getElementsByTagName("sentence")
    return sentences


def get_sentence_info(sentence):
    """
    Get sentence info.
    Function to extract Id and Text from Sentence XML node.
    Args:
        - sentence: xml node object with type sentence node.
    Returns:
        - (id, text): tuple with id strng and text string.
    """
    id = sentence.getAttribute("id")
    text = sentence.getAttribute("text")
    return (id, text)


def tokenize(s):
    """
    Tokenize string.
    Function to tokenize text into words (tokens). Downloads default NLTK
    tokenizer if not in machine.
    Args:
        - s: string with sentence to tokenize.
    Returns:
        - tokens: list of tuples (token, start-index, end-index)
    """
    tokenizer = Tokenizer()
    spans = tokenizer.span_tokenize(s)
    tokens = tokenizer.tokenize(s)
    tokens = [(t, s[0], s[1]) for t, s in zip(tokens, spans)]
    return tokens


def extract_entities(token_list):
    """
    """
    ents = []
    for i, token_t in enumerate(token_list):
        token, start, end = token_t
        # Rules to detect if token is entity
        isEnt = True
        if isEnt:
            # Rules to detect type of entity
            type = "drug"
            ent = {"name": token, "offset": f"{start}-{end}", "type": type}
            ents.append(ent)
    return ents


def output_entities(id, ents, outf):
    """
    """
    # with open(f"data/tmp/{outf}", "w") as fp:


def evaluate(inputdir, outputfile):
    """
    Evaluate results of NER model.
    Receives a data directory and the filename for the results to evaluate.
    Prints statistics about the predicted entities in the given output file.
    NOTE: outputfile must match the pattern: task9.1_NAME_NUMBER.txt

    Args:
        - inputdir: string with folder containing original XML.
        - outputfile: string with file name with the entities produced by your
          system (created by output entities).
    Returns:
        - Jar return object.
    """
    return os.system(f"java -jar eval/evaluateNER.jar {inputdir} {outputfile}")


def nerc(inputdir, outputfile):
    """
    """
    files = [f for f in glob(inputdir + "**/*.xml", recursive=True)]
    for file in files:
        tree = parseXML(file)
        for sentence in tree:
            (id, text) = get_sentence_info(sentence)
            token_list = tokenize(text)
            entities = extract_entities(token_list)
            output_entities(id, entities, outputfile)
    evaluate(inputdir, outputfile)


if __name__ == "__main__":
    # Get input folder or assign default
    if len(sys.argv) > 0:
        inputdir = sys.argv[0]
    else:
        inputdir = input_default_path
    # Assign output file for entities
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        print(f"[INFO] Created a new folder {tmp_path}")
    outputfile = f"{tmp_path}/baseline-NER-entities.dat"
    # Run NERC
    nerc(inputdir, outputfile)
