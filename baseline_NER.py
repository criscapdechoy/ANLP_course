"""
"""
# !/usr/bin/python3
from nltk.tokenize.regexp import WhitespaceTokenizer as Tokenizer
from re import sub
from xml.dom.minidom import parse
import os
# Global variables to control script flow
input_default_path = "data/Devel"
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
    text = sub(r"[(,.:;'\")]+", " ", s)
    tokenizer = Tokenizer()
    spans = tokenizer.span_tokenize(text)
    tokens = tokenizer.tokenize(text)
    tokens = [(t, s[0], s[1]) for t, s in zip(tokens, spans)]
    return tokens


def extract_entities(token_list):
    """
    """
    # Common drug suffixes
    with open("data/Rules/sufixes.txt", "r") as fp:
        terms = [s.replace("\n", "") for s in fp.readlines()]
    ents = []
    for i, token_t in enumerate(token_list):
        token, start, end = token_t
        type = None
        # Rules to detect if token is entity
        if token.isupper() and len(token) > 4:
            # Uppercase brand names
            # Avoid numerals and acronyms by limiting length
            type = "brand"
        for term in terms:
            # If common term in token, probably drug
            if term in token:
                type = "drug"
                break
        if type is not None:
            ent = {"name": token, "offset": f"{start}-{end}", "type": type}
            ents.append(ent)
    return ents


def output_entities(id, ents, outf):
    """
    Args:
        - id: string with document id.
        - ents: list of entities dictionaries with {name, offset, type}
        - outf: path for output file
    """
    with open(outf, "a") as fp:
        for ent in ents:
            offset = ent["offset"]
            name = ent["name"]
            type = ent["type"]
            txt = f"{id}|{offset}|{name}|{type}\n"
            fp.write(txt)


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
    files = [f"{inputdir}/{f}" for f in os.listdir(inputdir)]
    for file in files:
        tree = parseXML(file)
        for sentence in tree:
            (id, text) = get_sentence_info(sentence)
            token_list = tokenize(text)
            entities = extract_entities(token_list)
            output_entities(id, entities, outputfile)
    evaluate(inputdir, outputfile)
    return "DONE!"


if __name__ == "__main__":
    inputdir = input_default_path
    # Assign output file for entities
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        print(f"[INFO] Created a new folder {tmp_path}")
    # Evaluation output file config
    outputfile = f"{tmp_path}/task9.1_BASELINE_1.txt"
    if os.path.exists(outputfile):
        os.remove(outputfile)
    # Run NERC
    nerc(inputdir, outputfile)
