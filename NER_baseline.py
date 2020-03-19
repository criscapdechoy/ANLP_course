"""
"""
# !/usr/bin/python3
from math import inf
from nltk.tokenize.regexp import WhitespaceTokenizer as Tokenizer
from re import sub, match
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
    text = sub(r"[,.:;'\"]", " ", s)
    tokenizer = Tokenizer()
    spans = tokenizer.span_tokenize(text)
    tokens = tokenizer.tokenize(text)
    tokens = [(t, s[0], s[1]-1) for t, s in zip(tokens, spans)]
    return tokens


def extract_entities(token_list):
    """
    Extract entitites
    Fuction to extract and tag the entites of the give token lists, taggin each
    foun entity with a type given a set of rules.
    Args:
        - token_list: list of token strings with token words
    Returns:
        - ents: list of dictionaries with entities' name, type and offset.
    """
    # Common drug suffixes/prefixes
    with open("data/Rules/drug_suffixes.txt", "r") as fp:
        drug_suffixes = [s.replace("\n", "") for s in fp.readlines()]
    with open("data/Rules/group_suffixes.txt", "r") as fp:
        group_suffixes = [s.replace("\n", "") for s in fp.readlines()]
    with open("data/Rules/brand_suffixes.txt", "r") as fp:
        brand_suffixes = [s.replace("\n", "") for s in fp.readlines()]
    with open("data/Rules/drug_n_suffixes.txt", "r") as fp:
        drug_n_suffixes = [s.replace("\n", "") for s in fp.readlines()]
    ents = []
    i = 0
    i_max = len(token_list)
    while i < i_max:
        token, start, end = token_list[i]
        nxt_token, nxt_stat, nxt_end = token_list[i+1] if i < (i_max-1) else \
            ("EOS", inf, inf)
        prv_token, prv_stat, prv_end = token_list[i-1] if i > 0 else \
            ("BOS", inf, inf)
        type = None
        # Rules to detect if token is entity
        # Detect "XX acid" drugs
        if nxt_token == "acid":
            type = "drug"
            token = f"{token} {nxt_token}"
            end = nxt_end
            i += 1
        # Detect "XX agents", "XX drugs" and "XX drug" groups
        elif ((nxt_token == "agents") or (nxt_token == "drugs")):
            type = "group"
            token = f"{token} {nxt_token}"
            end = nxt_end
            i += 1
        # Detect {Digit}-{name} drug_n
        elif (match(r"\d+-[a-zA-Z]+", token)
              or sum(w in ["+", "-", "(", ")"] for w in token)
              or token.endswith(tuple(drug_n_suffixes))):
            type = "drug_n"
        # Uppercase brand names
        # Avoid numerals and acronyms by limiting length
        # Detect common brand suffixes
        elif (token.isupper() and len(token) > 4
              or token.endswith(tuple(brand_suffixes))):
            type = "brand"
        # Numerals and acronyms probably drugs
        # If common suffix in token, probably drug
        elif ((token.isupper() and len(token) <= 4)
              or token.endswith(tuple(drug_suffixes))):
            type = "drug"
        # Single group detections
        elif (token.endswith(tuple(group_suffixes))  # Detect group suffixes
              or match(r"[A-Z]+s$", token)):  # Detect plural acronyms ADs
            type = "group"
        # If type was set, then it's an entity
        if type is not None:
            ent = {"name": token, "offset": f"{start}-{end}", "type": type}
            ents.append(ent)
        # Pass to next token
        i += 1
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
