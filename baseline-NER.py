"""
"""
#!/usr/bin/python3
from nltk import tokenize
import os
import sys
import xml.dom.minidom

def parseXML(file):
    """
    """
    #TODO: implement

def get_sentence_info(sentence):
    """
    """
    #TODO: implement

def extract_entities(s):
    """
    """
    #TODO: implement

def output_entities(id, ents, outf):
    """
    """
    #TODO: implement

def evaluate(inputdir, outputfile):
    """
    Evaluate results of NER model.
    Receives a data directory and the filename for the results to evaluate.
    Prints statistics about the predicted entities in the given output file.
    NOTE: outputfile must match the pattern: task9.1_NAME_NUMBER.txt

    Args:
        - inputdir: string with folder containing original XML.
        - outputfile: string with file name with the entities produced by your system (created by output entities).
    Returns:
        - Jar return object.
    """
    return os.system(f"java -jar eval/evaluateNER.jar {inputdir}  {outputfile}")


def nerc(inputdir, outputfile):
    """
    """
    for file in inputdir:
        tree = parseXML(file)
        for sentence in tree:
            (id, text) = get_sentence_info(sentence)
            token_list = tokenize(text)
            entities = extract_entities(token_list)
            output_entities(id, entities, outputfile)
    evaluate(inputdir,outputfile)

if __name__ == "__main__":
    # Get input folder or assign default
    if len(sys.argv)>0:
        inputdir = sys.argv[0]
    else:
        inputdir = "data/Devel/"
    # Assign output file for entities
    if not os.path.exists("data/tmp"):
        os.makedirs("data/tmp")
    outputfile = "tmp/baseline-NER-entities.dat"
    # Run NERC
    # nerc(inputdir, outputfile)