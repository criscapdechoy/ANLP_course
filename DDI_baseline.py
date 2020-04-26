"""
DDI Baseline Script.
Python scripts that contains functions to perform a rule-based Drug-Drug
Interaction detection model.
"""
# !/usr/bin/python3
from nltk.parse.corenlp import CoreNLPDependencyParser
from os import listdir, system, path, makedirs
from sys import exit
from xml.dom.minidom import parse
# Global variables and procedures
input_default_path = "data/Test-DDI"
tmp_path = "data/tmp"
if not path.exists(tmp_path):  # Create dir if not exists
    makedirs(tmp_path)
    print(f"[INFO] Created a new folder {tmp_path}")
output_default_path = f"{tmp_path}/task9.2_BASELINE_2.txt"
# Get CoreNLP instance, which need to be running in http://localhost:9000
DependencyParser = CoreNLPDependencyParser(url="http://localhost:9000")
StanfordCoreNLPServer_error = (
    f"ERROR - StanfordCoreNLPServer connection error. "
    f"Server needs to be running before calling the sript. "
    f"Use the following command to run the server:\n"
    f"\tjava -mx4g -cp 'resources/stanford-corenlp/*' "
    f"edu.stanford.nlp.pipeline.StanfordCoreNLPServer "
    f"-port 9000 -timeout 30000"
    )


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


def analyze(s):
    """
    Analyze Text.
    Function which uses an already started Stanford CoreNLP server to analyze
    given text, that is tokenize, tag (Part of Speech tag) and parse.
    Args:
        - text: string with text to analyze.
    Returns:
        - analysis: output of CoreNLP dependency parser.
    """
    # Dependency Parsing
    try:
        analysis, = DependencyParser.raw_parse(s)
    except Exception:
        print(StanfordCoreNLPServer_error)
        exit()
    # Return analysis
    return analysis


def is_child(analysis, node_index, text):
    """
    Check is child.
    Given a DependencyGraph analysis of a sentence, and a node_index for a
    certain node of the graph (or None for the root), this function checks if
    there is a child which contains the given text.
    Args:
        - analysis: DependencyGraph object instance with setnence parsed
            information.
        - node_index: int with node index to check childs from.
        - text: text to look for in child nodes.
    Returns:
        - _: boolean True if text is found in children, False otherwise.
    """
    node = analysis.get_by_address(node_index)
    for triple in analysis.triples(node=node):
        if text in str(triple[2][0]).lower():
            return True
    return False


def get_clue_nodes(nodes, clues):
    """
    Get Clue Nodes.
    Function which gets verb (tag VB*) nodes from which to look for child
    entities. Gets a list of possible clue wordst that indicate a certain type
    of DDI and returns a list with the closest verb ancestors of the nodes
    which contain the clue words.
    Args:
        - nodes: list of current Dependency Graph nodes.
        - clues: list of clue words for a certain type of DDI.
    Returns:
        - _nodes: list of node dictionaries to search child entities from.
    """
    clue_nodes = [n for n in nodes if nodes[n]["lemma"] in clues]
    _nodes = []
    for clue in clue_nodes:
        node = nodes[clue]
        while node["tag"] != "TOP" and "VB" not in node["tag"]:
            node = nodes[node["head"]]
            if not node["tag"]:
                break
        _nodes.append(node["address"])
    return list(set(_nodes))


def check_interaction(analysis, entitities, e1, e2):
    """
    Check Interaction.
    Function to check for interaction of the given pair of entities, by looking
    at the parsed (analysed) sentences which contains it together with all the
    entities in the sentence.
    Args:
        - analysis: DependencyGraph object instance with setnence parsed
            information.
        - entities: dictionary of entities indexed by id with offset as value.
        - e1: string with id of the first entity to consider.
        - e2: string with id of the second entity to consider.
    Return:
        - is_ddi: integer with 0/1 value indicating whether the sentence states
            an interaction between e1 and e2.
        - ddi_type: string with the type of interaction, or null if none.
    """
    is_ddi = False
    ddi_type = "null"

    # Get analysis nodes
    nodes = analysis.nodes
    # Get entities text
    # Split/Lower used to take standard first part of multi-word Drugs
    e1_text = entitities[e1]["text"].split()[0].lower()
    e2_text = entitities[e2]["text"].split()[0].lower()

    # DDI type clues
    advise_clues = ["should", "must", "may", "recommend", "caution"]
    effect_clues = ["produce", "administer", "potentiate", "prevent", "effect"]
    int_clues = ["interact", "interaction"]
    mechanism_clues = ["reduce", "increase", "decrease"]

    # Avoid check if entities are the same or almost the same
    if e1_text == e2_text or e1_text in e2_text or e2_text in e1_text:
        pass

    # Check Advise clues
    elif any(
        is_child(analysis, clue, e1_text) or is_child(analysis, clue, e2_text)
        for clue in get_clue_nodes(nodes, advise_clues)
    ):
        is_ddi = True
        ddi_type = "advise"

    # Check Effect clues
    elif any(
        is_child(analysis, clue, e1_text) or is_child(analysis, clue, e2_text)
        for clue in get_clue_nodes(nodes, effect_clues)
    ):
        is_ddi = True
        ddi_type = "effect"

    # Check Mechanism clues
    elif any(
        is_child(analysis, clue, e1_text) or is_child(analysis, clue, e2_text)
        for clue in get_clue_nodes(nodes, mechanism_clues)
    ):
        is_ddi = True
        ddi_type = "mechanism"

    # Check Int clues
    elif any(
        is_child(analysis, clue, e1_text) or is_child(analysis, clue, e2_text)
        for clue in get_clue_nodes(nodes, int_clues)
    ):
        is_ddi = True
        ddi_type = "int"

    return "1" if is_ddi else "0", ddi_type


def evaluate(inputdir, outputfile):
    """
    Evaluate results of DDI model.
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
    return system(f"java -jar eval/evaluateDDI.jar {inputdir} {outputfile}")


def main(inputdir, outputfile):
    """
    Main function.
    Function that runs the module workflow for the DDI baseline, with rule
    based detection of DDI.
    Args:
        - inputdir: string with folder containing original XML.
        - outputfile: string with file name to save output to.
    """
    # Open output file
    outf = open(outputfile, "w")
    # process each file in directory
    files = listdir(inputdir)
    for f in files:
        # Parse XML file
        sentences = parseXML(f"{inputdir}/{f}")
        for s in sentences:
            # get sentence id/text
            sid = s.attributes["id"].value
            stext = s.attributes["text"].value
            if not stext:  # Do not process if sentence is empty
                continue

            # load sentence entities into a dictionary
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                id = e.attributes["id"].value
                offs = e.attributes["charOffset"].value.split("-")
                text = e.attributes["text"].value
                entities[id] = {"offset": offs, "text": text}

            # Tokenize, tag, and parse sentence
            analysis = analyze(stext)

            # for each pair in the sentence
            # decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                (is_ddi, ddi_type) = check_interaction(
                    analysis, entities, id_e1, id_e2)
                print("|".join([sid, id_e1, id_e2, is_ddi, ddi_type]),
                      file=outf)
    # Close outputfile
    outf.close()

    # get performance score
    evaluate(inputdir, outputfile)


if __name__ == "__main__":
    inputdir = input_default_path
    outputfile = output_default_path
    main(inputdir, outputfile)
