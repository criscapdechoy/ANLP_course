"""
DDI Machine Learning Script.
Python scripts that contains functions to train and evaluate a ML model to
detect Drug-Drug Interaction.
"""
# !/usr/bin/python3
from nltk.parse.corenlp import CoreNLPDependencyParser
from os import listdir, system, path, makedirs
from sklearn.ensemble import RandomForestClassifier
from sys import exit
from xml.dom.minidom import parse
import pickle

# Reference constants
MODELS = ["MaxEnt", "RandomForest"]

# Global variables and procedures
tmp_path = "data/tmp"
if not path.exists(tmp_path):  # Create dir if not exists
    makedirs(tmp_path)
    print(f"[INFO] Created a new folder {tmp_path}")
model = "RandomForest"
train_input_fn = "data/Train"
valid_input_fn = "data/Devel"
train_features_fn = f"{tmp_path}/DDI_ML_train_features.txt"
valid_features_fn = f"{tmp_path}/DDI_ML_valid_features.txt"
outputfile = f"{tmp_path}/task9.2_ML_1.txt"

# Model path to save it
ml_model_fn = f"{tmp_path}/DDI_ML_model"
# Specify local megam file
megam = "resources/megam_i686.opt"

# Random forest params
random_seed = 42
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


def get_entity_node(analysis, entities, entity):
    """
    Get Entity Node.
    Function which finds the node in the Dependency Tree which corresponds to
    the root of the entity.
    Args:
        - analysis: DependencyTree object instance with sentence analysis.
        - entities: dictionary with entity information.
        - entity: string with id of entity to get.
    Returns:
        - node: dictionary with node from DependencyTree.
    """
    # Get nodes list
    nodes = [analysis.nodes[k] for k in analysis.nodes]
    ent = entities[entity]["text"]
    # Capture possible tree nodes containing or that are contained in entity
    possible = sorted(
        [node for node in nodes if node["word"] is not None and
         (node["word"] in ent or ent in node["word"])],
        key=lambda x: x["head"])
    node = possible[0] if len(possible) else nodes[0]
    return node


def get_verb_ancestor(analysis, node):
    """
    Get Verb Ancestor.
    Function which looks in the node's antecessor nodes inthe analysis tree
    until it finds a verb VB, and returns such verb.
    Args:
        - analysis: DependencyTree object instance with sentence analysis.
        - node: dictionary with node to start from.
    Return:
        - node: dictionary with verb antecessor node from DependencyTree.
    """
    nodes = analysis.nodes
    while node["tag"] != "TOP" and "VB" not in node["tag"]:
        node = nodes[node["head"]]
        if not node["tag"]:
            break
    return node


def extract_features(analysis, entities, e1, e2):
    """
    Extract Features.
    Function which receives an analyzed sentence tree, the entities
    present in the sentence, and the ids of the two target entities and returns
    a list of features to pass to a ML model to predict DDI.
    Args:
        - analysis: DependencyGraph object instance with setnence parsed
            information.
        - entities: dictionary of entities indexed by id with offset as value.
        - e1: string with id of the first entity to consider.
        - e2: string with id of the second entity to consider.
    Return:
        - feats: list of features extracted from the tree and e1, e2.
    """
    feats = []
    # Get entity nodes from tree
    n1 = get_entity_node(analysis, entities, e1)
    n2 = get_entity_node(analysis, entities, e2)
    # Get verb ancestor from entities
    v1 = get_verb_ancestor(analysis, n1)
    v2 = get_verb_ancestor(analysis, n2)

    # Binary feature: e1 -conj-> e2
    e1_conj = n1["deps"]["conj"][0] if len(n1["deps"]["conj"]) else -1
    e1_conj_e2 = e1_conj == n2["address"]
    # Binary feature: e1 <-conj- e2
    e2_conj = n2["deps"]["conj"][0] if len(n2["deps"]["conj"]) else -1
    e2_conj_e1 = e2_conj == n1["address"]

    # Binary feature: same verb ancestor
    same_vb = v1["address"] == v2["address"]

    # TODO: implement more

    # Gather variables
    feats = [
        e1_conj_e2, e2_conj_e1,
        same_vb
    ]
    # Turn boolean to str 1/0
    feats = [str(int(f)) for f in feats]
    return feats


def output_features(id, e1, e2, type, features, out):
    """
    Output Features.
    Function which outputs to the given opened file object the entity pair
    specified with the features extracted from their sentence.
    Args:
        - id: string with sentence id.
        - e1: string with id of the first entity to consider.
        - e2: string with id of the second entity to consider.
        - type: string with gold class of DDI, for use in training.
        - features: list of extracted features from sentence tree.
        - outf: file object with opened file for writing output features.
    """
    feature_str = "\t".join(features)
    txt = f"{id}\t{e1}\t{e2}\t{type}\t{feature_str}\n"
    out.write(txt)


def output_ddi(id, id_e1, id_e2, type, outf):
    """
    Output DDI.
    Function which prints DDI detected to outuput file handle.
    Args:
        - id: string with sentence id.
        - id_e1: string with id of the first entity in the pair.
        - id_e2: string with id of the second entity in the pair.
        - type: string with predicted class of DDI.
        - outf: file object with opened file for writing output to.
    """
    is_ddi = "1" if type != "null" else "0"
    txt = f"{id}|{id_e1}|{id_e2}|{is_ddi}|{type}\n"
    outf.write(txt)


def get_features_labels(input):
    """
    Get Features & Labels.
    Function which opens the given filename and extracts the feature and label
    vectors, togehter with the sentence and pair entities ids.
    Args:
        - input: string with filename of file to extract features from.
    Returns:
        - ids: list of lists with sentence id and entity pairs ids.
        - feats: list of lists with binary feature vector.
        - labels: list of labels for each entity pair, for the trainer to use.
    """
    with open(input, "r") as fp:
        lines = fp.read()
    pairs = [sent.split("\t") for sent in lines.split("\n")[:-1]]
    ids = []
    labels = []
    feats = []
    for p in pairs:
        ids.append((p[0], p[1], p[2]))
        labels.append(p[3])
        feat = [int(elem) if elem.isdigit() else elem for elem in p[4:]]
        feats.append(feat)
    return ids, feats, labels


def build_features(inputdir, outputfile):
    """
    Build Features.
    Function which calls for each pair the extract_features function and
    outputs the resulting features to and opened outputfile.
    Args:
        - inputdir: string with folder containing original XML.
        - outputfile: string with file name to save features to.
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
                is_ddi = p.attributes["ddi"].value == 'true'
                try:
                    type = p.attributes["type"].value if is_ddi else "null"
                except KeyError:
                    # DDI Pairs which have ddi='true' but no type
                    # We consider them no DDI.
                    type = "null"
                feats = extract_features(analysis, entities, id_e1, id_e2)
                output_features(sid, id_e1, id_e2, type, feats, outf)
    # Close outputfile
    outf.close()


def learner(model, feature_input, output_fn):
    """
    Learner.
    Function which calls the learner with a given feature filename and an
    output filename to save model to.
    Args:
        - model: string with model type to use.
        - feature_input: string with filename of the file to extract features
            from to fit the model.
        - output_fn: string with filename of output file for trained model.
    """
    if model == "MaxEnt":
        # MaxEnt learner flow
        megam_features = f"{tmp_path}/megam_train_features.dat"
        megam_model = f"{output_fn}.megam"
        system(f"cat {feature_input}  | cut -f4- > \
            {megam_features}")
        system(f"./{megam} -quiet -nc -nobias multiclass \
            {megam_features} > {megam_model}")

    elif model == "RandomForest":
        _, x, y = get_features_labels(feature_input)
        # Create RF instance
        model = RandomForestClassifier(random_state=random_seed)
        # Train RF instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.randomForest", "wb") as fp:
            pickle.dump(model, fp)

    else:
        print(f"[ERROR] Model {model} not implemented")
        raise NotImplementedError


def classifier(model, feature_input, model_input, outputfile):
    """
    Classifier.
    Function which retrived a trainer model and predicts the output for a given
    validation set features file, to print output to another file.
    Args:
        - model: string with model type to use.
        - feature_input: string with filename of the file to extract features
            from to validate the model.
        - outputfile: string with filename of output file for validation
            predictions.
    """
    # Retrieve sentences, entities and feature vectos
    ids, x, _ = get_features_labels(feature_input)
    if model == "MaxEnt":
        # MaxEnt classifier flow
        megam_features = f"{tmp_path}/megam_valid_features.dat"
        megam_predictions = f"{tmp_path}/megam_predictions.dat"
        system(f"cat {feature_input} | cut -f4- > \
            {megam_features}")
        system(f"./{megam} -nc -nobias -predict {model_input}.megam multiclass\
            {megam_features} > {megam_predictions}")
        with open(megam_predictions, "r") as fp:
            lines = fp.readlines()
        predictions = [line.split("\t")[0] for line in lines]

    elif model == "RandomForest":
        # Retrieve model
        with open(f"{model_input}.randomForest", "rb") as fp:
            model = pickle.load(fp)
        # Predict classes
        predictions = model.predict(x)

    else:
        print(f"[ERROR] Model {model} not implemented")
        raise NotImplementedError

    # Ouput entites for each sentence
    with open(outputfile, "w") as outf:
        for (id, id_e1, id_e2), type in zip(ids, predictions):
            output_ddi(id, id_e1, id_e2, type, outf)


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


def main(model, train_input_fn, valid_input_fn, train_features_fn,
         valid_features_fn, ml_model_fn, outputfile):
    """
    Main function.
    Function that runs the module workflow for the DDI baseline, with rule
    based detection of DDI.
    Args:
        - inputdir: string with folder containing original XML.
        - outputfile: string with file name to save output to.
    """
    # Run train_features
    build_features(train_input_fn, train_features_fn)
    # Train model
    learner(model, train_features_fn, ml_model_fn)
    # Run validation features
    build_features(valid_input_fn, valid_features_fn)
    # Predict validation
    classifier(model, valid_features_fn, ml_model_fn, outputfile)
    # Evaluate prediciton
    evaluate(valid_input_fn, outputfile)


if __name__ == "__main__":
    main(
        model,
        train_input_fn,
        valid_input_fn,
        train_features_fn,
        valid_features_fn,
        ml_model_fn,
        outputfile
    )
