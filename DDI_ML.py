"""
DDI Machine Learning Script.
Python scripts that contains functions to train and evaluate a ML model to
detect Drug-Drug Interaction.
"""
# !/usr/bin/python3
# from nltk import jaccard_distance, edit_distance
from nltk.parse.corenlp import CoreNLPDependencyParser
from os import listdir, system, path, makedirs
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.svm import SVC
from sys import exit
from tqdm import tqdm
from xml.dom.minidom import parse
import pickle

# Reference constants
MODELS = ["MaxEnt", "MLP", "SVC", "GBC", "LR"]

# Global variables and procedures
tmp_path = "data/tmp"
if not path.exists(tmp_path):  # Create dir if not exists
    makedirs(tmp_path)
    print(f"[INFO] Created a new folder {tmp_path}")
model = "MaxEnt"
# model = "MLP"
# model = "SVC"
# model = "GBC"
# model = "LR"

train_input_fn = "data/Train"
valid_input_fn = "data/Devel"
# valid_input_fn = "data/Test-DDI"
train_features_fn = f"{tmp_path}/DDI_ML_train_features.txt"
valid_features_fn = f"{tmp_path}/DDI_ML_valid_features.txt"
# valid_features_fn = f"{tmp_path}/DDI_ML_test_features.txt"
outputfile = f"{tmp_path}/task9.2_ML_{model}_1.txt"

# Model path to save it
ml_model_fn = f"{tmp_path}/DDI_ML_model"
# Specify local megam file
megam = "resources/megam_i686.opt"

random_seed = 10
# MaxEnt params
feat_col = "4-"  # 0.3777
feat_col = "4-10,12,16-46,48"  # 0.3777
# feat_col = "4-10,17-27,32,34-46,49,53,58,61-65,76,84,85"  # 0.3901
# MLP params
hidden_layer_sizes = (45,)
alpha = 1
activation = "relu"
solver = "adam"
n_epochs = 100
early_stopping = False
verbose = False
# SVC
svc_C = 20
kernel = "rbf"
gamma = "scale"
# GBC params
n_estimators = 50
# LR params
C = 1e6
multi_class = 'ovr'
penalty = 'l2'
max_iter = 1000
lr_solver = 'lbfgs'
n_jobs = -1
lr_verbose = 0

# Dict global tags
dict_tags = {
    'JJ': 'JJ', 'JJR': 'JJ', 'JJS': 'JJ',
    'NN': 'NN', 'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN',
    'PRP': 'PRP', 'PRP$': 'PRP',
    'RB': 'RB', 'RBS': 'RB', 'RBR': 'RB',
    'VB': 'VB', 'VBD': 'VB', 'VBG': 'VB', 'VBN': 'VB', 'VBP': 'VB',
    'VBZ': 'VB',
    'WP': 'WP', 'WP$': 'WP',
    'CC': 'CC', 'CD': 'CF', 'DT': 'DT', 'EX': 'EX', 'FW': 'FW', 'IN': 'IN',
    'LS': 'LS', 'MD': 'MD',
    'PDT': 'DT', 'POS': 'POS',
    'RP': 'RP', 'SYS': 'SYS', 'TO': 'TO', 'UH': 'UH', 'WDT': 'DT',
    'WRB': 'WRB',
    'SYM': 'SYM', '-LRB-': '-LRB-', ',': ',', ':': ':', 'null': 'null'
}

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
    ents = entities[entity]["text"].split()
    # Capture possible tree nodes containing or that are contained in entity
    possible = sorted(
        [node for node in nodes if node["word"] is not None and
         any(ent in node["word"] for ent in ents)],
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


def get_dependency_address(node, dependency):
    """
    Get Dependency Address.
    Function which returns the address of a given dependency for a given node,
    or a non tractable value -1, which always evaluates to False in the
    features. To use when extracting features.
    Args:
        - node: dictionary with node to look dependencies from.
        - dependency: string with dependency name to look for in node.
    Return:
        - _: string with address of found dependency, or -1 if not found.
    """
    dep = node["deps"][dependency]
    # If dependency exists, return address
    # If dependency does not exist, return non-value
    return dep[0] if len(dep) else -1


def check_lemmas(analysis, lemmas):
    """
    Check Lemmas.
    Function which checks if the words in the sentence contain the given
    lemmas. Then returns the tree-higher encountered lemma, or "null" if none
    found.
    Args:
        - analysis: DependencyTree object instance with sentence analysis.
        - lemmas: list of strings with lemmas to check.
    Returns:
        - _: string with present lemma or None.
    """
    nds = analysis.nodes
    present = [nds[n] for n in nds
               if (nds[n]["word"] is not None and nds[n]["lemma"] in lemmas)]
    present = sorted(present, key=lambda x: x["head"])
    # return present[0]["lemma"] if len(present) else "null"
    return "True" if len(present) else "False"


def get_ancestors(analysis, node):
    """
    Get Ancestors.
    Function which returns the given node's ancestor nodes.
    Args:
        - analysis: DependencyTree object instance with sentence analysis.
        - node: dictionary with node to start from.
    Return:
        - node: dictionary with verb antecessor node from DependencyTree.
    """
    ancs = []
    nds = analysis.nodes
    while node["tag"] and node["tag"] != "TOP":
        ancs.append(node)
        node = nds[node["head"]]
    return ancs


def extract_features(analysis, entities, e1, e2):
    """
    Extract Features.
    Function which receives an analyzed sentence tree, the entities
    present in the sentence, and the ids of the two target entities and returns
    a list of features to pass to a ML model to predict DDI.
    Args:
        - analysis: DependencyGraph object instance with sentence parsed
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

    # Get ancestors nodes list for entity nodes and verb nodes
    ance1 = get_ancestors(analysis, n1)
    ance2 = get_ancestors(analysis, n2)
    ancev1 = get_ancestors(analysis, v1)
    ancev2 = get_ancestors(analysis, v2)

    # DDI-type characteristic lemmas
    advise_lemmas = ["administer", "use", "recommend", "consider", "approach",
                     "avoid", "monitor", "advise", "require", "contraindicate"]
    effect_lemmas = ["increase", "report", "potentiate", "enhance", "decrease",
                     "include", "result", "reduce", "occur", "produce",
                     "prevent", "effect"]
    int_lemmas = ["interact", "suggest", "report", "occur", "interfere",
                  "identify", "pose"]
    mechanism_lemmas = ["increase", "decrease", "result", "report", "expect",
                        "reduce", "inhibit", "show", "interfere", "cause",
                        "indicate", "demonstrate"]
    # advise_lemmas = ["should", "must", "may", "recommend", "caution"]
    # effect_lemmas = [
    # "produce", "administer", "potentiate", "prevent", "effect"]
    int_lemmas = ["interact", "interaction"]
    mechanism_lemmas = ["reduce", "increase", "decrease"]
    # Mix lemmas
    mix_lemmas = list(set(
        advise_lemmas + effect_lemmas + int_lemmas + mechanism_lemmas))
    # Modal verbs lemmas
    modal_vb = ["can", "could", "may", "might", "must", "will", "would",
                "shall", "should"]

    # Modal verbs and DDI-type lemmas present in sentence
    modal_present = check_lemmas(analysis, modal_vb)
    lemma_present = check_lemmas(analysis, mix_lemmas)
    advise_present = check_lemmas(analysis, advise_lemmas)
    effect_present = check_lemmas(analysis, effect_lemmas)
    int_present = check_lemmas(analysis, int_lemmas)
    mechanism_present = check_lemmas(analysis, effect_lemmas)

    # e2<-*-VB is part DDI-type lemmas
    advise_v1 = True if v1["lemma"] in advise_lemmas else "null"
    effect_v1 = True if v1["lemma"] in effect_lemmas else "null"
    int_v1 = True if v1["lemma"] in int_lemmas else "null"
    mechanism_v1 = True if v1["lemma"] in mechanism_lemmas else "null"
    # e2<-*-VB is part DDI-type lemmas
    advise_v2 = True if v2["lemma"] in advise_lemmas else "null"
    effect_v2 = True if v2["lemma"] in effect_lemmas else "null"
    int_v2 = True if v2["lemma"] in int_lemmas else "null"
    mechanism_v2 = True if v2["lemma"] in mechanism_lemmas else "null"

    # Check if entities hang from the same verb
    # v1_lemma = v1["lemma"]
    # v2_lemma = v2["lemma"]
    v1_equal_v2 = v1 == v2

    # Get head dependencies
    e1_rel = n1["rel"]
    e2_rel = n2["rel"]
    v1_rel = v1["rel"]
    v2_rel = v2["rel"]

    # Get node dependencies
    e1_deps = "_".join(n1["deps"].keys()) if len(n1["deps"]) else "null"
    e2_deps = "_".join(n2["deps"].keys()) if len(n2["deps"]) else "null"
    v1_deps = "_".join(v1["deps"].keys()) if len(v1["deps"]) else "null"
    v2_deps = "_".join(v2["deps"].keys()) if len(v2["deps"]) else "null"
    ance1_deps = "_".join([a["rel"] for a in ance1]) if len(ance1) else "null"
    ance2_deps = "_".join([a["rel"] for a in ance2]) if len(ance2) else "null"

    # Get node order
    e1_over_e2 = n1 in ance2
    v1_over_v2 = v1 in ancev2
    v2_over_v1 = v2 in ancev1

    # Common ancestor features
    common = ([n for n in ance1 if n in ance2] if len(ance1) > len(ance2) else
              [n for n in ance2 if n in ance1])
    common_rel = common[0]["rel"] if len(common) else "null"
    common_deps = ("_".join(common[0]["deps"].keys())
                   if len(common) and len(common[0]["deps"]) else "null")
    common_tag = common[0]["tag"] if len(common) else "null"
    common_tag = dict_tags[common_tag]
    common_dist_root = (len(ance1) - 1 - ance1.index(common[0])
                        if len(common) else 99)
    common_dist_e1 = ance1.index(common[0]) if len(common) else 99
    common_dist_e2 = ance2.index(common[0]) if len(common) else 99

    # Common ancestor son's rel for each entity's branch
    common_dep11_rel = (
        ance1[ance1.index(common[0]) - 1]["rel"]
        if len(common) and ance1.index(common[0]) > 0 else "null")
    common_dep12_rel = (
        ance1[ance1.index(common[0]) - 2]["rel"]
        if len(common) and ance1.index(common[0]) > 1 else "null")
    common_dep13_rel = (
        ance1[ance1.index(common[0]) - 3]["rel"]
        if len(common) and ance1.index(common[0]) > 2 else "null")
    common_dep21_rel = (
        ance2[ance2.index(common[0]) - 1]["rel"]
        if len(common) and ance2.index(common[0]) > 0 else "null")
    common_dep22_rel = (
        ance2[ance2.index(common[0]) - 2]["rel"]
        if len(common) and ance2.index(common[0]) > 1 else "null")
    common_dep23_rel = (
        ance2[ance2.index(common[0]) - 3]["rel"]
        if len(common) and ance2.index(common[0]) > 2 else "null")

    # Common ancestor son's tag for each entity's branch
    common_dep11_tag = (
        dict_tags[ance1[ance1.index(common[0]) - 1]["tag"]]
        if len(common) and ance1.index(common[0]) > 0 else "null")

    common_dep22_tag = (
        dict_tags[ance2[ance2.index(common[0]) - 2]["tag"]]
        if len(common) and ance2.index(common[0]) > 1 else "null")

    # # Tree address features
    # # e1<-conj-x<-dobj-VB-nmod->e2
    # e2_nmod = get_dependency_address(v2, "nmod") == n2["address"]
    # x_dobj = get_dependency_address(v1, "dobj")
    # nx = analysis.nodes[x_dobj] if x_dobj != -1 else v1
    # e1_conj_dobj = get_dependency_address(nx, "conj") == n1["address"]
    # e1_conj_dobj_nmod_e2 = e1_conj_dobj and e2_nmod

    # NER features
    # # Jackard dist and Edit dist
    # try:
    #     jaccard_dist = round(jaccard_distance(set(n1["lemma"]),
    #                                           set(n2["lemma"])) * 10, 0)
    #     edit_dist = edit_distance(n1["lemma"], n2["lemma"])
    #     edit_dist = round(edit_dist * 10 / (1 + edit_dist), 0)
    # except Exception:
    #     jaccard_dist = 10
    #     edit_dist = 10
    # Entity lemma features
    lemma1 = str(n1["lemma"])
    lemma2 = str(n2["lemma"])
    # # 2-Prefix/Suffix from lemma
    # pre2_1 = lemma1[:2].lower()
    # pre2_2 = lemma2[:2].lower()
    # suf2_1 = lemma1[-2:].lower()
    # suf2_2 = lemma2[-2:].lower()
    # 3-Prefix/Suffix from lemma
    pre3_1 = lemma1[:3].lower()
    pre3_2 = lemma2[:3].lower()
    suf3_1 = lemma1[-3:].lower()
    suf3_2 = lemma2[-3:].lower()
    # # Number of capitals in token
    # capitals2 = sum(i.isupper() for i in lemma2)

    # Gather variables
    feats = [
        modal_present,  # 5
        lemma_present,
        advise_present,
        effect_present,
        int_present,
        mechanism_present,  # 10
        advise_v1,
        effect_v1,
        int_v1,
        mechanism_v1,
        advise_v2,  # 15
        effect_v2,
        int_v2,
        mechanism_v2,
        v1_equal_v2,
        # v1_lemma,
        # v2_lemma,  # 15
        e1_rel,
        e2_rel,
        v1_rel,
        v2_rel,
        e1_deps,  # 20
        e2_deps,
        e1_over_e2,
        v1_over_v2,
        v2_over_v1,
        common_rel,  # 25
        common_tag,
        common_dist_root,
        common_dist_e1,
        common_dist_e2,
        common_deps,  # 30
        common_dep11_rel,
        common_dep12_rel,
        common_dep13_rel,
        common_dep21_rel,
        common_dep22_rel,  # 35
        common_dep23_rel,
        common_dep11_tag,
        common_dep22_tag,
        v1_deps,
        v2_deps,  # 40
        ance1_deps,
        ance2_deps,
        # e1_conj_dobj_nmod_e2,
        # jaccard_dist,
        # edit_dist,  # 45
        # pre2_1,
        # pre2_2,
        # suf2_1,
        # suf2_2,
        pre3_1,
        pre3_2,
        suf3_1,
        suf3_2,
    ]
    # Turn variables f to categorical var_i=f
    feats = [f"var_{i}={f}" for i, f in enumerate(feats)]
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
        feat = [elem.split("=")[1] for elem in p[4:]]
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
    for f in tqdm(files):
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
        system(f"cat {feature_input}  | cut -f {feat_col} > \
            {megam_features}")
        system(f"./{megam} -quiet -nc -nobias multiclass \
            {megam_features} > {megam_model}")

    elif model == "MLP":
        _, x_cat, y = get_features_labels(feature_input)
        # OneHotEncode variables
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x = encoder.transform(x_cat)
        # Create RF instance
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            max_iter=n_epochs,
            early_stopping=early_stopping,
            random_state=random_seed,
            verbose=verbose)
        # Train RF instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.MLP", "wb") as fp:
            pickle.dump([model, encoder], fp)

    elif model == "SVC":
        _, x_cat, y = get_features_labels(feature_input)
        # OneHotEncode variables
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x = encoder.transform(x_cat)
        # Create RF instance
        model = SVC(
                C=svc_C,
                kernel=kernel,
                gamma=gamma,
                random_state=random_seed
            )
        # Train RF instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.SVC", "wb") as fp:
            pickle.dump([model, encoder], fp)

    elif model == "GBC":
        _, x_cat, y = get_features_labels(feature_input)
        # OneHotEncode variables
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x = encoder.transform(x_cat)
        # Create RF instance
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            random_state=random_seed)
        # Train RF instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.GBC", "wb") as fp:
            pickle.dump([model, encoder], fp)

    elif model == "LR":
        _, x_cat, y = get_features_labels(feature_input)
        # OneHotEncode variables
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x = encoder.transform(x_cat)
        # Create LR instance
        model = LR(
            C=C,
            multi_class=multi_class,
            penalty=penalty,
            max_iter=max_iter,
            solver=lr_solver,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=lr_verbose)
        # Train RF instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.LR", "wb") as fp:
            pickle.dump([model, encoder], fp)

    elif model == "KNC":
        _, x_cat, y = get_features_labels(feature_input)
        # OneHotEncode variables
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x = encoder.transform(x_cat)
        # Create KNC instance
        model = KNC()
        # Train KNC instance
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.KNC", "wb") as fp:
            pickle.dump([model, encoder], fp)

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
        system(f"cat {feature_input} | cut -f {feat_col} > \
            {megam_features}")
        # system(f"cat {feature_input} | cut -f4- > \
        #     {megam_features}")
        system(f"./{megam} -quiet -nc -nobias -predict {model_input}.megam \
            multiclass {megam_features} > {megam_predictions}")
        with open(megam_predictions, "r") as fp:
            lines = fp.readlines()
        predictions = [line.split("\t")[0] for line in lines]

    elif model == "MLP":
        # Retrieve model
        with open(f"{model_input}.MLP", "rb") as fp:
            model, encoder = pickle.load(fp)
        # OneHotEncode variables
        x_ = encoder.transform(x)
        # Predict classes
        predictions = model.predict(x_)

    elif model == "SVC":
        # Retrieve model
        with open(f"{model_input}.SVC", "rb") as fp:
            model, encoder = pickle.load(fp)
        # OneHotEncode variables
        x_ = encoder.transform(x)
        # Predict classes
        predictions = model.predict(x_)

    elif model == "GBC":
        # Retrieve model
        with open(f"{model_input}.GBC", "rb") as fp:
            model, encoder = pickle.load(fp)
        # OneHotEncode variables
        x_ = encoder.transform(x)
        # Predict classes
        predictions = model.predict(x_)

    elif model == "LR":
        # Retrieve model
        with open(f"{model_input}.LR", "rb") as fp:
            model, encoder = pickle.load(fp)
        # OneHotEncode variables
        x_ = encoder.transform(x)
        # Predict classes
        predictions = model.predict(x_)

    elif model == "KNC":
        # Retrieve model
        with open(f"{model_input}.KNC", "rb") as fp:
            model, encoder = pickle.load(fp)
        # OneHotEncode variables
        x_ = encoder.transform(x)
        # Predict classes
        predictions = model.predict(x_)

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
    # build_features(train_input_fn, train_features_fn)
    # Train model
    learner(model, train_features_fn, ml_model_fn)
    # Run validation features
    # build_features(valid_input_fn, valid_features_fn)
    # Predict validation
    classifier(model, valid_features_fn, ml_model_fn, outputfile)
    # Evaluate prediction
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
