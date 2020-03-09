"""
"""
# !/usr/bin/python3
from nltk.tokenize.regexp import WhitespaceTokenizer as Tokenizer
from re import sub as reg_sub
from xml.dom.minidom import parse
from os import makedirs, listdir, system
from os.path import exists as path_exists
import pycrfsuite
# Reference constants
MODELS = ["CRF", "MaxEnt"]
LABELS = ["B-drug", "I-drug", "B-drug_n", "I-drug_n", "B-brand", "I-brand",
          "B-group", "I-group", "O"]
# Global variables to control script flow
tmp_path = "data/tmp"
model = "CRF"
# Assign output file for entities
if not path_exists(tmp_path):
    makedirs(tmp_path)
    print(f"[INFO] Created a new folder {tmp_path}")
# Training/Validation features files
train_input_fn = "data/Train"
valid_input_fn = "data/Devel"
train_features_fn = f"{tmp_path}/ML_train_features.txt"
valid_features_fn = f"{tmp_path}/ML_valid_features.txt"
# Model path to save it
ml_model_fn = f"{tmp_path}/ML_model"


def parseXML(file):
    """
    Parse XML file.
    Function to parse the XML file with path given as argument.

    Args:
        - file: string with path of xml file to parse.
    Returns:
        - sentences: list of xml elements of tag name "sentence".
        - entities: list of xml elements of tag name "entity".
    """
    xml = parse(file)
    sentences = xml.getElementsByTagName("sentence")
    entities = xml.getElementsByTagName("entity")
    return sentences, entities


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


def get_gold_entities(entities):
    """
    """
    ents = {}
    for ent in entities:
        text = ent.getAttribute("text").split(" ")
        type = ent.getAttribute("type")
        if len(text) > 1:
            ents[text[0]] = f"B-{type}"
            for t in text[1:]:
                ents[t] = f"I-{type}"
        else:
            ents[text[0]] = f"I-{type}"
    return ents


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
    text = reg_sub(r"[(,.:;'\")]+", " ", s)
    tokenizer = Tokenizer()
    spans = tokenizer.span_tokenize(text)
    tokens = tokenizer.tokenize(text)
    tokens = [(t, s[0], s[1]-1) for t, s in zip(tokens, spans)]
    return tokens


def extract_features(token_list):
    """
    Extract Features
    Fuction to extract features from each token of the given token list.
    Args:
        - token_list: list of token strings with token words
    Returns:
        - features: list of list of features for each token of the given list.
    """
    features = []
    for i, token_t in enumerate(token_list):
        token, start, end = token_t
        # Token form
        form = f"form={token}"
        # Suffix's 4 last letters
        suf4 = token[-4:]
        suf4 = f"suf4={suf4}"
        # Prev token
        if i == 0:
            prev = "prev=_BoS_"
        else:
            prev = f"prev={token_list[i-1][0]}"
        # Next token
        if i == (len(token_list)-1):
            nxt = "next=_EoS_"
        else:
            nxt = f"next={token_list[i+1][0]}"
        features.append([form, suf4, nxt, prev])
    return features


def output_features(id, tokens, features, gold_ents, out):
    """
    Args:
        - id: string with sentence id.
        - tokens: list of toknes for given sentence
        - ents: list of entities dictionaries with {name, offset, type}
        - gold_ents: dictionary with entities and global tag name.
        - outf: path for output file
    """
    for i, token in enumerate(tokens):
        text, start, end = token
        feature = "\t".join(features[i])
        gold = gold_ents[text] if text in gold_ents.keys() else "O"
        txt = f"{id}\t{text}\t{start}\t{end}\t{gold}\t{feature}\n"
        out.write(txt)


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
    return system(f"java -jar eval/evaluateNER.jar {inputdir} {outputfile}")


def build_features(inputdir, outputfile):
    """
    """
    files = [f"{inputdir}/{f}" for f in listdir(inputdir)]
    output = open(outputfile, "w")
    last = len(files) - 1
    for i, file in enumerate(files):
        tree, entities = parseXML(file)
        gold_ents = get_gold_entities(entities)
        for i, sentence in enumerate(tree):
            (id, text) = get_sentence_info(sentence)
            token_list = tokenize(text)
            features = extract_features(token_list)
            output_features(id, token_list, features, gold_ents, output)
            if len(token_list) > 0 and i < last:
                output.write("\n")
    output.close()
    return "DONE!"


def get_sentence_features(input):
    """
    """
    with open(input, "r") as fp:
        lines = fp.read()
    sentences = lines.split("\n\n")[:-1]
    X_feat = []
    Y_feat = []
    full_tokens = []
    for sent in sentences:
        tokens = sent.split("\n")
        feats = [token.split("\t") for token in tokens if len(token)]
        x = [f[5:] for f in feats if len(f)]
        y = [f[4] for f in feats if len(f)]
        full_tokens.append(feats)
        X_feat.append(x)
        Y_feat.append(y)
    return full_tokens, X_feat, Y_feat


def output_entities(id, tokens, classes, outf):
    """
    """
    for token, tag in zip(tokens, classes):
        if tag == "O":
            continue
        name, start, end = token
        offset = f"{start}-{end}"
        type = tag.split("-")[1]
        txt = f"{id}|{offset}|{name}|{type}\n"
        outf.write(txt)


def learner(model, feature_input, output_fn):
    """
    """
    _, X_train, Y_train = get_sentence_features(feature_input)
    if model == "CRF":
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        trainer.train(f"{output_fn}.crfsuite")
    elif model == "MaxEnt":
        pass
    else:
        print(f"[ERROR] Model {model} not implemented")
        raise NotImplementedError


def classifier(model, feature_input, model_input, outputfile):
    sentences, X_valid, Y_valid = get_sentence_features(feature_input)
    if model == "CRF":
        tagger = pycrfsuite.Tagger()
        tagger.open(f"{model_input}.crfsuite")
        predictions = [tagger.tag(x) for x in X_valid]
    elif model == "MaxEnt":
        pass
    else:
        print(f"[ERROR] Model {model} not implemented")
        raise NotImplementedError
    # Ouput entites for each sentence
    with open(outputfile, "w") as out:
        for sent, classes in zip(sentences, predictions):
            id = sent[0][0]
            tokens = [(word[1], word[2], word[3]) for word in sent if word]
            output_entities(id, tokens, classes, out)


if __name__ == "__main__":
    # Evaluation output file config
    outputfile = f"{tmp_path}/task9.1_ML_1.txt"
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
