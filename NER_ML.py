"""
"""
# !/usr/bin/python3
from nltk.tokenize.regexp import WhitespaceTokenizer as Tokenizer
from os import makedirs, listdir, system
from os.path import exists as path_exists
from re import sub as reg_sub, match
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from xml.dom.minidom import parse
import re
import numpy as np
import pickle
import pycrfsuite
# Reference constants
MODELS = ["CRF", "MaxEnt", "RandomForest"]
LABELS = ["B-drug", "I-drug", "B-drug_n", "I-drug_n", "B-brand", "I-brand",
          "B-group", "I-group", "O"]
# Global variables to control script flow
tmp_path = "data/tmp"
model = "RandomForest"
# Assign output file for entities
if not path_exists(tmp_path):
    makedirs(tmp_path)
    print(f"[INFO] Created a new folder {tmp_path}")
# Training/Validation features files
train_input_fn = "data/Train"
valid_input_fn = "data/Test-NER"
train_features_fn = f"{tmp_path}/ML_train_features.txt"
valid_features_fn = f"{tmp_path}/ML_valid_features.txt"
# Model path to save it
ml_model_fn = f"{tmp_path}/ML_model"
# Specify local megam file
megam = "resources/megam_i686.opt"
# Random forest params
random_seed = 42


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
    text = reg_sub(r"[,.:;'\"]", " ", s)
    tokenizer = Tokenizer()
    spans = tokenizer.span_tokenize(text)
    tokens = tokenizer.tokenize(text)
    tokens = [(t, s[0], s[1] - 1) for t, s in zip(tokens, spans)]
    return tokens


def extract_features(token_list):
    """
    Extract Features
    Function to extract features from each token of the given token list.
    Args:
        - token_list: list of token strings with token words
    Returns:
        - features: list of list of features for each token of the given list.
    """
    features = []
    for i, token_t in enumerate(token_list):
        token, start, end = token_t
        # Token form
        form = f"form={token.lower()}"
        # Suffix's 4 last letters
        suf4 = token[-4:].lower()
        suf4 = f"suf4={suf4}"
        # Suffix's 3 last letters
        suf3 = token[-3:]
        suf3 = f"suf3={suf3}"
        # Suffix's 2 last letters
        suf2 = token[-2:]
        suf2 = f"suf2={suf2}"
        # Prefix's 4 first letters
        pre4 = token[:4]
        pre4 = f"pre4={pre4}"
        # Prefix's 3 first letters
        pre3 = token[:3]
        pre3 = f"pre3={pre3}"
        # Prefix's 2 first letters
        pre2 = token[:2]
        pre2 = f"pre2={pre2}"
        # Prev token
        if i == 0:
            prev = "prev=_BoS_"
        else:
            prev = f"prev={token_list[i - 1][0].lower()}"
        # Next token
        if i == (len(token_list) - 1):
            nxt = "next=_EoS_"
            nxt_end = nxt
        else:
            nxt = f"next={token_list[i + 1][0].lower()}"
            # Next token end
            nxt_end = f"next={token_list[i + 1][0][-3:-1]}"
        # All token in capital letters
        capital_num = str(int(token.isupper()))
        capital = f"capital={capital_num}"
        # Begin with capital letter
        b_capital_num = str(int(token[0].isupper()))
        b_capital = f"b_capital={b_capital_num}"
        # Ends s for plurals
        ends_s_num = str(int(token.endswith('s')))
        ends_s = f"ends_s={ends_s_num}"
        # Number of has spaces in token
        # Number of digits in token
        digits = f"digits={sum(i.isdigit() for i in token)}"
        # Number of capitals in token
        capitals = f"capitals={sum(i.isupper() for i in token)}"
        # Number of hyphens in token
        hyphens = f"hyphens={sum(['-' == i for i in token])}"
        # Number of symbols in token
        symbols = f"symbols={len(re.findall(r'[()+-]', token))}"
        # Token length
        length = f"length={len(token)}"
        # Token has Digit-Captial combination
        dig_cap_num = str(int(bool(re.compile("([A-Z]+[0-9]+.*)").match(token) or re.compile(
            "([0-9]+[A-Z]+.*)").match(token))))
        dig_cap = f"dig_cap={dig_cap_num}"
        # Feats list
        if model == "MaxEnt":
            feats = [form, pre2, pre3, pre4, suf2, suf4]
        elif model == "CRF":
            # Minimum entities to reach Goal 3
            feats = [form, capital, nxt, pre2, suf2, prev,
                     capitals,
                     # Entities tu reach the maximum F1
                     pre3, pre4, suf4, dig_cap, hyphens, length
                     ]
        elif model == "RandomForest":
            # Entities to reach Goal 3
            feats = [suf2, pre2, nxt_end, b_capital, capital, dig_cap,
                     capitals[-1], digits[-1], hyphens[-1], symbols[-1], length[-1]]
        else:
            feats = [form, b_capital, ends_s, capital, dig_cap,
                     nxt, pre2, pre3, pre4, prev, suf2, suf3, suf4,
                     capitals, digits, hyphens, symbols, length]
        features.append(feats)
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
        # Turn back numeric variables
        # only for RandomForest model
        if model == "RandomForest":
            for i, token in enumerate(x):
                val = [int(elem) if elem.isdigit() else elem for elem in token]
                x[i] = val
        y = [f[4] for f in feats if len(f)]
        full_tokens.append(feats)
        X_feat.append(x)
        Y_feat.append(y)
    return full_tokens, X_feat, Y_feat


def output_entities(id, tokens, classes, outf):
    """
    """

    ind = 0
    while ind < len(tokens):
        tag = classes[ind]
        type = tag.split("-")[-1]
        if tag == "O":
            ind += 1
            continue
        elif "B" in tag:  # If Beginning of an entity
            name, start, end = tokens[ind]
            # Check if next token I-same_type
            # Continue search until EoS or no-match
            ind += 1
            tag_nxt = classes[ind] if ind < len(tokens) else "O"
            type_nxt = tag_nxt.split("-")[-1]
            while ind < len(tokens) and "I" in tag_nxt and type_nxt == type:
                name_nxt, _, end_nxt = tokens[ind]
                name = f"{name} {name_nxt}"
                end = end_nxt
                ind += 1
                tag_nxt = classes[ind] if ind < len(tokens) else "O"
                type_nxt = tag_nxt.split("-")[-1]
        else:  # I-tag
            name, start, end = tokens[ind]
            ind += 1
        # Print entity and continue
        offset = f"{start}-{end}"
        txt = f"{id}|{offset}|{name}|{type}\n"
        outf.write(txt)


def learner(model, feature_input, output_fn):
    """
    """
    _, X_train, Y_train = get_sentence_features(feature_input)
    if model == "CRF":
        # CRF learner flow
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, Y_train):
            trainer.append(xseq, yseq)
        trainer.train(f"{output_fn}.crfsuite")

    elif model == "MaxEnt":
        # MaxEnt learner flow
        megam_features = f"{tmp_path}/megam_train_features.dat"
        megam_model = f"{output_fn}.megam"
        system(f"cat {feature_input} | cut -f5- | grep -v ’^$’ > \
            {megam_features}")
        system(f"./{megam} -quiet -nc -nobias multiclass \
            {megam_features} > {megam_model}")

    elif model == "RandomForest":
        # Unlist sentences
        x_cat = []
        x_num = []
        y = []
        for x_sent, y_sent in zip(X_train, Y_train):
            x_cat_sent = [f[:6] for f in x_sent]
            x_num_sent = [f[6:] for f in x_sent]
            x_cat.extend(x_cat_sent)
            x_num.extend(x_num_sent)
            y.extend(y_sent)
        # One hot encoder to turn categorical variables to binary
        encoder = OneHotEncoder(handle_unknown="ignore")
        encoder.fit(x_cat)
        x_encoded = encoder.transform(x_cat).toarray()
        x = np.concatenate((x_encoded, x_num), axis=1)
        model = RandomForestClassifier(random_state=random_seed)
        model.fit(x, y)
        # Save model to pickle
        with open(f"{output_fn}.randomForest", "wb") as fp:
            pickle.dump([model, encoder], fp)

    else:
        print(f"[ERROR] Model {model} not implemented")
        raise NotImplementedError


def classifier(model, feature_input, model_input, outputfile):
    sentences, X_valid, Y_valid = get_sentence_features(feature_input)
    if model == "CRF":
        # CRF classifier flow
        tagger = pycrfsuite.Tagger()
        tagger.open(f"{model_input}.crfsuite")
        predictions = [tagger.tag(x) for x in X_valid]

    elif model == "MaxEnt":
        # MaxEnt classifier flow
        megam_features = f"{tmp_path}/megam_valid_features.dat"
        megam_predictions = f"{tmp_path}/megam_predictions.dat"
        system(f"cat {feature_input} | cut -f5- | grep -v ’^$’ > \
            {megam_features}")
        system(f"./{megam} -nc -nobias -predict {model_input}.megam multiclass\
            {megam_features} > {megam_predictions}")
        with open(megam_predictions, "r") as fp:
            lines = fp.readlines()
        pred_classes = [line.split("\t")[0] for line in lines]
        predictions = []
        start = 0
        for sent in X_valid:
            end = start + len(sent)
            predictions.append(pred_classes[start:end])
            start = end

    elif model == "RandomForest":
        with open(f"{model_input}.randomForest", "rb") as fp:
            model, encoder = pickle.load(fp)
        # Unlist sentences
        x_cat = []
        x_num = []
        for x_sent in X_valid:
            x_cat_sent = [f[:6] for f in x_sent]
            x_num_sent = [f[6:] for f in x_sent]
            x_cat.extend(x_cat_sent)
            x_num.extend(x_num_sent)
        # One hot encoder to turn categorical variables to binary
        x_encoded = encoder.transform(x_cat).toarray()
        x = np.concatenate((x_encoded, x_num), axis=1)
        pred_classes = model.predict(x)
        predictions = []
        start = 0
        for sent in X_valid:
            end = start + len(sent)
            predictions.append(pred_classes[start:end])
            start = end

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
    outputfile = f"{tmp_path}/task9.1_ML_{model}_1.txt"
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
