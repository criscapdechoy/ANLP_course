
from xml.dom.minidom import parse, parseString
import nltk
from os import listdir
from os.path import isfile, join
import os

def main():
    # Input parameters
    s = input()
    cd = os.getcwd()
    print( cd + s )
    if os.path.isdir( cd + s ):
        directory = open( cd + s, "r+")
    output_file = 'output.txt'
    '''
    # Open xml files
    listfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
    # Get info from xml files
    print( listfiles )
    datasource = open(directory)
    dom = parse(datasource)  # parse an open file'''

def nerc(inputdir, outputfile):
    '''   for file in inputdir:
            tree = parseXML(file)

    for sentence in tree:
        (id, text) = get
        sentence
        info(sentence)
    token
    list = tokenize(text)
    entities = extract
    entities(token list)
    output
    entities(id, entities, outputfile)
    evaluate(inputdir, outputfile)'''

if __name__ == '__main__':
    main()
