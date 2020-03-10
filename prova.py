id=0
tokens=[['a',0,3],['b',10,13],['c',20,23],['d',30,33],['cc',40,43]]
classes=['O','B-a','I-a','B-b','B-c']
outf=''

def output_entities(id, tokens, classes, outf):
    """
    """

    def find_entities(ind, name0='', start0=float('inf'), end0=-1, flag=False):
        token, tag = tokens[ind], classes[ind]
        name, start, end = token
        ind += 1
        print(ind, flag, token)

        if ("B" in tag) & (flag == False):
            print('B', ind, name, start, end, flag)
            name, start, end, ind, tag = find_entities(ind, name0=name, start0=start, end0=end, flag=True)
        elif ("I" in tag):
            print('I', ind, name, start, end, flag)
            name = name0 + name
            start = min(start0, start)
            end = max(end0, end)
            name, start, end, ind, tag = find_entities(ind, name0=name, start0=start, end0=end, flag=True)
        elif flag:
            print('finishing', ind, name, start, end, flag)
            ind = ind - 1
            print('going back to save ', type(name0), type(start0), type(end0), type(ind - 1), type(classes[ind - 1]))
            return 1,2,3,4,5

           # return name0, start0, end0, ind, tag
        elif flag == False:
            print('flag False')
            return name, start, end, ind, classes[ind]

    ind = 0
    while ind < len(tokens):
        if classes[ind] == "O":
            ind += 1
            continue
        else:
            print('gotten', find_entities(ind))
            name, start, end, ind, tag = find_entities(ind)


        offset = f"{start}-{end}"
        types = tag.split("-")[1]
        txt = f"{id}|{offset}|{name}|{types}\n"
        print(txt)


output_entities(id, tokens, classes, outf)
