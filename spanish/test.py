
with open("../../data/es.train.txt","r") as fd:
    for line in fd:
        if line.startswith("#begin document"):
            print(line)
