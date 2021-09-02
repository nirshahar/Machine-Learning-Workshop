from rdkit import Chem

filenames = ["agg.txt", "control_non_aggregates.txt"]


def convert_file(current):
    cnt = 0
    with open(current) as f:
        if 'control' in current:
            index = 1
        else:
            index = 0

        with open(current[:-4] + "_canonical.txt",  'a') as newf:
            for line in f:
                line = line.split()
                line = line[index]
                try:
                    converted = Chem.MolToSmiles(Chem.MolFromSmiles(line))
                    newf.write(converted+"\n")
                except:
                    cnt += 1
                    print("failure", cnt)


for file in filenames:
    convert_file(file)

print("Done converting")