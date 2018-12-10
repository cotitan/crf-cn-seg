import sys
import argparse

parse = argparse.ArgumentParser()

parse.add_argument("--input_file", type=str, default="data/pku_training.utf8", help="")
parse.add_argument("--output_file", type=str, default="data/train", help="")
parse.add_argument("--tag_schema", type=str, default="bmes", help="bi/bmes")
parse.add_argument("--encoding", type=str, default="utf8", help="input file encoding")

args = parse.parse_args()

fin = open(args.input_file, encoding=args.encoding)
fout = open(args.output_file + "." + args.tag_schema, "w")

for i, line in enumerate(fin):
    if line.strip() == "":
        continue
    else:
        words = line.strip().split()
        if args.tag_schema == "bi":
            for word in words:
                fout.write(word[0] + " B\n")
                for ch in word[1:]:
                    fout.write(ch + " I\n")
        elif args.tag_schema == "bmes":
            for word in words:
                if len(word) == 1:
                    fout.write(word + " S\n")
                else:
                    fout.write(word[0] + " B\n")
                    for ch in word[1:-1]:
                        fout.write(ch + " M\n")
                    fout.write(word[-1] + " E\n")
        fout.write("\n")

fout.close()
fin.close()
