
import sys

from utils.io_utils import * 

def main():

    scores_filepath = sys.argv[1]
    scores = read_json(scores_filepath)

    for name, candidate_set in scores["candidate_sets"].items():

        tokens = candidate_set["tokens"]

        # compute fraction of tokens with positive mean_dot 
        pos_count = 0 
        length = 0
        for token in tokens:
            if token["mean_dot"] is not None:
                if token["mean_dot"] > 0:
                    pos_count += 1
                length += 1
        fraction_positive = pos_count / length if length else 0
        print(f"{name}: {fraction_positive:.4f}")

if __name__ == "__main__":
	main()