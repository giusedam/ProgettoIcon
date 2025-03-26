import pandas as pd
from pyswip import Prolog
import os


os.environ['OMP_NUM_THREADS'] = "1"


queries = {
    "factors_us":"country(us, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)",
    "all_data":"person( GENDER, ACCURACY, COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P )",
    "factors_all":"person_factors( A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P )",
    "factors_all_weighted": "person_factors_weighted(WEIGHT, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P )",
    "factors_all_clustered": "person_clustered( A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P , CLUSTER )"
}

data_files = {
    "small_prolog": "small_data.pl",
    "big_prolog": "data.pl",
    "small_prolog_clustered": "small_data_clustered.pl"
}

facts_pieces = {
    "heads":{"person_raw":"person( ", "person_clustered": "person_clustered( "},
    "tails":{"person_raw":" ).", "person_clustered":" )."}
}

rules = {
    "small_prolog": ["country(COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) :- person( _, _, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), COUNTRY = COUNTRY1.",
                "country(COUNTRY, Results) :- findall(person( GENDER1, ACCURACY1, COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), person( GENDER1, ACCURACY1, COUNTRY, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), Results).",
                "gender(GENDER, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P) :- person( GENDER1, _, _, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), GENDER1 = GENDER.",
                "gender(GENDER, Results) :- findall(person( GENDER, ACCURACY1, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), person( GENDER, ACCURACY1, COUNTRY1, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ), Results).",
                "person_factors( A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ) :- person( _, _, _, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ).",
                "person_factors_weighted(WEIGHT, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P ) :- person( _, WEIGHT, _, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P )."
                ],
    "small_prolog_clustered": []
}


def read_csv(path, col_to_drop = []):
    data_file = pd.read_csv(path, delimiter="\t")

    for col in col_to_drop:
        try:
            data_file.drop(col, axis = 1, inplace=True)
            print(f"[+] Column {col} removed")
        except Exception as e:
            print(f"[-] {e}")
            continue

    return data_file


def shrink_data(data_file):
    data_copy = data_file.copy(deep = True)
    question_letters = [chr(i) for i in range(ord('A'), ord('P') + 1)]
    question_nums = {"A":10, "B":13, "C":10, "D":10, "E":10, "F":10,
                     "G":10, "H":10, "I":10, "J":10, "K":10, "L":10, 
                     "M":10, "N":10, "O":10, "P":10 }
    question_partitions = []
    for letter in question_letters:
        partition = [f"{letter}{num}" for num in range(1, question_nums[letter]+1)]
        question_partitions.append(partition)

    for i in range(len(question_letters)):
        norm_factor = 5*question_nums[question_letters[i]]
        data_copy[question_letters[i]] = round(data_copy[question_partitions[i]].sum(axis = 1)/norm_factor, 2)
        data_copy.drop(question_partitions[i], axis = 1, inplace = True)

    return data_copy


def rows_to_facts(data_file, fact_head, fact_tail):    
    cols = data_file.columns.tolist()
    #cols_upper = [col.upper() for col in cols]

    facts = []

    for index, row in data_file.iterrows():
        fact = fact_head
        for i in range(len(cols)):
            if i == len(cols)-1:
                fact += f"{row[cols[i]]}".lower()
            else:
                fact += f"{row[cols[i]]}, ".lower()
        fact += fact_tail
        facts.append(fact)
    
    return facts

def write_prolog(lines, prolog_path):
    try:
        with open(prolog_path, mode="w") as pf:
            for line in lines:
                pf.write(line + "\n")
    except Exception as e:
        print(f"{e}")


def query_to_dataset(prolog_path, prolog_query):
    dataset = []
    prolog = Prolog()

    try:
        prolog.consult(prolog_path)
        solutions = prolog.query(prolog_query)
    except Exception as e:
        print(e)

    for sol in solutions:
        dataset.append(list(sol.values()))
        

    return dataset


def clusters_to_facts(clusters, fact_head, fact_tail):
    facts = []

    for i in range(len(clusters)):
        for factors in clusters[i]:
            fact_body = (", ".join([str(factor) for factor in factors])) + f", {i}"
            fact = f"{fact_head}{fact_body}{fact_tail}"
            facts.append(fact)

    return facts


def main():
    data_path = "../data/16PF/data.csv"
    to_drop = ["source", "elapsed", "age"]
    fact_head = facts_pieces["heads"]["person_raw"]
    fact_tail = facts_pieces["tail"]["person_raw"]


    rules_small = rules["small_prolog"]

    def write_big():
        print("[*] Reading csv file...")
        df = read_csv(data_path, to_drop)
        print("[+] csv file read correctly")
        print("[*] Converting data in prolog clauses...")
        program = rows_to_facts(df, fact_head, fact_tail)
        print("[*] Writing program to prolog file...")
        write_prolog(program, data_files["big_prolog"])     
        print("[+] Program file written correctly")

    def write_small():
        print("[*] Reading csv file...")
        df = read_csv(data_path, to_drop)
        print("[+] csv file read correctly")
        print("[*] Shrinking data...")
        df = shrink_data(df)
        print("[*] Converting data in prolog clauses...")
        facts = rows_to_facts(df, fact_head, fact_tail)
        program = rules_small + facts
        print("[*] Writing program to prolog file...")
        write_prolog(program, data_files["small_prolog"])
        print("[+] Program file written correctly")




    
    
    


if __name__ == "__main__":
    main()