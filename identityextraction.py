import pandas as pd
import random
from ast import literal_eval

def get_identities_and_offsets(row):
    ch = random.choice(row['identities'])
    index = row['context'].index(ch)
    length = len(ch)
    return (ch, index, length)


def main():
    df = pd.read_csv('E:\ResearchWork\CurrentResearch/InfillingGPT2Custom/FinalDataFrame5.csv')
    df['identities'] = df['identities'].apply(lambda x: literal_eval(x))
    df['choices'] = df.apply(lambda x: get_identities_and_offsets(x), axis = 1)
    df.to_csv('DataFrameforGPT2.csv', index =False)

if __name__=='__main__':
    main()