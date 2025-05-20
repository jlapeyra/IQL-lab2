from ngram import Model
from tqdm import tqdm

def train(name:str, data_files:list[str], split_lines=True, n=5, maxlines=100_000):
    model = Model(n, name)
    for fn in (tqdm(data_files, 'training '+name) if len(data_files) >= 10 else data_files):
        with open(fn, 'r', encoding='utf-8') as f:
            if not split_lines:
                model.feed(f.read())
            else:
                for line in tqdm(f.readlines()[:maxlines], 'training with '+fn, leave=False):
                    model.feed(line)
    model.save()



if __name__ == '__main__':
    train('europarl-ca', ['data/europarl/europarl.ca.txt'])