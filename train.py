from ngram import Model
from tqdm import tqdm
import random

def train(name:str, data_files:list[str], split_lines=True, n=6, maxlines=400_000, maxfiles=2_000):
    model = Model(n, name)
    random.shuffle(data_files)
    data_files = data_files[:maxfiles]
    for fn in tqdm(data_files, 'Training '+name) if len(data_files) >= 5 else data_files:
        with open(fn, 'r', encoding='utf-8') as f:
            if not split_lines:
                model.feed(f.read())
            else:
                lines = f.readlines()[:maxlines]
                for line in tqdm(lines, 'Training from '+fn, leave=False) if len(lines) > 5_000 else lines:
                    model.feed(line.strip())
    model.save()


if __name__ == '__main__':
    import glob
    train('wikipedia-ca', glob.glob('data/wikipedia/ca/articles/*'))