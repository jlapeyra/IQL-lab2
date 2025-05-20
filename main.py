from ngram import Model
import glob

def main():
    model = Model(n=5, name='europerl-ca')
    model.load()
    prompt = None
    while prompt != '.':
        prompt = input('Prompt: ')
        prompt = prompt.split()
        model.generate(prompt, 30)

main()
