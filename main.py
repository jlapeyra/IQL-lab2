from ngram import Model
import glob

def main(n=6, name='europarl-es'):
    print('MODEL:', name)
    print('STRATEGY:', f'{n}-grams')
    print()
    print('Enter a prompt and the model will continue the sentence.')
    print('The prompt can be empty (just hit ENTER).')
    print('To quit, type -q.')
    print()
    model = Model(n, name)
    model.load()
    prompt = None
    while (prompt := input('Prompt (optional): ')) != '-q':
        count = 0
        while count < 50:
            sentence = model.generate(prompt)
            #print(sentence)
            count += len(sentence)

main()
