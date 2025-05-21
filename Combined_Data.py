import os

combined_data = open('FullTurkishData.txt', 'w', encoding='utf8')
for direc in os.listdir('categories/'):
    path = 'categories/' + direc + '/'
    if os.path.isdir(path):
        filename = path + 'linear.txt'
        try:
            with open(filename, 'r', encoding='utf8') as file:
                data = file.readlines()
                for line in data:
                    if line.strip() != '':
                        if '=' not in line:
                            combined_data.write(line)
        except:
            pass
combined_data.close()