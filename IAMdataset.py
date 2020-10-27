import glob
import os
LABEL_PATH = "/data/cain/data/IAM/lines.txt"
filePath = "/data/cain/data/IAM/"
from PIL import Image

from sklearn.model_selection import train_test_split

maxTextLen = 100
bad_samples = []
bad_samples_reference = ['a01-117-05-02.png', 'r06-022-03-05.png']
seed = 0

def truncateLabel(text, maxTextLen):
    # ctc_loss can't compute loss if it cannot find a mapping between text label and input
    # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
    # If a too-long label is provided, ctc_loss returns an infinite gradient
    cost = 0
    for i in range(len(text)):
        if i != 0 and text[i] == text[i - 1]:
            cost += 2
        else:
            cost += 1
        if cost > maxTextLen:
            return text[:i]
    return text

def parse_IAM_data(txt_file, make_charset=False, train_size=None):
    chars = set()
    x = []
    y = []
    with open(txt_file) as f:
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            # print(line)
            lineSplit = line.strip().split(' ')  ## remove the space and split with ' '
            # assert len(lineSplit) >= 9

            #TODO: get filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + 'lines/' + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +\
                       lineSplit[0] + '.png'

            # print(fileName)

            #TODO: read text
            gtText_list = lineSplit[8].split('|')

            gtText = truncateLabel(' '.join(gtText_list), maxTextLen)
            chars = chars.union(set(list(gtText)))  ## taking the unique characters present

            # check if image is not empty
            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                print(gtText)
                Image.open(fileName).show()
                continue

            # return
            x.append(fileName)
            y.append(gtText)

    print(chars)
    if make_charset:
        with open("IAM_charset.txt", "w", encoding="utf-8") as f:
            f.write("".join(chars))

    # splitting train and valid set
    x, x_val, y, y_val = train_test_split(x, y, train_size = 0.9, shuffle=True, random_state=seed)

    if isinstance(train_size, int):
        x_sup, x_unsup, y_sup, y_unsup = train_test_split(x, y, train_size = train_size, shuffle=True, random_state=seed)
    else:
        x_sup = x
        y_sup = y
        x_unsup = []
        y_unsup = []

    return x_sup, y_sup, x_unsup, y_unsup, x_val, y_val

            
x_sup, y_sup, x_unsup, y_unsup, x_val, y_val = parse_IAM_data(LABEL_PATH)
# for data, label in zip(x_val,y_val):
#     print(data, label)
            

