import jsonlines
from sklearn.model_selection import train_test_split
from operator import itemgetter

json_list = []
json_id_list = []
json_file = jsonlines.open('./_data/construction_data.jsonl')

for line in json_file:
    json_list.append(line)
    json_id_list.append(line['id'])

trn_list, val_list = train_test_split(json_id_list, test_size=0.25)

trn_set_dir = './_data/train_set'
val_set_dir = './_data/validation_set'

for data in json_list:
    if data['id'] in trn_list:
        f = open(trn_set_dir + '/' +
                 str(data['id']) + '.txt', 'w', encoding='utf-8')
    else:
        f = open(val_set_dir + '/' +
                 str(data['id']) + '.txt', 'w', encoding='utf-8')
    raw_txt = data['data'].split('\n')
    label_txt = data['data']
    
    data['label']['entities'] = sorted(data['label']['entities'], key=itemgetter('start_offset'), reverse=True)
    for loc in data['label']['entities']:
        replace_txt = '<' + label_txt[loc['start_offset']:loc['end_offset']] + ':' + loc['label'] + '>'
        label_txt = label_txt[:loc['start_offset']] + replace_txt + label_txt[loc['end_offset']:]
    label_txt = label_txt.split('\n')

    raw_txt = list(filter(None, raw_txt))
    label_txt = list(filter(None, label_txt))

    if len(raw_txt) == len(label_txt):
        count = 1
        for i in range(len(raw_txt)):
            if not raw_txt[i].isspace():
                f.write("## %d\n" % count)
                f.write("## %s\n" % raw_txt[i])
                f.write("## %s\n" % label_txt[i])
                f.write("\n")
                count += 1
    f.close()
