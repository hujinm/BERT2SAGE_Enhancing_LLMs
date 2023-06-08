MAIN_PATH = 'C:/Users/golde/Documents/bert2sage_data/Data/'
TAXON_IDS_PATH = MAIN_PATH + 'taxon_ids.txt'
OUTPUT_PATH = MAIN_PATH + 'combined.fa'

out = open(OUTPUT_PATH, 'w')
f = open(TAXON_IDS_PATH)
for id in f.readlines():
    id = str(int(id))
    current_path = MAIN_PATH + id + '/'
    seq = open(current_path + 'sequence.txt')
    for s in seq:
        out.write(s)