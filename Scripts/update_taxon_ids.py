import torch
import shutil

MAIN_PATH = 'C:/Users/golde/Documents/bert2sage_data/Data/'

updated_taxon_ids = []
f = open(MAIN_PATH + 'taxon_ids.txt')
for id in f.readlines():
    id = int(id)
    current_path = MAIN_PATH + str(id) + '/'
    try:
        Z  = torch.load(current_path + 'embedding.pt')
        if Z.sum() == 0:
            shutil.rmtree(current_path)
        else:
            updated_taxon_ids.append(id)
    except:
        shutil.rmtree(current_path)
f.close()

with open(MAIN_PATH + 'updated_taxon_ids.txt', 'a') as f:
	for id in updated_taxon_ids:
		f.write(str(id) + '\n')
f.close()