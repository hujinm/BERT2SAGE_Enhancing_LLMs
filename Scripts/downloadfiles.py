import requests
import os
import gzip
import shutil

MAIN_PATH = 'C:/Users/golde/Documents/bert2sage_data/Data/'
SEQUENCE_URL = ['https://stringdb-static.org/download/protein.sequences.v11.5/', '.protein.sequences.v11.5.fa.gz']
EDGE_URL = ['https://stringdb-static.org/download/protein.physical.links.v11.5/', '.protein.physical.links.v11.5.txt.gz']

def unzip_file(save_path, name):
    with gzip.open(save_path + name + '.gz', 'rb') as f_in:
        with open(save_path + name + '.txt', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(save_path + name +'.gz') 

def download_url(url, save_path, name, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path + name + '.gz', 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    unzip_file(save_path, name)

f = open(MAIN_PATH + 'taxon_ids.txt')
for id in f.readlines():
    id = int(id)
    sequence_url = SEQUENCE_URL[0] + str(id) + SEQUENCE_URL[1]
    edge_url = EDGE_URL[0] + str(id) + EDGE_URL[1]
    save_path = MAIN_PATH + str(id) + '/'
    try:
        os.mkdir(save_path)
        print("Downloading ", id)
        download_url(sequence_url, save_path, "sequence")
        download_url(edge_url, save_path, "edges")
    except:
        print("Already have ", id)
f.close()