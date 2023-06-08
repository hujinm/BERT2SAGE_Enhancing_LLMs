from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
import gc
from datetime import datetime

gc.collect()
torch.cuda.empty_cache()

MAIN_PATH = 'C:/Users/golde/Documents/bert2sage_data/Data_test/'
TAXON_IDS_PATH = MAIN_PATH + 'taxon_ids.txt'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
model.full() if device=='cpu' else model.half()
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

def generate_ids(current_path):
    f = open(current_path + 'sequence.fa')
    sequence_examples = ''.join(f.readlines()).split('>')
    f.close()
    sequence_names = {}
    for i in range(1,len(sequence_examples)):
        sequence_examples[i] = sequence_examples[i].split("\n")
        name = sequence_examples[i].pop(0)
        name = name.split()[0]
        sequence_names[name] = i - 1
        sequence_examples[i] = ''.join(sequence_examples[i])
    sequence_examples.pop(0)
    seq_lengths = [len(seq) for seq in sequence_examples]
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]
    ids = tokenizer.batch_encode_plus(sequence_examples, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    return input_ids, attention_mask, seq_lengths

def print_msg(msg, file):
    print(msg)
    file.write(msg + '\n')

def run_ProtBERT(input_ids, attention_mask, seq_lengths, log):
    N = len(input_ids)
    Z = torch.zeros(N, 1024)
    c = 0
    for i in range(N):
        try:
            with torch.no_grad():
                Z[i] = model(input_ids=input_ids[i:(i+1)],attention_mask=attention_mask[i:(i+1)]).last_hidden_state[:,:seq_lengths[i]].mean(dim=1)
            c += 1
            if c > N/50:
                print_msg("\t\t" + str(round(i/N*100)) + "%", log)
                c = 0
        except:
            print_msg("* Crashed at i = " + str(i), log)
            break
    return Z

f = open(TAXON_IDS_PATH)
for id in f.readlines():
    id = str(int(id))
    current_path = MAIN_PATH + id + '/'
    log = open(current_path + "log.txt", "a")
    log.write(str(datetime.now()) + '\n')
    print_msg('-' + id, log)
    if os.path.isfile(current_path + 'embedding.pt'):
        print_msg('* Embedding already made for ' + id + '.\n', log)
        continue
    print_msg("\tGenerating IDs for " + id + "...", log)
    input_ids, attention_mask, seq_lengths = generate_ids(current_path)
    print_msg("\tRunning  ProtBERT for " + id + "...", log)
    Z = run_ProtBERT(input_ids, attention_mask, seq_lengths, log)
    torch.save(Z ,current_path + 'embedding.pt')
    log.close()
f.close()