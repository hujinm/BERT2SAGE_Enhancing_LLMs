import torch
import os 
from torch.utils.data import Dataset, DataLoader

class ProBertEmbeddings(Dataset):
    """ProBert Embeddings dataset."""

    def __init__(self, data_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path      = data_path
        self.organism_paths, self.sequence_paths, self.edge_paths = self.get_organisms_paths()
     

    def __len__(self):
        return len(self.organism_paths)

    def get_organisms_paths(self) -> list :
      ids_path          = os.path.join(self.data_path,'taxon_ids.txt')
      organism_paths    = list()
      sequence_paths    = list()
      edge_paths        = list()

      with open(ids_path) as handler:
        for id in handler.readlines():
          id               = id.strip()
          current_path_o   = os.path.join(self.data_path, id,'embedding.pt')
          current_path_s   = os.path.join(self.data_path, id,'sequence.fa')
          current_path_e   = os.path.join(self.data_path, id,'edges.txt')
        
          organism_paths.append(current_path_o)
          sequence_paths.append(current_path_s)
          edge_paths.append(current_path_e)

      return  organism_paths,sequence_paths,edge_paths
    
    @staticmethod
    def get_classification_matrix(sequence_path:str,edge_path:str) -> torch.tensor:
      f = open(sequence_path)
      sequence_examples = ''.join(f.readlines()).split('>')
      sequence_names = {}
      for i in range(1,len(sequence_examples)):
        sequence_examples[i] = sequence_examples[i].split("\n")
        name = sequence_examples[i].pop(0)
        name = name.split()[0]
        sequence_names[name] = i - 1
        sequence_examples[i] = ''.join(sequence_examples[i])
      sequence_examples.pop(0)
      f.close()
      print(sequence_names)
      classifier_matrix = torch.zeros((len(sequence_names), len(sequence_names)))
      f = open(edge_path, 'r')
      f.readline()
      edges = [l.split() for l in f.readlines()]
      for edge in edges:
        print(edge)
        i = sequence_names[edge[2]]
        j = sequence_names[edge[3]]
        classifier_matrix[i,j] = 1

      return  classifier_matrix

    def __getitem__(self, idx):
      organism_path = self.organism_paths[idx]
      Z            = torch.load(organism_path)
      Y            = self.get_classification_matrix(sequence_path = self.sequence_paths[idx],
                                                    edge_path     = self.edge_paths[idx] ) 
      return Z,Y 