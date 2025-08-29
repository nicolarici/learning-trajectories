from transformers import GPTNeoXTokenizerFast
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import ast
import os

class LanguageDataset(Dataset):

    def __init__(self):
        raise Exception("LanguageDataset is an abstract class and should not be instantiated directly.")
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def split_dataset(self):
        
        train = self.df.sample(frac=self.args['dataset']['split']['train'], random_state=self.args['seed'])
        dev_test = self.df.drop(train.index)

        frac_test = self.args['dataset']['split']['test'] / (1 - self.args['dataset']['split']['train'])

        test = dev_test.sample(frac=frac_test, random_state=self.args['seed'])
        dev = dev_test.drop(test.index)

        train.reset_index(drop=True, inplace=True)
        dev.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        return train, dev, test
    

    def get_train_dataloader(self, shuffle=True):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle)
    

    def get_dev_dataloader(self):
        return DataLoader(self.dev, batch_size=self.batch_size, shuffle=False)
    

    def get_test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
    

    def get_special_token_ids(self):
        special_token_ids = ""
        for _, token in self.tokenizer.special_tokens_map.items():
            special_token_ids += str(self.tokenizer.convert_tokens_to_ids(token)) + ","

        return special_token_ids[:-1]
    
    def get_num_labels(self):
        return self.num_labels


class DiskDataset(LanguageDataset):

    def __init__(self, args, representation_path=None, labels_path=None):

        self.args = args

        if representation_path is not None and labels_path is not None:
            self.representation = torch.load(os.path.join("disk_data", args['model']['data_dir'], representation_path))
            self.labels = torch.load(os.path.join("disk_data", args['model']['data_dir'], labels_path))

        else:
            self.train = DiskDataset(args, "train_representations.pt", "train_labels.pt")
            self.dev = DiskDataset(args, "dev_representations.pt", "dev_labels.pt")
            self.test = DiskDataset(args, "test_representations.pt", "test_labels.pt")

        self.batch_size = args['probe_training']['batch_size']
        self.num_labels = args['dataset']['label_space_size']


    def __len__(self):
        return len(self.representation[0])
    

    def __getitem__(self, idx):

        rep, att, span = {}, {}, {}
        for layer in self.representation.keys():
            rep[layer] = self.representation[layer][idx]
            att[layer] = torch.zeros_like(rep[layer])      # Dummy attention mask
            span[layer] = torch.zeros_like(rep[layer])     # Dummy span

        label = self.labels[idx]

        return rep, att, label, span


    def get_special_token_ids(self):
        return self.args['model']['special_token_ids']


class PythiaDataset(LanguageDataset):

    def __init__(self, args, df=None):

        self.args = args

        if 'step' in args['model']:
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(args['model']['name'], revision=f"step{args['model']['step']}")
        else:
            self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(args['model']['name'])
        self.tokenizer.pad_token = self.tokenizer.decode(50278)

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_json(args['dataset']['path'], lines=True)
            df_train, df_dev, df_test = self.split_dataset()
            self.train = PythiaDataset(args, df_train)
            self.dev = PythiaDataset(args, df_dev)
            self.test = PythiaDataset(args, df_test)

        self.batch_size = args['probe_training']['batch_size']
        self.num_labels = len(set(self.df.label))


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        encoding = self.tokenizer(self.df.loc[idx, "text"], return_tensors='pt', padding='max_length', truncation=True, max_length=self.args['dataset']['max_length'])

        label = torch.zeros(self.num_labels)
        label[self.df.loc[idx, "label"]] = 1

        tok_span = []
        for w in encoding.word_ids():
            if w in ast.literal_eval(self.df.loc[idx, "span"]):
                tok_span.append(True)
            else:
                tok_span.append(False)


        return encoding['input_ids'].squeeze(), encoding['attention_mask'].squeeze(), label, torch.tensor(tok_span)
