from transformers import AutoModel
import torch
import torch.nn as nn


class LanguageModel(nn.Module):

    def __init__(self):
        super(LanguageModel, self).__init__()


    def forward(self):
        raise NotImplementedError
    

    def get_hidden_dim(self):
        return self.language_model.config.hidden_size
    

    def get_num_layers(self):
        return self.language_model.config.num_hidden_layers


    def remove_all_special_tokens_and_compute_mean(self, layer_hs, input_ids, attention_mask):
        
        # Generate a mask for all non-special tokens (True for non-special tokens, False for special tokens and padding tokens)
        non_special_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        for special_token_id in self.special_token_ids:
            non_special_tokens_mask &= (input_ids != special_token_id)
        
        non_special_tokens_mask &= (attention_mask != 0)
        expanded_mask = non_special_tokens_mask.unsqueeze(-1).expand(layer_hs.size())

        # Zero out the embeddings of special tokens
        filtered_embeddings = torch.where(expanded_mask, layer_hs, torch.tensor(0.0, device=layer_hs.device))

        # Compute the sum of embeddings and the count of non-special tokens for each sentence (if the count is zero, we will avoid division by zero setting the value to 1)
        embeddings_sum = filtered_embeddings.sum(dim=1)
        non_special_tokens_count = non_special_tokens_mask.sum(dim=1, keepdim=True)
        non_special_tokens_count = non_special_tokens_count.masked_fill_(non_special_tokens_count == 0, 1)
        
        # Compute the mean embedding vector for each sentence        
        return embeddings_sum / non_special_tokens_count.float()


    def span_mean(self, hs, fos_span):

        span_mean = torch.zeros(hs.shape[0], hs.shape[2], dtype=hs.dtype, device=hs.device)

        # Iterate over each element in the batch dimension
        for i in range(hs.shape[0]):
            # Apply mask and select the vectors for the current batch element
            selected_vectors = hs[i][fos_span[i]]

            # Compute the mean of the selected vectors if any are True, else retain zeros
            if selected_vectors.nelement() != 0:
                span_mean[i] = selected_vectors.mean(dim=0)

        return span_mean


class DiskModel(LanguageModel):

    def __init__(self, args):
        super(DiskModel, self).__init__()

        self.args = args
        self.special_token_ids = list(set(map(int, args['model']['special_token_ids'].split(','))))
        self.to(args['device'])
    

    def forward(self, input_tensor, attention_mask, layers, fos_span):
        return input_tensor
    
    def get_hidden_dim(self):
        return self.args['model']['hidden_dim']
    

    def get_num_layers(self):
        return self.args['model']['num_layers']


class PythiaModel(LanguageModel):

    def __init__(self, args):
        super(PythiaModel, self).__init__()

        self.args = args

        if 'step' in args['model']:
            self.language_model = AutoModel.from_pretrained(args['model']['name'], revision=f"step{args['model']['step']}")

        else:
            self.language_model = AutoModel.from_pretrained(args['model']['name'])

        if args['model']['freeze']:
            for param in self.language_model.parameters():
                param.requires_grad = False


        self.special_token_ids = list(set(map(int, args['model']['special_token_ids'].split(','))))
        self.to(args['device'])


    def forward(self, input_ids, attention_mask, layers, fos_span):

        lhs, pkv, hs = self.language_model(input_ids, attention_mask, output_hidden_states=True, return_dict=False)


        def compute_sentence_representation(hs, layer, input_ids, attention_mask, fos_span):
            if self.args['model']['sentence_rep'] == 'hs-mean':
                return self.remove_all_special_tokens_and_compute_mean(hs[layer], input_ids, attention_mask)

            elif self.args['model']['sentence_rep'] == 'span-mean':
                return self.span_mean(hs[layer], fos_span)
            
            elif self.args['model']['sentence_rep'] == 'random':
                return torch.rand_like(hs[layer][:, 0, :], dtype=torch.float32).to(self.args['device'])

            else:
                raise ValueError(f"Invalid sentence representation type --{self.args['model']['sentence_rep']}--; choose (hs-mean, span-mean, random)")
        

        out = {}
        for layer in layers:
            out[layer] = compute_sentence_representation(hs, layer, input_ids, attention_mask, fos_span)

        return out

