from torch import optim
from tqdm import tqdm
import torch
import json
import sys
import os


class ProbeRegimen:
    """Basic regimen for training and running inference on probes.
  
    Attributes:
      args: the experiment arguments
      reporter: a reporter.Reporter object for reporting results
    """
  
    def __init__(self, args, reporter):
        self.args = args
        
        self.reporter = reporter
        self.reports = []
        
        self.max_epochs = args['probe_training']['epochs']
        self.params_path = args['reporting']['root']
        self.wait_without_improvement_for = self.args['probe_training']['wait_without_improvement_for']


    def set_optimizer(self, probe):
        """Sets the optimizer for the training regimen.
      
        Args:
          probe: the probe PyTorch model the optimizer should act on.
        """

        return optim.Adam(probe.parameters(), 
                            lr=self.args['probe_training']['learning_rate'], 
                            weight_decay=self.args['probe_training']['weight_decay'])



    def train_until_convergence(self, probes, model, loss, train_dataset, dev_dataset, portion, layers, eval_datasets={}):
        """ Trains a probe until a convergence criterion is met.
    
        Trains until loss on the development set does not improve by more than epsilon
        for 5 straight epochs.
    
        Writes parameters of the probe to disk, at the location specified by config.
    
        Args:
          probe: An instance of probe.Probe, transforming model outputs to predictions
          model: An instance of model.Model, transforming inputs to word reprs
          loss: An instance of loss.Loss, computing loss between predictions and labels
          train_dataset: a torch.DataLoader object for iterating through training data
          dev_dataset: a torch.DataLoader object for iterating through dev data
          portion: the current portion of the training data
          language_layer: the layer of the model to use for word representations
          eval_datasets: a dictionary of torch.DataLoader objects for evaluating on additional datasets
        """

    
        eval_on_exit_layers = layers.copy()

        def loss_on_dataset(dataset, layer, name=''):
            
            current_loss = 0
            num_targets = 0

            if self.args['verbose']: loop = tqdm(dataset, desc=f"[eval batch {name} on layer {layer}]")
            else: loop = dataset

            for batch in loop:
                try:
                    input_ids, attention_masks, labels, fos_span = [b.to(self.args['device']) for b in batch]
                except:
                    input_ids, attention_masks, labels, fos_span = [b for b in batch]
                
                word_representations = model(input_ids, attention_masks, eval_on_exit_layers, fos_span)

                rep = word_representations[layer]
                predictions = probes[layer](rep)

                batch_loss, count = loss(predictions, labels)
                current_loss += batch_loss.detach().cpu().numpy() * count
                num_targets += len(labels)
            

            return {f"loss_{name}": current_loss,
                    f"num_targets_{name}": num_targets}
    

        def num_targets(dataset):
            num_targets = 0
            for batch in dataset:
                _, _, labels, _ = batch
                num_targets += len(labels)
            return num_targets


        # Run the best-dev probe on testset.
        def eval_on_exit():

            if self.args['verbose']:
                tqdm.write("\nEvaling on exit...")

            online_coding_list = {l: {} for l in eval_on_exit_layers}
            for l in eval_on_exit_layers:
                probes[l].load_state_dict(torch.load(os.path.join(self.params_path,f"probe_layer_{l}.pt")))
                probes[l].eval()

                results = {'train_targets': num_targets(train_dataset), 
                           'probe_params_path': os.path.join(self.params_path,f"probe_layer_{l}.pt")
                          }
                
                for name, dataset in eval_datasets.items():
                    results.update(loss_on_dataset(dataset, l, name=name))

                online_coding_list[l].update(results)

            return online_coding_list


        optimizers = {l: self.set_optimizer(probes[l]) for l in layers}
        min_dev_loss = {l: sys.maxsize for l in layers}
        min_dev_loss_epoch = {l: 0 for l in layers}


        if self.args['verbose']: train_loop = tqdm(range(self.max_epochs), desc=f'[training on {portion*100}% of data]')
        else: train_loop = range(self.max_epochs)

        # Looping through epochs.
        for ep in train_loop:

            epoch_train_loss = {l: 0 for l in layers}
            epoch_train_epoch_count = {l: 0 for l in layers}
            epoch_dev_epoch_count = {l: 0 for l in layers}
            epoch_train_loss_count = {l: 0 for l in layers}


            # Looping through batches.
            for batch in train_dataset:
                
                for l in layers:
                    probes[l].train()
                    optimizers[l].zero_grad()

                # Forward.
                try:
                    input_ids, attention_masks, labels, fos_span = [b.to(self.args['device']) for b in batch]
                except:
                    input_ids, attention_masks, labels, fos_span = [b for b in batch]

                word_representations = model(input_ids, attention_masks, layers, fos_span)

                predictions = {}
                batch_loss = {}
                count = {}

                for l in layers:
                    predictions[l] = probes[l](word_representations[l])

                    # Loss and Backprop (over the normalized loss)

                    batch_loss[l], count[l] = loss(predictions[l], labels)
                    batch_loss[l].backward()
                    optimizers[l].step()

                    # Report the loss (NOT normalized by sentences)

                    epoch_train_loss[l] += batch_loss[l].detach().cpu().numpy() * count[l]
                    epoch_train_epoch_count[l] += 1
                    epoch_train_loss_count[l] += count[l]


            ###### LOOP TRAINING END ######



            # Evaluation over dev_dataset.
            epoch_dev_loss = {l: 0 for l in layers}
            epoch_dev_loss_count = {l: 0 for l in layers}

            for batch in dev_dataset:

                # Forward.
                try:
                    input_ids, attention_masks, labels, fos_span = [b.to(self.args['device']) for b in batch]
                except:
                    input_ids, attention_masks, labels, fos_span = [b for b in batch]

                word_representations = model(input_ids, attention_masks, layers, fos_span)

                predictions = {}
                for l in layers:
                    probes[l].eval()
                    
                    predictions[l] = probes[l](word_representations[l])
                    batch_loss[l], count[l] = loss(predictions[l], labels)

                    epoch_dev_loss[l] += batch_loss[l].detach().cpu().numpy() * count[l]
                    epoch_dev_loss_count[l] += count[l]
                    epoch_dev_epoch_count[l] += 1


            ###### LOOP EVALUATION END ######


            if self.args['verbose']:
                for l in layers:
                    tqdm.write(f"[epoch {ep:02d}] - [layer {l}] Train loss: {epoch_train_loss[l]/epoch_train_loss_count[l]:.4f}, Dev loss: {epoch_dev_loss[l]/epoch_dev_loss_count[l]:.4f}")

            layers_to_remove = []
            for l in layers:
                if epoch_dev_loss[l] / epoch_dev_loss_count[l] < min_dev_loss[l] - 0.001:
                    torch.save(probes[l].state_dict(), os.path.join(self.params_path,f"probe_layer_{l}.pt")) 
                    
                    min_dev_loss[l] = epoch_dev_loss[l] / epoch_dev_loss_count[l]
                    min_dev_loss_epoch[l] = ep

                # Early stopping.
                elif min_dev_loss_epoch[l] < ep - self.wait_without_improvement_for:
                    if self.args['verbose']:
                        tqdm.write(f"[layer {l}] Min dev loss no imroved for {self.wait_without_improvement_for} epochs. Early stopping!")
                    
                    with open(os.path.join(self.reporter.reporting_root, f'train_report.json'), 'w') as f:
                        json.dump(self.reports, f)

                    layers_to_remove.append(l)

                # Reporting.
                current_report = {'layer': l,
                                  'portion': portion,
                                  'epoch': ep,
                                  'dev_loss': epoch_dev_loss[l]/epoch_dev_loss_count[l],
                                  'train_loss': epoch_train_loss[l]/epoch_train_loss_count[l],
                                  'min_dev_loss': min_dev_loss[l],
                                  'min_dev_loss_epoch': min_dev_loss_epoch[l],
                                  'probe_path': os.path.join(self.params_path, f"probe_layer_{l}.pt")}

                self.reports.append(current_report)

            layers = [l for l in layers if l not in layers_to_remove]

            with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
                json.dump(self.reports, f)

            
            # Exiting
            if len(layers) == 0:
                return eval_on_exit()
         
        ###### LOOP EPOCHS END ######


        with open(os.path.join(self.reporter.reporting_root, 'train_report.json'), 'w') as f:
            json.dump(self.reports, f)

        return eval_on_exit()


    def predict(self, probe, model, daloader):
        """ Runs probe to compute predictions on a dataset.
    
        Args:
          probe: An instance of probe.Probe, transforming model outputs to predictions
          model: An instance of model.Model, transforming inputs to word reprs
          daloader: A pytorch.DataLoader object 
          language_layer: the layer of the model to use for word representations
    
        Returns:
          A list of predictions for each batch in the batches yielded by the dataset
        """

        probe.eval()
        predictions_by_batch = []

        if self.args['verbose']: predict_loop = tqdm(daloader, desc=f'[predicting on layer {probe.layer}]')
        else: predict_loop = daloader

        for batch in predict_loop:

            try:
                input_ids, attention_masks, _, fos_span = [b.to(self.args['device']) for b in batch]
            except:
                input_ids, attention_masks, _, fos_span = [b for b in batch]
   
            word_representations = model(input_ids, attention_masks, [probe.layer], fos_span)
            predictions = probe(word_representations[probe.layer])

            predictions_by_batch.append(predictions.detach().cpu().numpy())

        return predictions_by_batch
