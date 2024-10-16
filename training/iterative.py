# Adapted from https://github.com/benbergner/ips.git
# This script defines a single epoch of training and calls in ips.py and transformer.py for the entire forward pass
# The loss function and backward pass are defined in this script
import sys
import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import adjust_learning_rate

def init_batch(device, conf):
    """
    Initialize the  memory buffer for the batch consisting of M patches
    """
    if conf.is_image:
        mem_patch =torch.zeros((conf.B, conf.M, conf.n_chan_in, *conf.patch_size)).to(device)
    else:
        mem_patch= torch.zeros((conf.B, conf.M, conf.n_chan_in)).to(device)

    if conf.use_pos:
        mem_pos_enc =  torch.zeros((conf.B, conf.M, conf.D)).to(device)
    else:
        mem_pos_enc= None

    # Init the labels for the batch (for multiple tasks in mnist)
    labels = {}
    for task in conf.tasks.values():
        if task['metric'] == 'multilabel_accuracy':
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B, ), dtype=torch.int64).to(device)
    
    return mem_patch, mem_pos_enc, labels

def fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
               mem_patch_iter, mem_pos_enc_iter, conf):
    """
    Fill the patch, pos enc and  label buffers and update helper variables
    """
    n_seq, len_seq = mem_patch_iter.shape[:2 ]
    mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
    if conf.use_pos:
        mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
    
    for task in conf.tasks.values():
        labels[task['name']][n_prep: n_prep + n_seq] = data[task['name']]
    
    n_prep += n_seq
    n_prep_batch += 1

    batch_data = (mem_patch , mem_pos_enc, labels, n_prep, n_prep_batch)

    return batch_data

def shrink_batch(mem_patch , mem_pos_enc , labels , n_prep , conf):
    """
    Adjust batch by removing empty instances (may occur in last batch of an epoch)
    """
    mem_patch = mem_patch[:n_prep]
    if conf.use_pos:
        mem_pos_enc = mem_pos_enc[:n_prep]
    
    for task in conf.tasks.values():
        labels[task['name']] =labels[task['name']][:n_prep]
    
    return mem_patch, mem_pos_enc, labels

def compute_diversity_loss(attn_maps):
    if attn_maps is None:
        raise ValueError("Attention  maps not computed.")
    
    # attn_maps has shape [batch_size, num_heads, seq_length]
    batch_size, num_heads,  _, _= attn_maps.shape
    diversity_losses = []

    # Here we compute the cosine similarity between each pair of attention heads
    for  i  in range( num_heads):
        for j in range(i + 1, num_heads):
            #  Compute cosine similarity between different heads
            attn_map_i = attn_maps[:, i, :, :].view( batch_size , -1)
            attn_map_j = attn_maps[:, j, :, :].view(batch_size, -1)
            
            # Compute cosine similarity  along the last dimension and then the mean over the batch
            diversity_loss = F.cosine_similarity(attn_map_i, attn_map_j, dim=-1).mean()
            diversity_losses.append(diversity_loss)
    
    # Convert the list to tensors for computing the mean and std deviation to be plotted during training.
    diversity_losses_tensor=torch.stack(diversity_losses)
    mean_diversity_loss= diversity_losses_tensor.mean()
    median_diversity_loss = diversity_losses_tensor.median()
    variance_diversity_loss = diversity_losses_tensor.var()
        
    # Calculate the average diversity loss
    overall_diversity_loss = (2/(num_heads *(num_heads - 1))) * torch.sum(diversity_losses_tensor)
    
    return overall_diversity_loss , variance_diversity_loss

def compute_semantic_loss(branch_outputs, labels, criterions, conf):
    """
    Compute the semantic loss for each task  using the output for each head, and print mean, median, and variance of losses.
    """
    semantic_losses = []
    # So we compute the average loss for each head and then add it to the overall loss function
    for branch_preds in branch_outputs:
        branch_loss = 0
        for task in conf.tasks.values():
            t_name, t_act=task['name'], task['act_fn']
            criterion = criterions[t_name]
            label = labels[t_name]
            
            branch_pred = branch_preds[t_name].squeeze(-1)
            if t_act == 'softmax':
                pred_loss = torch.log(branch_pred + conf.eps)
                label_loss = label
            else:
                pred_loss = branch_pred.view(-1)
                label_loss = label.view(-1).type(torch.float32)
            
            branch_loss += criterion(pred_loss, label_loss)
        
        semantic_losses.append(branch_loss)
    
    # Stack the semantic losses into a tensor
    semantic_losses_tensor = torch.stack(semantic_losses)

    # Get the number of  tasks
    num_tasks = len(conf.tasks.values())

    # We further divide the semantic loss in the tensor by the number of tasks
    semantic_losses_tensor/= num_tasks

    # Compute the mean, median, and variance for plotting in the training procedure
    mean_semantic_loss =semantic_losses_tensor.mean()
    median_semantic_loss =semantic_losses_tensor.median()
    variance_semantic_loss =semantic_losses_tensor.var()

    return mean_semantic_loss, variance_semantic_loss


def compute_loss(net , mem_patch , mem_pos_enc , criterions , labels , conf ):
    """
   Compute the overall loss function that includes semantic and diversity loss.
    """

    # Obtain predictions
    main_output, branch_outputs = net(mem_patch, mem_pos_enc)

    # Compute losses for each task and sum them up
    loss = 0
    task_losses, task_preds, task_labels = {},{},{}
    for task in conf.tasks.values():
        t_name, t_act = task['name'], task['act_fn']

        criterion = criterions[t_name]
        label = labels[t_name]

        # Main output loss
        main_pred = main_output[t_name].squeeze(-1)
        if t_act == 'softmax':
            pred_loss = torch.log(main_pred + conf.eps)
            label_loss = label
        else:
            pred_loss = main_pred.view(-1)
            label_loss = label.view(-1).type(torch.float32)

        main_task_loss = criterion(pred_loss, label_loss)
        task_losses[t_name]=main_task_loss.item()

        task_preds[t_name]=main_pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += main_task_loss

    # Average task losses        
    loss /= len(conf.tasks.values())

    diversity_loss,variance_diversity_loss = torch.tensor(0.0), torch.tensor(0.0)
    semantic_loss, variance_semantic_loss = torch.tensor(0.0), torch.tensor(0.0)

    if conf.semantic_diversity_loss:
        # only include the diversity loss if it is specified in the config file
        diversity_loss, variance_diversity_loss = compute_diversity_loss(net.transf.attn_maps)

        # Compute semantic loss
        semantic_loss, variance_semantic_loss = compute_semantic_loss(branch_outputs, labels, criterions, conf)

    # Total loss
    total_loss = loss+(diversity_loss * 3) +semantic_loss

    return total_loss, task_losses, task_preds, task_labels, diversity_loss, semantic_loss, variance_diversity_loss, variance_semantic_loss


# The train one epoch function is completely unchanged by us
def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    net.train()

    n_prep, n_prep_batch = 0, 0 
    mem_pos_enc = None
    start_new_batch = True

    times = []
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False

            if conf.track_efficiency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or  is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
            optimizer.zero_grad()

            loss,task_losses, task_preds, task_labels, diversity_loss, semantic_loss, variance_diversity_loss, variance_semantic_loss = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)

            loss.backward()
            optimizer.step()

            if conf.track_efficiency:
                end_event.record()
                torch.cuda.synchronize()
                if epoch == conf.track_epoch and data_it > 0 and not is_last_batch:
                    times.append(start_event.elapsed_time(end_event))
                    print("time: ", times[-1])

            log_writer.update(task_losses, task_preds, task_labels, diversity_loss, semantic_loss, variance_diversity_loss, variance_semantic_loss)

            n_prep = 0
            start_new_batch = True
    
    if conf.track_efficiency:
        if epoch == conf.track_epoch:
            print("avg. time: ", np.mean(times))

            stats = torch.cuda.memory_stats()
            peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.4f} GB")

            print("TORCH.CUDA.MEMORY_SUMMARY: ", torch.cuda.memory_summary())
            sys.exit()

# THe evaluate function is completely  unchanged by us
@torch.no_grad()
def evaluate(net,  criterions, data_loader,  device, log_writer, conf, epoch):
    net.eval()

    n_prep,  n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True

    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        batch_data  =  fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc , labels =  shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)

            loss, task_losses, task_preds, task_labels, diversity_loss, semantic_loss, variance_diversity_loss, variance_semantic_loss = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)

            log_writer.update(task_losses, task_preds, task_labels, diversity_loss, semantic_loss, variance_diversity_loss, variance_semantic_loss)

            n_prep = 0
            start_new_batch = True
