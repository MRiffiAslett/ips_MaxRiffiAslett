import sys
import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import adjust_learning_rate

def init_batch(device, conf):
    """
    Initialize the memory buffer for the batch consisting of M patches
    """
    if conf.is_image:
        mem_patch = torch.zeros((conf.B, conf.M, conf.n_chan_in, *conf.patch_size)).to(device)
    else:
        mem_patch = torch.zeros((conf.B, conf.M, conf.n_chan_in)).to(device)

    if conf.use_pos:
        mem_pos_enc = torch.zeros((conf.B, conf.M, conf.D)).to(device)
    else:
        mem_pos_enc = None

    # Init the labels for the batch (for multiple tasks in mnist)
    labels = {}
    for task in conf.tasks.values():
        if task['metric'] == 'multilabel_accuracy':
            labels[task['name']] = torch.zeros((conf.B, conf.n_class), dtype=torch.float32).to(device)
        else:
            labels[task['name']] = torch.zeros((conf.B,), dtype=torch.int64).to(device)
    
    return mem_patch, mem_pos_enc, labels

def fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
               mem_patch_iter, mem_pos_enc_iter, conf):
    """
    Fill the patch, pos enc and label buffers and update helper variables
    """
    n_seq, len_seq = mem_patch_iter.shape[:2]
    mem_patch[n_prep:n_prep+n_seq, :len_seq] = mem_patch_iter
    if conf.use_pos:
        mem_pos_enc[n_prep:n_prep+n_seq, :len_seq] = mem_pos_enc_iter
    
    for task in conf.tasks.values():
        labels[task['name']][n_prep:n_prep+n_seq] = data[task['name']]
    
    n_prep += n_seq
    n_prep_batch += 1

    batch_data = (mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch)

    return batch_data

def shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf):
    """
    Adjust batch by removing empty instances (may occur in last batch of an epoch)
    """
    mem_patch = mem_patch[:n_prep]
    if conf.use_pos:
        mem_pos_enc = mem_pos_enc[:n_prep]
    
    for task in conf.tasks.values():
        labels[task['name']] = labels[task['name']][:n_prep]
    
    return mem_patch, mem_pos_enc, labels

def compute_diversity_loss(attn_maps):
    if attn_maps is None:
        raise ValueError("Attention maps not computed. Ensure forward pass is run before computing diversity loss.")
    
    # Assuming attn_maps has shape [batch_size, num_heads, seq_length, seq_length]
    batch_size, num_heads, _, _ = attn_maps.shape
    diversity_losses = []
    
    for i in range(num_heads):
        for j in range(i + 1, num_heads):
            # Compute cosine similarity between different heads
            attn_map_i = attn_maps[:, i, :, :].view(batch_size, -1)
            attn_map_j = attn_maps[:, j, :, :].view(batch_size, -1)
            
            # Compute cosine similarity along the last dimension and then mean over the batch
            diversity_loss = F.cosine_similarity(attn_map_i, attn_map_j, dim=-1).mean()
            diversity_losses.append(diversity_loss)
    
    # Convert list to tensor for statistical computation
    diversity_losses_tensor = torch.stack(diversity_losses)
    mean_diversity_loss = diversity_losses_tensor.mean()
    median_diversity_loss = diversity_losses_tensor.median()
    variance_diversity_loss = diversity_losses_tensor.var()
    
    # Print out the statistics
    print(f"Diversity Loss Statistics - Mean: {mean_diversity_loss:.4f}, Median: {median_diversity_loss:.4f}, Variance: {variance_diversity_loss:.4f}")
    
    # Calculate the average diversity loss as originally intended
    overall_diversity_loss = (2 / (num_heads * (num_heads - 1))) * torch.sum(diversity_losses_tensor)
    
    return overall_diversity_loss

def compute_semantic_loss(branch_outputs, labels, criterions, conf):
    """
    Compute the semantic loss for each task using the branch outputs, and print mean, median, and variance of losses.
    """
    semantic_losses = []
    
    for branch_preds in branch_outputs:
        branch_loss = 0
        for task in conf.tasks.values():
            t_name, t_act = task['name'], task['act_fn']
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
    
    semantic_losses_tensor = torch.stack(semantic_losses)
    mean_semantic_loss = semantic_losses_tensor.mean()
    median_semantic_loss = semantic_losses_tensor.median()
    variance_semantic_loss = semantic_losses_tensor.var()

    # Print out the mean, median, and variance of semantic loss
    print(f"Semantic Loss - Mean: {mean_semantic_loss:.4f}, Median: {median_semantic_loss:.4f}, Variance: {variance_semantic_loss:.4f}\n\n")

    return semantic_losses_tensor.mean()



def compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf):
    """
    Obtain predictions, compute losses for each task and get some logging stats.
    """

    # Obtain predictions
    main_output, branch_outputs = net(mem_patch, mem_pos_enc)

    # Compute losses for each task and sum them up
    loss = 0
    task_losses, task_preds, task_labels = {}, {}, {}
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
        task_losses[t_name] = main_task_loss.item()

        task_preds[t_name] = main_pred.detach().cpu().numpy()
        task_labels[t_name] = label.detach().cpu().numpy()

        loss += main_task_loss

    # Average task losses        
    loss /= len(conf.tasks.values())

    # Compute diversity loss
    diversity_loss = compute_diversity_loss(net.transf.attn_maps)

    # Compute semantic loss
    semantic_loss = compute_semantic_loss(branch_outputs, labels, criterions, conf)

    # Total loss
    total_loss = loss + diversity_loss + semantic_loss

    return total_loss, [task_losses, task_preds, task_labels]



def train_one_epoch(net, criterions, data_loader, optimizer, device, epoch, log_writer, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    net.train()

    # Initialize helper variables
    n_prep, n_prep_batch = 0, 0 # num of prepared images/batches
    mem_pos_enc = None
    start_new_batch = True

    times = [] # only used when tracking efficiency stats
    # Loop through dataloader
    for data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data['input'].to(device) if conf.eager else data['input']

        # If starting a new batch, create placeholders for data which are filled later
        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False

            # If tracking efficiency, record time from here.
            if conf.track_efficiency:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
        
        # Apply IPS to input patches
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        # Fill batch placeholders with patches from IPS step
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        # Check if the current batch is full or if it is the last batch
        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        # Do training step as soon as batch is full (or last batch)
        if batch_full or is_last_batch:

            if not batch_full:
                # Last batch may not be full, so remove empty instances
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            # Calculate and set new learning rate
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch, conf.lr, optimizer, data_loader, data_it+1)
            optimizer.zero_grad()

            # Compute loss
            loss, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            # Backpropagate error and update parameters
            loss.backward()
            optimizer.step()

            # If tracking efficiency, log the time and memory usage
            if conf.track_efficiency:
                end_event.record()
                torch.cuda.synchronize()
                if epoch == conf.track_epoch and data_it > 0 and not is_last_batch:
                    times.append(start_event.elapsed_time(end_event))
                    print("time: ", times[-1])

            # Update log
            log_writer.update(task_losses, task_preds, task_labels)

            # Reset helper variables
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


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterions, data_loader, device, log_writer, conf):

    # Set the network to evaluation mode
    net.eval()

    # Remaining parts similar to training loop
    n_prep, n_prep_batch = 0, 0
    mem_pos_enc = None
    start_new_batch = True
    
    for data in data_loader:
        image_patches = data['input'].to(device) if conf.eager else data['input']

        if start_new_batch:
            mem_patch, mem_pos_enc, labels = init_batch(device, conf)
            start_new_batch = False
        
        mem_patch_iter, mem_pos_enc_iter = net.ips(image_patches)
        
        batch_data = fill_batch(mem_patch, mem_pos_enc, labels, data, n_prep, n_prep_batch,
                                mem_patch_iter, mem_pos_enc_iter, conf)
        mem_patch, mem_pos_enc, labels, n_prep, n_prep_batch = batch_data

        batch_full = (n_prep == conf.B)
        is_last_batch = n_prep_batch == len(data_loader)

        if batch_full or is_last_batch:

            if not batch_full:
                mem_patch, mem_pos_enc, labels = shrink_batch(mem_patch, mem_pos_enc, labels, n_prep, conf)
            
            _, task_info = compute_loss(net, mem_patch, mem_pos_enc, criterions, labels, conf)
            task_losses, task_preds, task_labels = task_info

            log_writer.update(task_losses, task_preds, task_labels)

            n_prep = 0
            start_new_batch = True