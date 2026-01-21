# import matplotlib.pyplot as plt
import subprocess

import numpy as np
import torch
# import plotly.graph_objs as go
from torch.utils.data import Dataset, DataLoader
# from scipy.ndimage import gaussian_filter1d
def convert_seconds_to_dhms(seconds):
    # days, remainder = divmod(seconds, 86400)  # 一天有86400秒
    hours, remainder = divmod(seconds, 3600)  # 一小时有3600秒
    minutes, seconds = divmod(remainder, 60)  # 一分钟有60秒
    return hours, minutes, int(seconds)
class Dataset(Dataset):
    def __init__(self, heldin, recon_data, vel, conds, device):
        super().__init__()
        self.heldin = heldin.to(device)
        self.recon_data = recon_data.to(device)
        self.vel = vel.to(device)
        self.conds = conds.to(device)

    def __len__(self):
        return self.heldin.shape[0]

    def __getitem__(self, i):
        return self.heldin[i], self.recon_data[i], self.vel[i], self.conds[i]

def assert_gpu_runtime():
    error_message = '\n\n⚠️ ERROR: GPU is not detected! ⚠️\n\nPlease change the runtime type and re-run the notebook. To do so:\n\tSelect "Runtime" > "Change runtime type" > "Hardware accelerator" > "GPU"'
    sp_call = subprocess.call('nvidia-smi', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # run `nvidia-smi` to see if we can see the GPU, if not raise AssertionError
    assert  sp_call == 0, f'{error_message}'

# function to plot a 3D scatterplot using plotly
def create_3d_scatterplot(data_tuples, xaxis_title, yaxis_title, zaxis_title, fig_size=(700, 500), use_lines=True):
    data = [go.Scatter3d(x=[v[0]], y=[v[1]], z=[v[2]], mode='markers', name=n) for v, n in data_tuples]
    if use_lines:
      data.extend([go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]], mode='lines', line=dict(color='black'), showlegend=False) for v, n in data_tuples])
    data.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker=dict(size=3, color='black'), name='center (0,0,0)'))
    layout_dict = dict(xaxis=dict(title=xaxis_title, range=[-1, 1]),
                       yaxis=dict(title=yaxis_title, range=[-1, 1]),
                       zaxis=dict(title=zaxis_title, range=[-1, 1]),
                       camera=dict(eye=dict(x=1.3, y=-1.3, z=1.3), center=dict(x=0.065, y=0.0, z=-0.075)),
                       aspectmode='cube')
    layout = go.Layout(scene=layout_dict, margin=dict(l=0,r=0,b=0,t=0), width=fig_size[0], height=fig_size[1])
    return go.Figure(data=data, layout=layout).show(config={'displayModeBar': False})


def add_conds_to_trial_data(trial_data_in, dataset_in):
    cond_fields = ['trial_type', 'trial_version']
    combinations = sorted(dataset_in.trial_info[cond_fields].dropna().set_index(cond_fields).index.unique().tolist())
    combinations = np.array(combinations)
    trial_data = trial_data_in.copy()
    trial_info = dataset_in.trial_info
    trial_nums = trial_info.trial_id.values
    trial_data['trial_cond'] = np.zeros(len(trial_data))
    for i,comb in enumerate(combinations):
        # Need a list of all the trial_ids that match cond
        flag1 = trial_info.trial_type.values == comb[0]
        flag2 = trial_info.trial_version.values == comb[1]
        flag3 = np.logical_and(flag1, flag2)
        trial_flag =np.where(flag3) # a list of indices in trial_info that
        cond_trials = trial_nums[trial_flag]
        trial_data.loc[np.isin(trial_data.trial_id, cond_trials),'trial_cond'] = i
    return trial_data


def get_2dataloaders(dataset1, dataset2, device, config_dict):
    spk_field = "spikes"
    hospk_field = "heldout_spikes"
    align_field = "move_onset_time"
    align_window = (-250, 450)
    align_field_fwd = "move_onset_time"
    align_window_fwd = (450, 650)

    dataloaders = {}

    for trial_split in ["train", "val"]:
        split_to_mask = lambda x: (dataset.trial_info.split == x) if isinstance(x, str) else x
        trial_mask = split_to_mask(trial_split)
        allow_nans = trial_split != "train"
        trial_data = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field=align_field,
                                             align_range=align_window)
        trial_data = add_conds_to_trial_data(trial_data, dataset)

        trial_data_fwd = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans,
                                                 align_field=align_field_fwd, align_range=align_window_fwd)
        trial_data_fwd = add_conds_to_trial_data(trial_data_fwd, dataset)

        grouped = list(trial_data.groupby('trial_id', sort=False))
        grouped_fwd = list(trial_data_fwd.groupby('trial_id', sort=False))

        heldin = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped]))
        heldout = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped]))
        heldin_fwd = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped_fwd]))
        heldout_fwd = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped_fwd]))

        vel = torch.Tensor(np.stack([trial["hand_vel"].to_numpy() for _, trial in grouped]))
        conds = torch.Tensor(np.stack([trial["trial_cond"].to_numpy() for _, trial in grouped]))

        heldin_full = torch.cat([heldin, heldin_fwd], dim=1)
        heldout_full = torch.cat([heldout, heldout_fwd], dim=1)
        recon_data = torch.cat([heldin_full, heldout_full], dim=2)

        tensor_dataset = Dataset(heldin, recon_data, vel, conds, device)
        print(tensor_dataset.heldin.shape)
        print(tensor_dataset.recon_data.shape)
        print(tensor_dataset.vel.shape)
        print(tensor_dataset.conds.shape)
        dataloader = DataLoader(tensor_dataset,
                                # batch_size = 25,
                                # num_workers = 0,
                                # shuffle = True
                                batch_size=config_dict['batch_size'],
                                num_workers=config_dict['num_workers'],
                                shuffle=config_dict['shuffle']
                                )

        dataloaders[trial_split] = dataloader

    return dataloaders
def get_dataloaders(dataset, device, config_dict):
    spk_field = "spikes"
    hospk_field="heldout_spikes"
    align_field = "move_onset_time"
    align_window= (-250, 450)
    align_field_fwd= "move_onset_time"
    align_window_fwd= (450,650)

    dataloaders = {}

    for trial_split in ["train", "val"]:
        split_to_mask = lambda x: (dataset.trial_info.split == x) if isinstance(x, str) else x
        trial_mask = split_to_mask(trial_split)
        allow_nans = trial_split != "train"
        trial_data = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field, align_range = align_window)
        trial_data = add_conds_to_trial_data(trial_data, dataset)

        trial_data_fwd = dataset.make_trial_data(ignored_trials=~trial_mask, allow_nans=allow_nans, align_field = align_field_fwd, align_range = align_window_fwd)
        trial_data_fwd = add_conds_to_trial_data(trial_data_fwd, dataset)

        grouped = list(trial_data.groupby('trial_id', sort=False))
        grouped_fwd = list(trial_data_fwd.groupby('trial_id', sort=False))

        heldin = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped]))
        heldout = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped]))
        heldin_fwd = torch.Tensor(np.stack([trial[spk_field].to_numpy() for _, trial in grouped_fwd]))
        heldout_fwd = torch.Tensor(np.stack([trial[hospk_field].to_numpy() for _, trial in grouped_fwd]))

        vel= torch.Tensor(np.stack([trial["hand_vel"].to_numpy() for _, trial in grouped]))
        conds = torch.Tensor(np.stack([trial["trial_cond"].to_numpy() for _, trial in grouped]))

        heldin_full = torch.cat([heldin, heldin_fwd], dim=1)
        heldout_full = torch.cat([heldout, heldout_fwd], dim = 1)
        recon_data = torch.cat([heldin_full, heldout_full], dim =2)

        tensor_dataset = Dataset(heldin, recon_data, vel, conds, device)
        # print(tensor_dataset.heldin.shape)
        # print(tensor_dataset.recon_data.shape)
        # print(tensor_dataset.vel.shape)
        # print(tensor_dataset.conds.shape)
        dataloader = DataLoader(tensor_dataset,
                                # batch_size = 25,
                                # num_workers = 0,
                                # shuffle = True
                                batch_size=config_dict['batch_size'],
                                num_workers=config_dict['num_workers'],
                                shuffle=config_dict['shuffle']
                                )

        dataloaders[trial_split] = dataloader
    
    return dataloaders


def get_submission_inputs(dataset):
    from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors

    train_dict = make_train_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='train', # trial_split=['train', 'val'], for Test phase
                                        save_file=False,
                                        include_forward_pred=True)

    eval_dict = make_eval_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='val', # trial_split='test', for Test phase
                                        save_file=False)

    training_input = torch.Tensor(
        np.concatenate([
            train_dict['train_spikes_heldin'],
            np.zeros(train_dict['train_spikes_heldin_forward'].shape), # zeroed inputs for forecasting
        ], axis=1))

    training_output = torch.Tensor(
        np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2))

    eval_input = torch.Tensor(
        np.concatenate([
            eval_dict['eval_spikes_heldin'],
            np.zeros((
                eval_dict['eval_spikes_heldin'].shape[0],
                train_dict['train_spikes_heldin_forward'].shape[1],
                eval_dict['eval_spikes_heldin'].shape[2]
            )),
        ], axis=1))
    
    tlen = train_dict['train_spikes_heldin'].shape[1]
    num_heldin = train_dict['train_spikes_heldin'].shape[2]
    
    return training_input, training_output, eval_input, tlen, num_heldin

def get_prog_bars(n_epochs):
    from tqdm import tqdm
    bar_format = ('Epochs: {n_fmt} / %s {bar} {percentage:3.0f}%% - ETR:{remaining}' % n_epochs)
    prog_bar = tqdm(
        range(n_epochs),
        unit='epochs',
        desc='Epochs',
        bar_format=bar_format,
        leave=False
    )
    loss_bar = tqdm(
        1,
        unit='',
        desc='',
        bar_format='{desc}',
        leave=False
    )
    return prog_bar, loss_bar

def set_seeds(seed):
    import os
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_submission_inputs(dataset, trial_split='train_val'):
    from nlb_tools.make_tensors import make_train_input_tensors, make_eval_input_tensors

    train_dict = make_train_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='train' if trial_split=='train_val' else ['train', 'val'], 
                                        save_file=False,
                                        include_forward_pred=True)

    eval_dict = make_eval_input_tensors(dataset=dataset,
                                        dataset_name='mc_maze_small',
                                        trial_split='val' if trial_split=='train_val' else 'test', 
                                        save_file=False)

    training_input = torch.Tensor(
        np.concatenate([
            train_dict['train_spikes_heldin'],
            np.zeros(train_dict['train_spikes_heldin_forward'].shape), # zeroed inputs for forecasting
        ], axis=1))

    training_output = torch.Tensor(
        np.concatenate([
            np.concatenate([
                train_dict['train_spikes_heldin'],
                train_dict['train_spikes_heldin_forward'],
            ], axis=1),
            np.concatenate([
                train_dict['train_spikes_heldout'],
                train_dict['train_spikes_heldout_forward'],
            ], axis=1),
        ], axis=2))

    eval_input = torch.Tensor(
        np.concatenate([
            eval_dict['eval_spikes_heldin'],
            np.zeros((
                eval_dict['eval_spikes_heldin'].shape[0],
                train_dict['train_spikes_heldin_forward'].shape[1],
                eval_dict['eval_spikes_heldin'].shape[2]
            )),
        ], axis=1))
    
    tlen = train_dict['train_spikes_heldin'].shape[1]
    num_heldin = train_dict['train_spikes_heldin'].shape[2]
    
    return training_input, training_output, eval_input, tlen, num_heldin

def smooth_gaussian(data, kernel_size):
    for i  in range(data.shape[1]):
        data[:,i] = gaussian_filter1d(data[:,i], kernel_size)
    return data

def add_conds_to_trial_data(trial_data_in, dataset_in):
    cond_fields = ['trial_type', 'trial_version']
    combinations = sorted(dataset_in.trial_info[cond_fields].dropna().set_index(cond_fields).index.unique().tolist())
    combinations = np.array(combinations)
    trial_data = trial_data_in.copy()
    trial_info = dataset_in.trial_info
    trial_nums = trial_info.trial_id.values
    trial_data['trial_cond'] = np.zeros(len(trial_data))
    for i,comb in enumerate(combinations):
        # Need a list of all the trial_ids that match cond
        flag1 = trial_info.trial_type.values == comb[0]
        flag2 = trial_info.trial_version.values == comb[1]
        flag3 = np.logical_and(flag1, flag2)
        trial_flag =np.where(flag3) # a list of indices in trial_info that
        cond_trials = trial_nums[trial_flag]
        trial_data.loc[np.isin(trial_data.trial_id, cond_trials),'trial_cond'] = i
    return trial_data

def plot_cond_avg_fr(trial_data, cond_list, neuron_list, kernel_size):
    n_conds = len(cond_list)
    n_time = 140

    fig, axes = plt.subplots(nrows=5, ncols= 3, figsize = (12, 8), sharex= True)
    mean_fr = np.zeros((n_time, 107, n_conds))
    std_fr = np.zeros((n_time, 107, n_conds))
    for i, cond in enumerate(cond_list):
        trials = trial_data[trial_data.trial_cond == cond]
        grouped = list(trials.groupby('trial_id', sort=False))
        fr = np.float64(np.stack([trial['spikes'].to_numpy() for _, trial in grouped]))
        for j in range(len(grouped)):
            fr[j,:, :] =  np.squeeze(smooth_gaussian(fr[j,:,:], kernel_size))*200
        mean_fr[:,:,i] = np.mean(fr, axis=0)
        std_fr[:,:,i] = np.std(fr, axis=0)
    t = np.linspace(-250, 450, 140)
    for i, n_ind in enumerate(neuron_list):
        ax = axes.flat[i]
        for j, cond in enumerate(cond_list):
            ax.plot(t, mean_fr[:,n_ind,j], linewidth = 3, color = plt.cm.tab10(int(cond)%5), label=f'Condition {cond}')
            ax.fill_between(t,
                            mean_fr[:,n_ind,j] - std_fr[:,n_ind,j],
                            mean_fr[:,n_ind,j] + std_fr[:,n_ind,j],
                            color = plt.cm.tab10(int(cond)%5),
                            alpha=0.2)
        ax.set_title(f"Neuron {n_ind}", fontsize=8, fontweight='medium', pad=10, loc='center', y=0.9)
        ax.axvline(0, linestyle='--', color='k', linewidth=0.75, dashes=(7, 5))
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
    for i in range(5):
        axes[i, 0].set_ylabel("Firing Rate (hz)", fontsize=8)
    for i in range(3):
        axes[4, i].set_xlabel("Time after movement onset (ms)", fontsize=8)
    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.15))
    plt.suptitle(f'Smoothed Spikes - PSTHs for Different Conditions', fontsize=12, fontweight='bold', y=0.97)

def plot_st_fr(trial_data, cond_list, neuron_list, kernel_size):
    fr_dict = {}

    fig, axes = plt.subplots(nrows=5, ncols= 3, figsize = (12, 8), sharex= True)
    for i, cond in enumerate(cond_list):
        trials = trial_data[trial_data.trial_cond == cond]
        grouped = trials.groupby('trial_id', sort=False)
        fr_dict.update({trial_id: np.squeeze(smooth_gaussian(group['spikes'].to_numpy().astype(float), kernel_size))*200 
                        for trial_id, group in grouped})
    t = np.linspace(-250, 450, 140)
    for i, n_ind in enumerate(neuron_list):
        ax = axes.flat[i]
        for j, cond in enumerate(cond_list):
            trials = trial_data[trial_data.trial_cond == cond]
            trial_ids = trials.trial_id.unique()
            for k, trial_id in enumerate(trial_ids):
                trial_fr = fr_dict[trial_id][:, n_ind]
                label=f'Condition {cond}' if k == 0 else None
                ax.plot(t, trial_fr, linewidth = 3, color = plt.cm.tab10(int(cond)%5), alpha=0.75, label=label)
        ax.set_title(f"Neuron {n_ind}", fontsize=8, fontweight='medium', pad=10, loc='center', y=0.9)
        ax.axvline(0, linestyle='--', color='k', linewidth=0.75, dashes=(7, 5))
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
    for i in range(5):
        axes[i, 0].set_ylabel("Firing Rate (hz)", fontsize=8)
    for i in range(3):
        axes[4, i].set_xlabel("Time after movement onset (ms)", fontsize=8)
    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.15))
    plt.suptitle(f'Smoothed Spikes - Single Trial Firing Rates for Different Conditions', fontsize=12, fontweight='bold', y=0.97)

def plot_cond_avg_pred_fr(pred_rates, conds_trial, cond_list, neuron_list):
    rates_plot = pred_rates[:,:140,:]
    t = np.linspace(-250, 350, 140)
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 8), sharex=True)
    fig.suptitle('NDT Inferred Rates - PSTHs for Different Conditions', fontsize=12, fontweight='bold', y=0.97)
    for i, unit_num in enumerate(neuron_list):
        ax = axes.flat[i]
        for j, cond in enumerate(cond_list):
            trials = np.squeeze(rates_plot[conds_trial == cond, :, unit_num])
            mean_trials = np.mean(trials, axis=0)
            std_trials = np.std(trials, axis=0)
            ax.plot(t, mean_trials * 200, linewidth = 3, color = plt.cm.tab10(int(cond)%5), label=f'Condition {cond}')
            ax.fill_between(t, (mean_trials - std_trials) * 200, (mean_trials + std_trials) * 200,
                            color=plt.cm.tab10(int(cond)%5), alpha=0.2)
        ax.set_title(f"Neuron {unit_num}", fontsize=8, fontweight='medium', pad=10, loc='center', y=0.9)
        ax.axvline(0, linestyle='--', color='k', linewidth=0.75, dashes=(7, 5))
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        if i % 3 == 0:
            ax.set_ylabel("Firing Rate (hz)", fontsize=8)
        if i // 3 == 4:
            ax.set_xlabel("Time after movement onset (ms)", fontsize=8)

    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.15))

def plot_pred_st_fr(pred_rates, conds_trial, cond_list, neuron_list):
    rates_plot = pred_rates[:,:140,:]
    t = np.linspace(-250, 350, 140)
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 8), sharex=True)
    fig.suptitle('NDT Inferred Rates - Single Trial Firing Rates for Different Conditions', fontsize=12, fontweight='bold', y=0.97)
    for i, unit_num in enumerate(neuron_list):
        ax = axes.flat[i]
        for j, cond in enumerate(cond_list):
            trials = np.squeeze(rates_plot[conds_trial == cond, :, unit_num])
            for i, trial in enumerate(trials):
                label=f'Condition {cond}' if i == 0 else None
                ax.plot(t, trial * 200, linewidth = 3, color = plt.cm.tab10(int(cond)%5), alpha=0.75, label=label)
        ax.set_title(f"Neuron {unit_num}", fontsize=8, fontweight='medium', pad=10, loc='center', y=0.9)
        ax.axvline(0, linestyle='--', color='k', linewidth=0.75, dashes=(7, 5))
        ax.xaxis.set_tick_params(labelsize=8)
        ax.yaxis.set_tick_params(labelsize=8)
        if i % 3 == 0:
            ax.set_ylabel("Firing Rate (hz)", fontsize=8)
        if i // 3 == 4:
            ax.set_xlabel("Time after movement onset (ms)", fontsize=8)

    axes[0, 1].legend(ncol=4, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.15))

def plot_ex_spikes_vel(spikes, velocity, trial_num, cond):
    fs = 9
    fig, ax = plt.subplots(3, 1, figsize=(8,8), gridspec_kw={'height_ratios': [2.5, 1, 1]})  
    im = ax[0].imshow(spikes[trial_num,:140,:107].T, aspect='auto', interpolation='none')
    ax[0].grid(False)
    xticks_locs = np.linspace(10, 130, num=7)
    xticks_labels = np.linspace(-200, 400, num=7).astype(int)
    ax[0].set_xticks(xticks_locs)
    ax[0].set_xticklabels(xticks_labels)
    ax[0].set_ylabel('Neurons', fontsize=fs)
    ax[0].set_title('Binned Spike Counts', fontsize=fs+1, y=0.99)
    ax[0].axvline(50, linestyle='--', color='w', linewidth=0.75, dashes=(10, 5))
    ax[0].xaxis.set_tick_params(labelsize=fs)
    ax[0].yaxis.set_tick_params(labelsize=fs)
    ax[1].plot(velocity[:, 0], label=f'Condition {int(cond)}', color=plt.cm.tab10(int(cond)%5))
    ax[1].set_title('Hand X Velocity', fontsize=fs+1, y=0.99)
    ax[2].plot(velocity[:, 1], color=plt.cm.tab10(int(cond)%5))
    ax[2].set_title('Hand Y Velocity', fontsize=fs+1, y=0.99)
    ax[0].set_xticks([])
    ax[1].set_xticks([])
    ax[1].legend(fontsize=fs-1, loc='upper right', bbox_to_anchor=(1.15, 0.035))
    for i in range(1, 3):
        ax[i].set_ylabel('Velocity', fontsize=fs)
        ax[i].xaxis.set_tick_params(labelsize=fs)
        ax[i].yaxis.set_tick_params(labelsize=fs)
        ax[i].margins(x=0)
        ax[i].axvline(50, linestyle='--', color='k', linewidth=0.75, dashes=(10, 5))
        ax[i].set_xticks(xticks_locs)
        ax[i].set_xticklabels([])
    ax[2].set_xticklabels(xticks_labels)
    plt.xlabel('Time after movement onset (ms)', fontsize=fs)
    plt.suptitle('Example Trial - Binned Spike Counts and Velocities', fontsize=fs+2, fontweight='bold', y=0.98)
    cbar_ax = fig.add_axes([0.82, 0.58, 0.05, 0.26])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.yaxis.set_tick_params(labelsize=fs)
    plt.subplots_adjust(hspace=0.1, right=0.8, top=0.93, bottom=0.05)
    plt.show()

def plot_hand_vel(conds_to_plot, pos, conds):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    label_plotted = {cond: False for cond in conds_to_plot}
    for hand_pos, cond in zip(pos, conds):
        if cond in conds_to_plot:
            label = f'Condition {int(cond)}' if not label_plotted[cond] else None
            x_values = hand_pos[50:115,0]
            y_values = hand_pos[50:115,1]
            plt.plot(x_values, y_values, color=plt.cm.tab10(int(cond)%5), label=label)
            dx = x_values[-1] - x_values[-2]
            dy = y_values[-1] - y_values[-2]
            plt.arrow(x_values[-1], y_values[-1], dx, dy, shape='full', lw=3, length_includes_head=True, head_width=3, color=plt.cm.tab10(int(cond)%5))
            label_plotted[cond] = True
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    plt.xlabel('X Position', fontsize=8)
    plt.ylabel('Y Position', fontsize=8)
    plt.title("Hand Position per Trial for Different Conditions (First 325ms)", fontsize=11, fontweight='bold', y=1.05)
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0][9:])))
    ax.legend(handles, labels, ncol=4, fontsize=8, loc='center', bbox_to_anchor=(0.5, 1.03))

def compute_similarity(data_tuples):
    # Initialize an empty list to store the names for easy access
    names = []
    
    # For each tuple in the data_tuples list
    for item in data_tuples:
        # Append the name to the names list
        names.append(item[1])
    
    # For each unique combination of two vectors in the data_tuples list
    for i in range(len(data_tuples)):
        for j in range(i+1, len(data_tuples)):
            # Compute the dot product (similarity) and print it
            similarity = np.dot(data_tuples[i][0], data_tuples[j][0])
            print(f'Similarity between {names[i]} and {names[j]}: {similarity:.2f}')
