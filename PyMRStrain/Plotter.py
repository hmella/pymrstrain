import matplotlib.pyplot as plt
import numpy as np

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volumes,caxis=None):
    remove_keymap_conflicts({'j', 'k'})
    volumes = [np.transpose(volume, (1,0,2)) for volume in volumes]
    fig, ax = plt.subplots(1, len(volumes))
    for (i, volume) in enumerate(volumes):
      ax[i].volume = volume
      ax[i].index = 0
      if caxis==None:
        ax[i].imshow(volume[...,ax[i].index], cmap=plt.get_cmap('Greys_r'), vmin=volume.min(), vmax=volume.max())
      else:
        ax[i].imshow(volume[...,ax[i].index], cmap=plt.get_cmap('Greys_r'), vmin=caxis[0], vmax=caxis[1])
      ax[i].invert_yaxis()

    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    for ax in fig.axes:
      if event.key == 'j':
          previous_slice(ax)
      elif event.key == 'k':
          next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    """Go to the previous slice."""
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[2]  # wrap around using %
    ax.images[0].set_array(volume[...,ax.index])

def next_slice(ax):
    """Go to the next slice."""
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[2]
    ax.images[0].set_array(volume[...,ax.index])