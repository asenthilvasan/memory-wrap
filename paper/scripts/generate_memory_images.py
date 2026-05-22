import os
import sys
import warnings
# Make `utils` importable regardless of cwd by anchoring on this file's dir.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Silence noisy torch FutureWarnings (weights_only default, deprecated AMP APIs, etc.).
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
import torch # type: ignore
import torchvision # type: ignore

import numpy as np
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patheffects as path_effects # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
# Upper bound on test images to generate. SVHN test has 26k images which is
# far more than needed for qualitative comparison. 0 = no limit (original behaviour).
absl.flags.DEFINE_integer("max_images", 0,
    "Max test images to render (0 = all). Stops after floor(max_images/batch_size_test) batches.")
absl.flags.DEFINE_bool("shuffle_test", True, "Shuffle test images before rendering.")
absl.flags.DEFINE_integer("seed", 42, "Random seed for shuffled visualization sampling.")
absl.flags.DEFINE_integer("top_k", 0,
    "If >0, render only the top_k highest-weighted memory items. 0 = all items with positive weight.")
absl.flags.DEFINE_bool("annotate", True,
    "Overlay each tile with its sparsemax weight and memory-item class label.")
absl.flags.DEFINE_string("variant_name", None,
    "Override the auto-derived variant name used in the save path. Useful when the "
    "checkpoint path does not contain the dataset name (e.g. /root/cinic_run/...).")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS



def run(path:str,dataset_dir:str):
    """ Function to generate memory images for testing images using a given
    model. Memory images show the samples in the memory set that have an
    impact on the current prediction.

    Args:
        path (str): model path
        dataset_dir (str): dir where datasets are stored
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:{}".format(device))    
    # load model
    checkpoint = torch.load(path, map_location=device)
    modality = checkpoint['modality']
    if modality not in ['memory','encoder_memory']:
        raise ValueError(f'Model\'s modality (model type) must be one of [\'memory\',\'encoder_memory\'], not {modality}.')
    dataset_name = checkpoint['dataset_name']
    model = utils.get_model( checkpoint['model_name'],checkpoint['num_classes'],model_type=modality)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()


    # load data
    train_examples = checkpoint['train_examples']
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        name_classes= ['airplane','automobile',	'bird',	'cat','deer','dog',	'frog'	,'horse','ship','truck']
    else:
        name_classes = range(checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=100,size_train=train_examples)
    if FLAGS.shuffle_test:
        generator = torch.Generator()
        generator.manual_seed(FLAGS.seed)
        test_loader = torch.utils.data.DataLoader(
            test_loader.dataset,
            batch_size=batch_size_test,
            shuffle=True,
            generator=generator)
    memory_iter = iter(mem_loader)
    
    #saving stuff
    # Resolve variant_name in priority order:
    #   1) explicit --variant_name flag
    #   2) path component right after dataset_name (original behaviour)
    #   3) path component just above <model_name> — assumes the layout
    #      .../<variant>/<model_name>/<budget>/<seed>.pt used by all known
    #      checkpoint dirs (e.g. /root/cinic_run/models/encoder_memory_supcon_frozen/mobilenet/500/1.pt)
    #   4) modality
    path_parts = os.path.normpath(path).split(os.sep)
    model_name = checkpoint['model_name']
    if FLAGS.variant_name:
        variant_name = FLAGS.variant_name
    elif dataset_name in path_parts:
        dataset_idx = path_parts.index(dataset_name)
        variant_name = (path_parts[dataset_idx + 1]
                        if dataset_idx + 1 < len(path_parts) else modality)
    elif model_name in path_parts:
        model_idx = path_parts.index(model_name)
        variant_name = path_parts[model_idx - 1] if model_idx >= 1 else modality
    else:
        variant_name = modality
    dir_save = "../images/mem_images/"+dataset_name+"/"+variant_name+"/" + model_name + "/"
    if not os.path.isdir(dir_save): 
        os.makedirs(dir_save)

    def get_image(image, revert_norm=True):
        if revert_norm:
            im = undo_normalization(image)
        else:
            im = image
        im = im.squeeze().cpu().detach().numpy()
        transformed_im = np.transpose(im, (1, 2, 0))
        return transformed_im


    # Optional early-stop if --max_images was set. int division so we always
    # render complete batches (no truncated final batch).
    max_batches = FLAGS.max_images // FLAGS.batch_size_test if FLAGS.max_images > 0 else None
    nrow = 4
    grid_padding = 2
    for batch_idx, (images, _) in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        print("Batch:{}/{}".format(batch_idx, len(test_loader)), end='\r')
        try:
            memory, memory_labels = next(memory_iter)
        except StopIteration:
            memory_iter = iter(mem_loader)
            memory, memory_labels = next(memory_iter)

        images = images.to(device)
        memory = memory.to(device)
        memory_labels = memory_labels.to(device)

        # compute output
        outputs,rw = model(images,memory,return_weights=True)
        _, predictions = torch.max(outputs, 1)

        # compute memory outputs
        mem_val,memory_sorted_index = torch.sort(rw,descending=True)
        fig = plt.figure(figsize=(batch_size_test*2, 4),dpi=300)
        columns = batch_size_test
        rows = 2
        for ind in range(len(images)):
            input_selected = images[ind].unsqueeze(0)

            # M_c u M_e : set of sample with a positive impact on prediction
            positive_mask = mem_val[ind] > 0
            m_ec = memory_sorted_index[ind][positive_mask]
            weights_sel = mem_val[ind][positive_mask]
            if FLAGS.top_k > 0:
                m_ec = m_ec[:FLAGS.top_k]
                weights_sel = weights_sel[:FLAGS.top_k]
            labels_sel = memory_labels[m_ec]

            # get reduced memory
            reduced_mem = undo_normalization(memory[m_ec])
            npimg = torchvision.utils.make_grid(
                reduced_mem, nrow=nrow, padding=grid_padding).cpu().numpy()

            # build and store image

            fig.add_subplot(rows, columns, ind+1)
            plt.imshow((get_image(input_selected)* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            plt.title('Prediction:{}'.format(name_classes[predictions[ind]]))
            plt.axis('off')
            ax2 = fig.add_subplot(rows, columns, batch_size_test+1+ind)
            plt.imshow((np.transpose(npimg, (1,2,0))* 255).astype(np.uint8),interpolation='nearest', aspect='equal')
            plt.title('Used Samples')
            plt.axis('off')

            if FLAGS.annotate and m_ec.numel() > 0:
                # Overlay weight + class label on each tile. Tile (i,j) in the
                # make_grid output starts at pixel
                # (grid_padding + j*(W+grid_padding), grid_padding + i*(H+grid_padding)).
                tile_h, tile_w = reduced_mem.shape[-2], reduced_mem.shape[-1]
                stride_h = tile_h + grid_padding
                stride_w = tile_w + grid_padding
                for k in range(m_ec.numel()):
                    row_k, col_k = divmod(k, nrow)
                    x = grid_padding + col_k * stride_w + tile_w / 2
                    y_top = grid_padding + row_k * stride_h + 1
                    y_bot = grid_padding + row_k * stride_h + tile_h - 1
                    w_val = float(weights_sel[k].item())
                    cls_name = name_classes[int(labels_sel[k].item())]
                    txt_top = ax2.text(
                        x, y_top, f"{w_val:.2f}",
                        ha='center', va='top', color='white', fontsize=4)
                    txt_bot = ax2.text(
                        x, y_bot, cls_name,
                        ha='center', va='bottom', color='white', fontsize=4)
                    for t in (txt_top, txt_bot):
                        t.set_path_effects([
                            path_effects.Stroke(linewidth=0.6, foreground='black'),
                            path_effects.Normal()])
        fig.tight_layout()
        fig.savefig(dir_save+str(batch_idx*batch_size_test+ind)+".png")
        plt.close()
        print('Generated {}/{} images'.format(batch_idx,len(test_loader)),end='\r')


def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)