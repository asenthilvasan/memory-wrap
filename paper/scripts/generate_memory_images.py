import sys
sys.path.append('..')
import json
import os
import random
from typing import Dict, List, Sequence

import torch # type: ignore
import torchvision # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore
import absl.flags
import absl.app
import utils.datasets as datasets
import utils.utils as utils

# user flags
absl.flags.DEFINE_string("path_model", None, "Path of the trained model")
absl.flags.DEFINE_integer("batch_size_test", 3, "Number of samples for each image")
absl.flags.DEFINE_string("dir_dataset", '../datasets/', "dir path where datasets are stored")
absl.flags.DEFINE_string("source_image_dir", None, "Directory containing existing generated PNG files whose filenames are test-set indices to analyze")
absl.flags.DEFINE_string("output_dir", None, "Directory where generated testing images and analysis.json will be saved")
absl.flags.DEFINE_integer("num_wrong_predictions", 20, "Number of wrong predictions to analyze when image_indices is empty")
absl.flags.DEFINE_list("image_indices", [], "Comma-separated list of test-set indices to analyze")
absl.flags.DEFINE_integer("max_random_trials", 100, "Maximum number of random memory sets to test for each selected image")
absl.flags.DEFINE_integer("random_seed", 0, "Seed used to sample memory sets")
absl.flags.DEFINE_integer("memory_size", 0, "Memory-set size to evaluate. If 0, use the size stored in the checkpoint")
absl.flags.mark_flag_as_required("path_model")

FLAGS = absl.flags.FLAGS


def get_dataset_labels(dataset) -> List[int]:
    if isinstance(dataset, torch.utils.data.Subset):
        parent_labels = get_dataset_labels(dataset.dataset)
        return [int(parent_labels[index]) for index in dataset.indices]
    if hasattr(dataset, 'targets'):
        labels = dataset.targets
    elif hasattr(dataset, 'labels'):
        labels = dataset.labels
    elif hasattr(dataset, 'samples'):
        labels = [sample[1] for sample in dataset.samples]
    else:
        return [int(dataset[index][1]) for index in range(len(dataset))]
    if torch.is_tensor(labels):
        return [int(item) for item in labels.tolist()]
    if isinstance(labels, np.ndarray):
        return [int(item) for item in labels.tolist()]
    return [int(item) for item in list(labels)]


def get_class_names(dataset_name: str, num_classes: int) -> List[str]:
    if dataset_name == 'CIFAR10' or dataset_name == 'CINIC10':
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return [str(index) for index in range(num_classes)]


def get_indices_from_image_dir(image_dir: str) -> List[int]:
    if not os.path.isdir(image_dir):
        raise ValueError(f'source image directory does not exist: {image_dir}')
    indices = []
    for filename in os.listdir(image_dir):
        stem, extension = os.path.splitext(filename)
        if extension.lower() != '.png':
            continue
        if not stem.isdigit():
            continue
        indices.append(int(stem))
    if len(indices) == 0:
        raise ValueError(f'no PNG files with numeric filenames were found in {image_dir}')
    return sorted(set(indices))


def build_indices_by_class(labels: Sequence[int], num_classes: int) -> Dict[int, List[int]]:
    indices_by_class = {class_index: [] for class_index in range(num_classes)}
    for index, label in enumerate(labels):
        indices_by_class[int(label)].append(index)
    return indices_by_class


def sample_memory_indices(candidate_indices: Sequence[int], memory_size: int, rng: random.Random) -> List[int]:
    candidate_indices = list(candidate_indices)
    if len(candidate_indices) == 0:
        return []
    if memory_size <= 0 or memory_size >= len(candidate_indices):
        return list(candidate_indices)
    return rng.sample(candidate_indices, memory_size)


def materialize_memory(dataset, dataset_labels: Sequence[int], dataset_indices: Sequence[int], device: torch.device):
    if len(dataset_indices) == 0:
        raise ValueError('memory set cannot be empty.')
    memory = torch.stack([dataset[index][0] for index in dataset_indices]).to(device)
    memory_labels = torch.tensor([int(dataset_labels[index]) for index in dataset_indices], dtype=torch.long)
    return memory, memory_labels


def get_distribution(labels: Sequence[int], class_names: Sequence[str]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for label in labels:
        label_name = class_names[int(label)]
        distribution[label_name] = distribution.get(label_name, 0) + 1
    return distribution


def evaluate_memory_set(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, memory: torch.Tensor, memory_labels: torch.Tensor, memory_dataset_indices: Sequence[int], class_names: Sequence[str]):
    with torch.no_grad():
        outputs, rw = model(images, memory, return_weights=True)
        probabilities = torch.softmax(outputs, dim=1)
    predictions = torch.argmax(outputs, dim=1)
    mem_val, memory_sorted_index = torch.sort(rw, descending=True)
    memory_label_values = [int(item) for item in memory_labels.tolist()]
    full_distribution = get_distribution(memory_label_values, class_names)
    sample_reports = []
    for ind in range(len(images)):
        positive_local_indices_tensor = memory_sorted_index[ind][mem_val[ind] > 0].detach().cpu()
        if positive_local_indices_tensor.numel() == 0:
            positive_local_indices_tensor = memory_sorted_index[ind][:min(1, memory.shape[0])].detach().cpu()
        positive_local_indices = [int(item) for item in positive_local_indices_tensor.tolist()]
        positive_dataset_indices = [int(memory_dataset_indices[item]) for item in positive_local_indices]
        positive_labels = [memory_label_values[item] for item in positive_local_indices]
        top_local_indices = [int(item) for item in memory_sorted_index[ind][:min(5, memory.shape[0])].detach().cpu().tolist()]
        top_weights = [float(item) for item in mem_val[ind][:min(5, memory.shape[0])].detach().cpu().tolist()]
        predicted_label = int(predictions[ind].item())
        true_label = int(labels[ind].item())
        sample_reports.append({
            'prediction': predicted_label,
            'prediction_name': class_names[predicted_label],
            'prediction_confidence': float(probabilities[ind][predicted_label].item()),
            'correct': predicted_label == true_label,
            'memory_size': int(memory.shape[0]),
            'memory_dataset_indices': [int(item) for item in memory_dataset_indices],
            'memory_class_distribution': full_distribution,
            'positive_memory_count': len(positive_local_indices),
            'positive_memory_local_indices': positive_local_indices,
            'positive_memory_dataset_indices': positive_dataset_indices,
            'positive_memory_class_distribution': get_distribution(positive_labels, class_names),
            'true_class_count_in_memory': int(sum(label == true_label for label in memory_label_values)),
            'true_class_count_in_positive_memory': int(sum(label == true_label for label in positive_labels)),
            'predicted_class_count_in_memory': int(sum(label == predicted_label for label in memory_label_values)),
            'predicted_class_count_in_positive_memory': int(sum(label == predicted_label for label in positive_labels)),
            'top_weighted_memory': [
                {
                    'rank': rank + 1,
                    'local_index': local_index,
                    'dataset_index': int(memory_dataset_indices[local_index]),
                    'label': memory_label_values[local_index],
                    'label_name': class_names[memory_label_values[local_index]],
                    'weight': top_weights[rank],
                }
                for rank, local_index in enumerate(top_local_indices)
            ],
        })
    return {
        'predictions': predictions.detach().cpu(),
        'weights': rw.detach().cpu(),
        'sample_reports': sample_reports,
    }


def save_memory_figure(sample_info: Dict[str, object], scenario_bundles: Sequence[Dict[str, object]], dir_save: str, undo_normalization):
    def get_image(image, revert_norm: bool = True):
        if image.dim() == 4:
            image = image.squeeze(0)
        if revert_norm:
            image = undo_normalization(image)
        image = image.cpu().detach().numpy()
        return np.transpose(image, (1, 2, 0))

    def get_memory_grid(memory: torch.Tensor, positive_indices: Sequence[int]):
        selected_indices = list(positive_indices)
        if len(selected_indices) == 0:
            selected_indices = [0]
        restored_memory = torch.stack([undo_normalization(memory[index]) for index in selected_indices])
        npimg = torchvision.utils.make_grid(restored_memory, nrow=min(4, len(restored_memory))).cpu().numpy()
        return np.transpose(npimg, (1, 2, 0))

    rows = len(scenario_bundles)
    fig = plt.figure(figsize=(6, rows * 3), dpi=300)
    for row, scenario_bundle in enumerate(scenario_bundles):
        record = scenario_bundle['record']
        memory = scenario_bundle['memory']
        fig.add_subplot(rows, 2, row * 2 + 1)
        plt.imshow((np.clip(get_image(sample_info['image']), 0.0, 1.0) * 255).astype(np.uint8), interpolation='nearest', aspect='equal')
        plt.title('#{} {}\nTrue:{} Pred:{}'.format(sample_info['image_index'], scenario_bundle['title'], sample_info['label_name'], record['prediction_name']))
        plt.axis('off')
        fig.add_subplot(rows, 2, row * 2 + 2)
        plt.imshow((np.clip(get_memory_grid(memory, record['positive_memory_local_indices']), 0.0, 1.0) * 255).astype(np.uint8), interpolation='nearest', aspect='equal')
        plt.title('Used Samples ({}/{})'.format(record['positive_memory_count'], record['memory_size']))
        plt.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(dir_save, '{}.png'.format(sample_info['image_index'])))
    plt.close(fig)



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
    name_classes = get_class_names(dataset_name, checkpoint['num_classes'])
    load_dataset = getattr(datasets, 'get_'+dataset_name)
    undo_normalization = getattr(datasets, 'undo_normalization_'+dataset_name)
    batch_size_test = FLAGS.batch_size_test
    memory_size = FLAGS.memory_size if FLAGS.memory_size > 0 else checkpoint['mem_examples']
    _, _, test_loader, mem_loader = load_dataset(dataset_dir,batch_size_train=50, batch_size_test=batch_size_test,batch_size_memory=memory_size,size_train=train_examples)
    ordered_test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size_test, pin_memory=True, shuffle=False)
    memory_dataset = mem_loader.dataset
    memory_dataset_labels = get_dataset_labels(memory_dataset)
    memory_pool_indices = list(range(len(memory_dataset_labels)))
    indices_by_class = build_indices_by_class(memory_dataset_labels, checkpoint['num_classes'])

    #saving stuff
    if FLAGS.output_dir is not None:
        dir_save = os.path.abspath(FLAGS.output_dir)
    else:
        dir_save = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'images', 'mem_images', dataset_name, modality, checkpoint['model_name']))
    if not os.path.isdir(dir_save):
        os.makedirs(dir_save)

    rng = random.Random(FLAGS.random_seed)
    explicit_image_indices = [int(item) for item in FLAGS.image_indices if item != '']
    if FLAGS.source_image_dir is not None and len(explicit_image_indices) > 0:
        raise ValueError('Use either source_image_dir or image_indices, not both.')
    if FLAGS.source_image_dir is not None:
        requested_indices = get_indices_from_image_dir(FLAGS.source_image_dir)
        print('Using {} existing generated images from {}'.format(len(requested_indices), FLAGS.source_image_dir))
    else:
        requested_indices = explicit_image_indices
    selected_samples = []

    if requested_indices:
        test_dataset = ordered_test_loader.dataset
        for image_index in requested_indices:
            if image_index < 0 or image_index >= len(test_dataset):
                raise ValueError(f'image index {image_index} is outside the test dataset range [0, {len(test_dataset)-1}].')
            image, label = test_dataset[image_index]
            baseline_memory_indices = sample_memory_indices(memory_pool_indices, memory_size, rng)
            baseline_memory, baseline_memory_labels = materialize_memory(memory_dataset, memory_dataset_labels, baseline_memory_indices, device)
            baseline_result = evaluate_memory_set(model, image.unsqueeze(0).to(device), torch.tensor([label], device=device), baseline_memory, baseline_memory_labels, baseline_memory_indices, name_classes)
            selected_samples.append({
                'image_index': image_index,
                'image': image.cpu(),
                'label': int(label),
                'label_name': name_classes[int(label)],
                'baseline_record': baseline_result['sample_reports'][0],
                'baseline_memory': baseline_memory.detach().cpu(),
            })
    else:
        for batch_idx, (images, labels) in enumerate(ordered_test_loader):
            print("Batch:{}/{}".format(batch_idx, len(ordered_test_loader)), end='\r')
            baseline_memory_indices = sample_memory_indices(memory_pool_indices, memory_size, rng)
            baseline_memory, baseline_memory_labels = materialize_memory(memory_dataset, memory_dataset_labels, baseline_memory_indices, device)
            images_device = images.to(device)
            labels_device = labels.to(device)

            # compute output
            batch_result = evaluate_memory_set(model, images_device, labels_device, baseline_memory, baseline_memory_labels, baseline_memory_indices, name_classes)

            # only keep wrongly predicted images
            for ind in range(len(images)):
                if batch_result['sample_reports'][ind]['correct']:
                    continue
                img_index = batch_idx * batch_size_test + ind
                selected_samples.append({
                    'image_index': img_index,
                    'image': images[ind].cpu(),
                    'label': int(labels[ind]),
                    'label_name': name_classes[int(labels[ind])],
                    'baseline_record': batch_result['sample_reports'][ind],
                    'baseline_memory': baseline_memory.detach().cpu(),
                })
                if len(selected_samples) >= FLAGS.num_wrong_predictions:
                    break
            if len(selected_samples) >= FLAGS.num_wrong_predictions:
                break

    analysis_records = []
    corrected_random_sets = 0
    for sample_index, sample_info in enumerate(selected_samples):
        image = sample_info['image'].unsqueeze(0).to(device)
        label = torch.tensor([sample_info['label']], device=device)
        scenario_bundles = [{
            'title': 'Original memory',
            'record': sample_info['baseline_record'],
            'memory': sample_info['baseline_memory'],
        }]

        class_memory_indices = sample_memory_indices(indices_by_class[sample_info['label']], memory_size, rng)
        if class_memory_indices:
            class_memory, class_memory_labels = materialize_memory(memory_dataset, memory_dataset_labels, class_memory_indices, device)
            class_result = evaluate_memory_set(model, image, label, class_memory, class_memory_labels, class_memory_indices, name_classes)
            scenario_bundles.append({
                'title': 'Correct-class memory',
                'record': class_result['sample_reports'][0],
                'memory': class_memory.detach().cpu(),
            })
            class_analysis = class_result['sample_reports'][0]
        else:
            class_analysis = {
                'available': False,
                'message': 'No samples for the true class were available in the memory pool.',
            }

        random_record = {
            'found': False,
            'trials_attempted': 0,
            'record': None,
        }
        if not sample_info['baseline_record']['correct']:
            for trial in range(1, FLAGS.max_random_trials + 1):
                random_memory_indices = sample_memory_indices(memory_pool_indices, memory_size, rng)
                random_memory, random_memory_labels = materialize_memory(memory_dataset, memory_dataset_labels, random_memory_indices, device)
                random_result = evaluate_memory_set(model, image, label, random_memory, random_memory_labels, random_memory_indices, name_classes)
                if random_result['sample_reports'][0]['correct']:
                    random_record = {
                        'found': True,
                        'trials_attempted': trial,
                        'record': random_result['sample_reports'][0],
                    }
                    corrected_random_sets += 1
                    scenario_bundles.append({
                        'title': 'Random corrected ({})'.format(trial),
                        'record': random_result['sample_reports'][0],
                        'memory': random_memory.detach().cpu(),
                    })
                    break
            if not random_record['found']:
                random_record['trials_attempted'] = FLAGS.max_random_trials

        analysis_records.append({
            'image_index': int(sample_info['image_index']),
            'true_label': int(sample_info['label']),
            'true_label_name': sample_info['label_name'],
            'baseline': sample_info['baseline_record'],
            'correct_class_memory': class_analysis,
            'random_search': random_record,
        })
        save_memory_figure(sample_info, scenario_bundles, dir_save, undo_normalization)
        print('Generated {}/{} images'.format(sample_index + 1, len(selected_samples)), end='\r')

    with open(os.path.join(dir_save, 'analysis.json'), 'w', encoding='utf-8') as analysis_file:
        json.dump(analysis_records, analysis_file, indent=2)
    print('\nSaved {} analyses to {}. Found {} correcting random memory sets.'.format(len(analysis_records), dir_save, corrected_random_sets))


def main(argv):

    run(FLAGS.path_model,FLAGS.dir_dataset)

if __name__ == '__main__':
  absl.app.run(main)