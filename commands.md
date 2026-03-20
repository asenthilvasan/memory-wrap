
The following command uses the PNG filenames as the test indices to analyze.

For each of those images, it will run:

baseline memory set
correct-class-only memory set
random memory sets up to max_random_trials tries

run this inside the scripts directory:
```
python generate_memory_images.py \
  --path_model=../models/2000.pt \
  --dir_dataset=../datasets/ \
  --source_image_dir=../images/mem_images/SVHN/encoder_memory/mobilenet \
  --output_dir=../images/mem_images/SVHN/encoder_memory/mobilenet_testing \
  #or 
  --image_indices=4 \
  --max_random_trials=5
  ```