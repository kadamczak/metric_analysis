from pathlib import Path

# =================
# Datasets
# =================

dataset_root_dir = Path('c:/VisualStudioRepositories/MUSIC_DATA/datasets')

dataset_scene_dir = dataset_root_dir / 'multilabel' / 'scene'
dataset_scene_trimmed_dir = dataset_root_dir / 'multilabel' / 'scene_trimmed'

dataset_emotions_dir = dataset_root_dir / 'multilabel' / 'emotions'
dataset_emotions_trimmed_dir = dataset_root_dir / 'multilabel' / 'emotions_trimmed'

dataset_yeast_dir = dataset_root_dir / 'multilabel' / 'yeast'

dataset_cal500_dir = dataset_root_dir / 'multilabel' / 'cal500'

dataset_enron_dir = dataset_root_dir / 'multilabel' / 'enron'


# =================
# Output
# =================
report_output_root_dir = Path('C:/VisualStudioRepositories/MUSIC_DATA/metric_analysis/output')