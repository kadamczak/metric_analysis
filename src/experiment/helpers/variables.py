from pathlib import Path

# =================
# Datasets
# =================

dataset_root_dir = Path('c:\VisualStudioRepositories\MUSIC_DATA\datasets')

dataset_scene_dir = dataset_root_dir / 'multilabel' / 'scene'
dataset_scene_trimmed_dir = dataset_root_dir / 'multilabel' / 'scene_trimmed'


# =================
# Output
# =================
report_output_root_dir = Path('C:\VisualStudioRepositories\MUSIC_DATA\metric_analysis\output')

report_output_dirs = {
    name: report_output_root_dir / name
    for name in [
        'binary_balanced', 'binary_unbalanced',
        'multiclass_balanced', 'multiclass_unbalanced',
        'multilabel_balanced', 'multilabel_unbalanced'
    ]
}