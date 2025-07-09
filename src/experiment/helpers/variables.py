from pathlib import Path

report_output_root_dir = Path('../../output')

report_output_dirs = { 'binary_balanced': report_output_root_dir / 'binary_balanced',
                       'binary_unbalanced': report_output_root_dir / 'binary_unbalanced',
                       
                       'multiclass_balanced': report_output_root_dir / 'multiclass_balanced',          
                       'multiclass_unbalanced': report_output_root_dir / 'multiclass_unbalanced',
                       
                       'multilabel_balanced': report_output_root_dir / 'multilabel_balanced',
                       'multilabel_unbalanced': report_output_root_dir / 'multilabel_unbalanced'}