import os
import sys
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
#sys.path.append(project_root)

#path setting
project = '14s'
model_version = os.path.basename(__file__)[:-3]
home_dir = os.path.expanduser("~")
work_dir = os.path.join(home_dir,'model_output', '14s', model_version, 'analyze')

input_file = os.path.join(work_dir, 'reorganized_predictions.csv')
output_dir = os.pah.join(work_dir, 'change-ratio&amplitude' )


#calculate_settting

