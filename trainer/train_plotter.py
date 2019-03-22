import os
from typing import Optional

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from misc.utils import create_dir_if_necessary
from trainer.utils import METRIC_F1, METRIC_LOSS, METRIC_PRECISSION, METRIC_RECALL, METRIC_ACCURACY

class TrainPlotter(object):
	
	def __init__(self, image_path: str, max_iteration: int, loss_name:str, experiment_name:str, dataset_name:str):
		self.df = None
		self.image_path = image_path
		self.max_iteration = max_iteration
		self.loss_name = loss_name
		self.experiment_name = experiment_name
		self.dataset_name = dataset_name

		self.confusion_matrix_path = os.path.join(image_path, 'confusion matices')
		self.loss_path = os.path.join(image_path, 'loss curves')
		self.loss_path_general = os.path.join(self.loss_path, 'general')
		self.loss_path_heads = os.path.join(self.loss_path, 'heads')
		self.f1_curves = os.path.join(image_path, 'f1 curves')

		self.loss_series: Optional[pd.core.series.Series] = None
		self.general_f1_series: Optional[pd.core.series.Series] = None
		self.head_f1_series: Optional[pd.core.series.Series] = None

		self._init_folders()
		self._init_sns()

	def _init_sns(self):
		sns.set(style="whitegrid")
		sns.set_color_codes()
		sns.set_context("paper")
		sns.set(rc={"font.size":14, "axes.labelsize":18})

	def _init_folders(self):
		# confusion matrices
		# loss curves
		#	general
		#	heads
		# f1 curves
		create_dir_if_necessary(self.confusion_matrix_path)
		create_dir_if_necessary(self.loss_path)
		create_dir_if_necessary(self.loss_path_general)
		create_dir_if_necessary(self.loss_path_heads)
		create_dir_if_necessary(self.f1_curves)


	def update(self, df: pd.DataFrame):
		self.loss_series = (df['metric type']==METRIC_LOSS)&(df['is general'] == 1.0)
		self.general_f1_series = (df['metric type']==METRIC_F1)&(df['is general'] == 1.0)
		self.head_f1_series = (df['metric type']=='f1')&(df['is general'] == 0.0)&(df['head category'] == '')
		self.df = df		

	def plot(self, format:str='jpg'):
		it = int(self.df['iteration'].iloc[-1])

		path = os.path.join(self.loss_path, f'{it}_loss')
		self.generate_lineplot(self.loss_series, f'{self.experiment_name} train and validation losses for {self.dataset_name}', self.loss_name, path, self.max_iteration, format, 'iterator type')

		path = os.path.join(self.f1_curves, f'{it}_f1')
		self.generate_lineplot(self.general_f1_series, f'{self.experiment_name} train and validation f1 scores for {self.dataset_name}', 'f1 score', path, self.max_iteration, format, 'iterator type')
		
		path = os.path.join(self.loss_path_heads, f'{it}_f1')
		self.generate_barplot(self.head_f1_series, 'head name', f'{self.experiment_name} F1 Scores for individual aspect heads on {self.dataset_name}', 'f1 score', 'aspect', path)

	def generate_lineplot(self, series: pd.core.series.Series, title:str, y_label:str, path:str, max_iterations:int=None, file_format:str='jpg', hue:Optional[str]=None):
		plt.figure(figsize=(10,6))       
		ax = sns.lineplot(x='iteration', y='value', hue=hue, data=self.df[series])
		plt.title(title, fontsize=20) 
		plt.ylabel(y_label)
		plt.xlabel("Iteration")

		if max_iterations is not None:
			plt.xlim((0, max_iterations))
		plt.savefig(f'{path}.{file_format}', format=file_format)

	def generate_barplot(self, series: pd.core.series.Series, x_value:str, title:str, y_label:str, x_label:str, path:str, hue:Optional[str]=None, file_format:str='jpg', angle_xticks:bool=True):
		plt.figure(figsize=(20,10))

		ax = sns.barplot(data=self.df[series], color='b', x=x_value, y='value')
		plt.title(title, fontsize=20) 
		plt.ylabel(y_label)
		plt.xlabel(x_label) 

		if angle_xticks:
			plt.xticks(rotation=45, ha="right")

		plt.savefig(f'{path}.{file_format}', format=file_format)




