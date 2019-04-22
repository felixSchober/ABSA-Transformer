from trainer.early_stopping import EarlyStopping

class VariableEarlyStopping(EarlyStopping):
	def __init__(self, optimizer: torch.nn.Module, model: torch.nn.Module, hp: RunConfiguration, evaluator: TrainEvaluator, checkpoint_dir: str):
		super(VariableEarlyStopping, self).__init__(optimizer, model, hp, evaluator, checkpoint_dir)
		
