from trainer.train_evaluator import TrainEvaluator

class TrainEvaluatorGermEval(TrainEvaluator):

	def __init__(self, *args):
		super(TrainEvaluatorGermEval, self).__init__(*args)
		self.combinations = self._create_gold_combinations(len(self.dataset.target_names), self.dataset.target_size)


	def _create_gold_combinations(self, num_aspects=19, num_sentiments=4):
		classes = range(num_aspects)
		sentiments = range(num_sentiments)
		r = []
		for c in classes:
			for s in sentiments:
				r.append((c, s))

		return r

	def calculate_metrics(self, eval_entries):
		tp = 0
		fp = 0
		tn = 0
		fn = 0

		for cls, sent in self.combinations:
			for (g_cls, g_sent), (p_cls, p_sent) in eval_entries:
				if g_cls == cls and g_sent == sent:
					if g_cls == p_cls and g_sent == p_sent:
						tp += 1
					else:
						fn += 1
				else:
					if p_cls == cls and p_sent == sent:
						fp += 1
					else:
						tn += 1
		return (tp, fp, fn, tn)

	def get_tensor_eval_entries(self, target, prediction):
		eval_entries = []
		for aspect_index, (prediction_aspect, target_aspect) in enumerate(zip([prediction], [target])):
			for y_hat, y in zip(prediction_aspect, target_aspect):
					# y is applicable
					if y != y_hat and y > 0:
						eval_entries.append(((aspect_index, y), (aspect_index, 0)))

						if y_hat > 0:
							eval_entries.append(((aspect_index, 0), (aspect_index, y_hat)))

					elif y == y_hat and y > 0:
						eval_entries.append(((aspect_index, y), (aspect_index, y_hat)))
					elif y_hat > 0:
							eval_entries.append(((aspect_index, 0), (aspect_index, y_hat)))

		return eval_entries

	def calculate_f1(self, target, prediction):
		macro_f1 = 0.0
		f1_scores = []

		# target and prediction are already transposed
		eval_entries = self.get_tensor_eval_entries(target, prediction)
		tp, fp, fn, _ = self.calculate_metrics(eval_entries)

		metrics = {'tp': tp, 'fp': fp, 'fn': fn}
		empty_metric = {'tp': 0, 'fp': 0, 'fn': 0}
		micro_f1 = self.calculate_binary_aspect_f1(metrics)
		return (micro_f1, [micro_f1, micro_f1, micro_f1, micro_f1], [empty_metric, metrics, empty_metric, empty_metric])
