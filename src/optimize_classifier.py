import GPy
import GPyOpt

from sklearn.model_selection import StratifiedKFold

import numpy as np

def initialize_classifier(classifier, hyperparameters):
	if(classifier.__name__ == 'SVC'):
		current_cost = float(hyperparameters[:, 0])
		current_kernel = int(hyperparameters[:, 1])

		if(current_kernel):
			current_kernel = 'rbf'
		else:
			current_kernel = 'linear'

		model_name = './models_optimization_logs/' + classifier.__name__ + '_' \
		+ current_kernel + '_' + str(current_cost)

		clf = classifier(C=current_cost, kernel=current_kernel)

	elif(classifier.__name__ == 'KNeighborsClassifier'):
		current_k = int(hyperparameters[:, 0])

		model_name = './models_optimization_logs/' + classifier.__name__ + '_' \
		+ str(current_k)

		clf = classifier(n_neighbors=current_k)


	elif(classifier.__name__ == 'RandomForestClassifier'):
		current_estimators = int(hyperparameters[:, 0])

		model_name = './models_optimization_logs/' + classifier.__name__ + '_' \
		+ str(current_estimators)

		clf = classifier(n_estimators=current_estimators)

	elif(classifier.__name__ == 'XGBClassifier'):
		current_max_depth = int(hyperparameters[:, 0])
		current_learning_rate = float(hyperparameters[:, 1])
		current_estimators = int(hyperparameters[:, 2])

		model_name = './models_optimization_logs/' + classifier.__name__ + '_' \
		+ str(current_max_depth) + '_' + str(current_learning_rate) + '_' \
		+ str(current_estimators)

		clf = classifier(max_depth=current_max_depth,
			learning_rate=current_learning_rate,
			n_estimators=current_estimators)

	return clf, model_name


def optimize_classifier(X, y, classifier, hyperparameters, seed=15):
	def bayesian_optimization_function(x):
		n_splits = 5
		fold = StratifiedKFold(n_splits=n_splits, random_state=seed)

		cross_validation_accuracy = 0

		for train_index, test_index in fold.split(X, y):
			X_train, X_test = np.array(X)[train_index], np.array(X)[test_index]
			y_train, y_test = np.array(y)[train_index], np.array(y)[test_index]

			#TODO: replace by initialize classifier function
			clf, model_name = initialize_classifier(classifier, x)
			
			clf.fit(np.array(X_train), np.array(y_train))

			cross_validation_accuracy += clf.score(X_test, y_test)

		cross_validation_accuracy /= n_splits
		print("Model: " + model_name +
		' | Accuracy: ' + str(cross_validation_accuracy))

		return 1 - cross_validation_accuracy

	optimizer = GPyOpt.methods.BayesianOptimization(
	f=bayesian_optimization_function, domain=hyperparameters, initial_design_numdata=5)

	optimizer.run_optimization(max_iter=10, verbosity=True)

	print("Values for the model should be: ")
	print("optimized parameters: {0}".format(optimizer.x_opt))
	print("optimized eval_accuracy: {0}".format(1 - optimizer.fx_opt))

	return optimizer.x_opt