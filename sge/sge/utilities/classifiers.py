from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import NuSVC, SVC


def ada(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier

	if args['n_estimators'] is None:
		args['n_estimators'] = 50

	if args['learning_rate'] is None:
		args['learning_rate'] = 1.0

	return AdaBoostClassifier(n_estimators=args['n_estimators'], learning_rate=args['learning_rate'], algorithm=args['algorithm'],
							  random_state=42)



def bernouli_nb(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB

	if args['alpha'] is None:
		args['alpha'] = 1.0

	if args['binarize'] is None:
		args['binarize'] = 0.0

	return BernoulliNB(alpha=args['alpha'], binarize=args['binarize'], fit_prior=args['fit_prior'])



def gaussian_nb(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB

	return GaussianNB()



def gradient_boosting(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

	if args['learning_rate'] is None:
		args['learning_rate'] = 0.1

	if args['n_estimators'] is None:
		args['n_estimators'] = 100

	if args['subsample'] is None:
		args['subsample'] = 1.0

	if args['min_weight_fraction_leaf'] is None:
		args['min_weight_fraction_leaf'] = 0.0

	if args['max_depth'] is None:
		args['max_depth'] = 3

	if args['presort'] is None:
		args['presort'] = 'auto'



	return GradientBoostingClassifier(loss=args['loss'], learning_rate=args['learning_rate'], n_estimators=args['n_estimators'], 
									  subsample=args['subsample'], min_weight_fraction_leaf=args['min_weight_fraction_leaf'], 
									  max_depth=args['max_depth'], max_features=args['max_features'],
									  max_leaf_nodes=args['max_leaf_nodes'], warm_start=args['warm_start'], presort=args['presort'],
									  random_state=42)



def knn(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier

	if args['n_neighbors'] is None:
		args['n_neighbors'] = 5

	if args['weights'] is None:
		args['weights'] = 'uniform'

	if args['algorithm'] is None:
		args['algorithm'] = 'auto'

	if args['leaf_size'] is None:
		args['leaf_size'] = 30

	if args['p'] is None:
		args['p'] = 2

	if args['metric'] is None:
		args['metric'] = 'minkowski'


	return KNeighborsClassifier(n_neighbors=args['n_neighbors'], weights=args['weights'], algorithm=args['algorithm'],
								leaf_size=args['leaf_size'], p=args['p'], metric=args['metric'])



def lda(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis

	if args['solver'] is None:
		args['solver'] = 'svd'

	if args['tol'] is None:
		args['tol'] = 0.0001


	return LinearDiscriminantAnalysis(solver=args['solver'], store_covariance=args['store_covariance'], tol=args['tol'])



def logistic(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

	if args['tol'] is None:
		args['tol'] = 1e-4

	if args['solver'] is None:
		args['solver'] = 'liblinear'

	if args['max_iter'] is None:
		args['max_iter'] = 100


	return LogisticRegression(tol=args['tol'], fit_intercept=args['fit_intercept'], 
							  class_weight=args['class_weight'], solver=args['solver'],
							  max_iter=args['max_iter'], warm_start=args['warm_start'],
							  random_state=42)



def logistic_cv(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

	if args['tol'] is None:
		args['tol'] = 1e-4

	if args['solver'] is None:
		args['solver'] = 'lbfgs'

	if args['max_iter'] is None:
		args['max_iter'] = 100

	return LogisticRegressionCV(fit_intercept=args['fit_intercept'], cv=args['cv'], solver=args['solver'],
							    tol=args['tol'], max_iter=args['max_iter'], class_weight=args['class_weight'],
							    refit=args['refit'], random_state=42)



def multinominal_nb(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

	if args['alpha'] is None:
		args['alpha'] = 1.0

	return MultinomialNB(alpha=args['alpha'], fit_prior=args['fit_prior'])



def ncentroid(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid

	return NearestCentroid(metric=args['metric'], shrink_threshold=args['shrink_threshold'])



def passive_aggressive(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier

	if args['loss'] is None:
		args['loss'] = 'hinge'


	return PassiveAggressiveClassifier(fit_intercept=args['fit_intercept'], early_stopping=True, 
									   shuffle=args['shuffle'], loss=args['loss'],
									   warm_start=args['warm_start'], class_weight=args['class_weight'],
									   max_iter=args['max_iter'])


def perceptron(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron

	if args['alpha'] is None:
		args['alpha'] = 0.0001

	if args['tol'] is None:
		args['tol'] = 1e-3

	if args['eta0'] is None:
		args['eta0'] = 1

	return Perceptron(penalty=args['penalty'], alpha=args['alpha'], fit_intercept=args['fit_intercept'],
					  tol=args['tol'], shuffle=args['shuffle'], eta0=args['eta0'], early_stopping=True,
					  class_weight=args['class_weight'], warm_start=args['warm_start'],
					  max_iter=args['max_iter'], random_state=42)



def qda(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis

	if args['tol'] is None:
		args['tol'] = 1e-4

	return QuadraticDiscriminantAnalysis(reg_param=args['reg_param'], store_covariance=args['store_covariance'],
									     tol=args['tol'])



def radius_neighbors(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier

	if args['radius'] is None:
		args['radius'] = 1.0

	if args['leaf_size'] is None:
		args['leaf_size'] = 30

	if args['p'] is None:
		args['p'] = 2

	return RadiusNeighborsClassifier(radius=args['radius'], weights=args['weights'], algorithm=args['algorithm'],
									 leaf_size=args['leaf_size'], p=args['p'], metric=args['metric'], outlier_label=100000)



def ridge(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifier

	if args['alpha'] is None:
		args['alpha'] = 1.0

	if args['tol'] is None:
		args['tol'] = 0.001

	return RidgeClassifier(alpha=args['alpha'], fit_intercept=args['fit_intercept'], normalize=args['normalize'],
						   copy_X=args['copy_X'], max_iter=args['max_iter'], tol=args['tol'],
						   class_weight=args['class_weight'], solver=args['solver'], random_state=42)



def ridge_cv(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV

	if args['alpha'] is None:
		args['alpha'] = (1.0, 1.0, 10.0)
	else:
		alp = float(args['alpha'])
		args['alpha'] = (alp, alp*10, alp*100, alp/10, alp/100)


	return RidgeClassifierCV(alphas=args['alpha'], fit_intercept=args['fit_intercept'],
							 normalize=args['normalize'], scoring=None, cv=args['cv'],
							 class_weight=args['class_weight'])




def sgd(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier

	if args['alpha'] is None:
		args['alpha'] = 0.0001

	if args['l1_ratio'] is None:
		args['l1_ratio'] = 0.15

	if args['tol'] is None:
		args['tol'] = 1e-3

	return SGDClassifier(loss=args['loss'], penalty=args['penalty'], alpha=args['alpha'],
						 l1_ratio=args['l1_ratio'], fit_intercept=args['fit_intercept'],
						 tol=args['tol'], shuffle=args['shuffle'], learning_rate=args['learning_rate'],
						 eta0=args['eta0'], early_stopping=True, class_weight=args['class_weight'],
						 warm_start=args['warm_start'], average=args['average'],
						 max_iter=args['max_iter'], random_state=42)



def random_forest(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier

	if args['n_estimators'] is None:
		args['n_estimators'] = 10

	if args['min_weight_fraction_leaf'] is None:
		args['min_weight_fraction_leaf'] = 0.0

	return RandomForestClassifier(n_estimators=args['n_estimators'], criterion=args['criterion'], max_depth=args['max_depth'],
					    		  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=args['min_weight_fraction_leaf'],
					    		  max_features=args['max_features'], max_leaf_nodes=args['max_leaf_nodes'], bootstrap=args['bootstrap'],
					    		  oob_score=args['oob_score'], warm_start=args['warm_start'], class_weight=args['class_weight'],
					    		  random_state=42)



def extra_trees(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier

	if args['n_estimators'] is None:
		args['n_estimators'] = 10

	if args['min_weight_fraction_leaf'] is None:
		args['min_weight_fraction_leaf'] = 0.0

	return ExtraTreesClassifier(n_estimators=args['n_estimators'], criterion=args['criterion'], max_depth=args['max_depth'],
							    min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=args['min_weight_fraction_leaf'],
							    max_features=args['max_features'], max_leaf_nodes=args['max_leaf_nodes'], bootstrap=args['bootstrap'],
					    		oob_score=args['oob_score'], warm_start=args['warm_start'], class_weight=args['class_weight'],
					    		random_state=42)



def decision_tree(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier

	if args['min_weight_fraction_leaf'] is None:
		args['min_weight_fraction_leaf'] = 0.0

	return DecisionTreeClassifier(criterion=args['criterion'], splitter=args['splitter'], max_depth=args['max_depth'],
								  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=args['min_weight_fraction_leaf'],
								  max_features=args['max_features'], max_leaf_nodes=args['max_leaf_nodes'],
								  class_weight=args['class_weight'], presort=args['presort'], random_state=42)



def extra_tree(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier

	if args['min_weight_fraction_leaf'] is None:
		args['min_weight_fraction_leaf'] = 0.0

	return ExtraTreeClassifier(criterion=args['criterion'], splitter=args['splitter'], max_depth=args['max_depth'],
								  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=args['min_weight_fraction_leaf'],
								  max_features=args['max_features'], max_leaf_nodes=args['max_leaf_nodes'],
								  class_weight=args['class_weight'], random_state=42)



def nu_svc(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC

	if args['degree'] is None:
		args['degree'] = 3

	if args['tol'] is None:
		args['tol'] = 1e-3

	return NuSVC(kernel=args['kernel'], degree=args['degree'], shrinking=args['shrinking'], 
				 probability=args['probability'], tol=args['tol'], class_weight=args['class_weight'],
				 max_iter=args['max_iter'], random_state=42)



def svc(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC

	if args['degree'] is None:
		args['degree'] = 3

	if args['tol'] is None:
		args['tol'] = 1e-3

	return SVC(kernel=args['kernel'], degree=args['degree'], shrinking=args['shrinking'], 
				 probability=args['probability'], tol=args['tol'], class_weight=args['class_weight'],
				 max_iter=args['max_iter'], random_state=42)
