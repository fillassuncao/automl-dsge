from sklearn.preprocessing import Binarizer, Imputer, MaxAbsScaler, MinMaxScaler, Normalizer, PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import FastICA, IncrementalPCA, PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.feature_selection import SelectPercentile, RFE, RFECV, SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectFromModel, VarianceThreshold


def binarizer(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer

	return Binarizer(threshold=args['threshold'], copy=True)



def feature_agglomeration(args):
	#http://#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.FeatureAgglomeration.html

	if args['n_clusters'] is None:
		args['n_clusters'] = 2


	return FeatureAgglomeration(n_clusters=args['n_clusters'], affinity=args['affinity'],
							    compute_full_tree=args['compute_full_tree'], linkage=args['linkage'])



def fast_ica(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

	if args['tol'] is None:
		args['tol'] = 0.0001

	if args['max_iter'] is None:
		args['max_iter'] = 100

	return FastICA(n_components=args['n_components'], algorithm=args['algorithm'], 
				   whiten=args['whiten'], fun=args['fun'], max_iter=args['max_iter'],
				   tol=args['tol'], random_state=42)



def gaussian_random_projection(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html

	if args['n_components'] is None:
		args['n_components'] = 'auto'

	if args['eps'] is None:
		args['eps'] = 0.1

	return GaussianRandomProjection(n_components=args['n_components'], eps=args['eps'], random_state=42)



def imputer(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html
	#TODO: deprecated

	return Imputer(missing_values='NaN', strategy=args['strategy'], axis=0, copy=True)



def incremental_pca(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

	return IncrementalPCA(n_components=args['n_components'], whiten=args['whiten'], copy=True)



def max_abs(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html

	return MaxAbsScaler(copy=True)



def min_max(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

	return MinMaxScaler(feature_range=(0, 1), copy=True)



def normalizer(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

	return Normalizer(norm=args['norm'], copy=True)



def nystroem(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html


	return Nystroem(kernel=args['kernel'], gamma=args['gamma'], coef0=args['coef0'],
					degree=args['degree'], n_components=args['n_components'],
					random_state=42)	



def pca(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

	return PCA(n_components=args['n_components'], whiten=args['whiten'], copy=True,
			   random_state=42)



def polynomial_features(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

	if args['degree'] is None:
		args['degree'] = 2

	return PolynomialFeatures(degree=args['degree'], interaction_only=args['interaction_only'],
							  include_bias=args['include_bias'])



def rbf_sampler(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.RBFSampler.html

	if args['gamma'] is None:
		args['gamma'] = 1.0

	if args['n_components'] is None:
		args['n_components'] = 100

	return RBFSampler(gamma=args['gamma'], n_components=args['n_components'], random_state=42)



def percentile(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html

	from sklearn.feature_selection import f_classif, chi2

	if args['percentile'] is None:
		args['percentile'] = 10

	if args['score_function'] == 'chi2':
		args['score_function'] = chi2
	elif args['score_function'] == 'f_classif':
		args['score_function'] = f_classif
 
	return SelectPercentile(score_func=args['score_function'], percentile=args['percentile'])



def rfe(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

	from sklearn.svm import SVR

	if args['step'] is None:
		args['step'] = 1

	return RFE(estimator=SVR(kernel='linear'), n_features_to_select=args['n_features_to_select'],
			   step=args['step'])



def rfe_cv(args):
	# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

	from sklearn.svm import SVR

	if args['step'] is None:
		args['step'] = 1


	return RFECV(estimator=SVR(kernel='linear'), step=args['step'], cv=args['cv'], scoring=args['scoring'])



def standard_scaler(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

	return StandardScaler(with_mean=args['with_mean'], with_std=args['with_std'], copy=True)



def robust_scaler(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html

	return RobustScaler(with_centering=args['with_centering'], with_scaling=args['with_scaling'], copy=True)



def select_fdr(args):
	# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html

	from sklearn.feature_selection import f_classif, chi2

	if args['alpha'] is None:
		args['alpha'] = 0.05

	if args['score_function'] == 'chi2':
		args['score_function'] = chi2
	elif args['score_function'] == 'f_classif':
		args['score_function'] = f_classif

	return SelectFdr(score_func=args['score_function'], alpha=args['alpha'])



def select_fpr(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html

	from sklearn.feature_selection import f_classif, chi2

	if args['alpha'] is None:
		args['alpha'] = 0.05

	if args['score_function'] == 'chi2':
		args['score_function'] = chi2
	elif args['score_function'] == 'f_classif':
		args['score_function'] = f_classif

	return SelectFpr(score_func=args['score_function'], alpha=args['alpha'])



def select_fwe(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html

	from sklearn.feature_selection import f_classif, chi2

	if args['alpha'] is None:
		args['alpha'] = 0.05

	if args['score_function'] == 'chi2':
		args['score_function'] = chi2
	elif args['score_function'] == 'f_classif':
		args['score_function'] = f_classif

	return SelectFwe(score_func=args['score_function'], alpha=args['alpha'])



def select_k_best(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html

	from sklearn.feature_selection import f_classif, chi2

	if args['k'] is None:
		args['k'] = 10

	if args['score_function'] == 'chi2':
		args['score_function'] = chi2
	elif args['score_function'] == 'f_classif':
		args['score_function'] = f_classif

	return SelectKBest(score_func=args['score_function'], k=args['k'])



def select_from_model(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html

	from sklearn.svm import SVR

	return SelectFromModel(estimator=SVR(kernel='linear'), threshold=args['threshold'], prefit=args['prefit'])



def sparse_random_projection(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.SparseRandomProjection.html

	if args['eps'] is None:
		args['eps'] = 0.1

	return SparseRandomProjection(n_components=args['n_components'], density=args['density'], eps=args['eps'],
								  dense_output=args['dense_output'])



def variance_threshold(args):
	#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html

	return VarianceThreshold()



def truncated_svd(args):
	# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

	if args['n_components'] is None:
		args['n_components'] = 2

	if args['n_iter'] is None:
		args['n_iter'] = 5

	return TruncatedSVD(n_components=args['n_components'], algorithm=args['algorithm'], n_iter=args['n_iter'],
						tol=args['tol'], random_state=42)