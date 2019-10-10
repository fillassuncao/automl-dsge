import warnings
from time import gmtime, strftime
from sge.utilities.classifiers import *
from sge.utilities.preprocessing import *
import random
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
import numpy as np
import multiprocessing as mp


KEYWORDS = ['preprocessing', 'classifier']
TIMEOUT = 5*60

def exec_timeout(func,args,timeout):
    pool = mp.Pool(1, maxtasksperchild=1)
    result = pool.apply_async(func, args)
    pool.close()

    try:
        s = result.get(timeout)
        return s
    except mp.TimeoutError:
        pool.terminate()
        return -1.0, -1.0, None

def customwarn(message, category, filename, lineno, file=None, line=None):
    with open("log.txt","a+") as file:
        file.write(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +" : "+  warnings.formatwarning(message, category, filename, lineno)+"\n")

warnings.showwarning = customwarn

class AutoML():
    def __init__(self, problem, invalid_fitness=9999999):
        self.problem = problem

    def load_test_data(self, run):

        test_df = pd.read_csv('data/%s/%s-Test%d.csv' % (self.problem, self.problem, run%10), header=0, delimiter=",")

        objectList = list(test_df.select_dtypes(include=['object']).columns)

        if ('class' in objectList and len(objectList)>=1):
            test_df = test_df.apply(LabelEncoder().fit_transform)

        test_data = test_df.ix[:,:-1].values
        test_target = test_df["class"].values

        dataset = {'X_test': test_data, 'y_test': test_target}

        return dataset


    def load_train_data(self, run):
        dataset_train_file = 'data/%s/%s-Training%d.csv' % (self.problem, self.problem, run%10)

        training_df = pd.read_csv(dataset_train_file, header=0, delimiter=",")

        objectList = list(training_df.select_dtypes(include=['object']).columns)

        if ('class' in objectList and len(objectList)>=1):
            training_df = training_df.apply(LabelEncoder().fit_transform)

        train_data = training_df.ix[:,:-1].values
        train_target = training_df["class"].values

        # X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_target, train_size=0.7, test_size=0.3, stratify=train_target)

        self.dataset['X_train'] =  train_data
        self.dataset['y_train'] = train_target
        # self.dataset['X_validation'] =  X_validation
        # self.dataset['y_validation'] = y_validation


    def process_float(self, value):
        min_value, max_value = map(float, value.replace("'", "").replace('RANDFLOAT(','').replace(')','').split(','))
        return random.uniform(min_value, max_value)

    def process_int(self, value):
        min_value, max_value = map(int, value.replace("'", "").replace('RANDINT(','').replace(')','').split(','))
        return random.randint(min_value, max_value)


    def parse_phenotype(self, phenotype):
        modules = []
        new_module = None

        phenotype = phenotype.replace('  ', ' ').replace('\n', '')

        for pair in phenotype.split(' '):
            keyword, value = pair.split(':')

            if keyword in KEYWORDS:
                if new_module is not None:
                    modules.append(new_module)

                new_module = {'module': keyword, 'module_function': eval(value)}

            else:
                try:
                    if 'random' == value:
                        new_module[keyword] = 'random'
                    elif 'RANDFLOAT' in value:
                        new_module[keyword] = self.process_float(value)
                    elif 'RANDINT' in value:
                        new_module[keyword] = self.process_int(value)
                    else:
                        new_module[keyword] = eval(value)
                except NameError:
                    new_module[keyword] = value

        modules.append(new_module)

        return modules



    def assemble_pipeline(self, modules):

        pipeline_methods = []
        for module in modules:
            pipeline_methods.append(module['module_function'](module))        

        try:
            pipeline = make_pipeline(*pipeline_methods)
        except Exception:
            return -0.5, -0.5, None

        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True)
            scores = cross_val_score(pipeline, self.dataset['X_train'], self.dataset['y_train'], cv=cv, scoring='f1_weighted')
        except ValueError:
            return -0.5, -0.5, None
        except MemoryError:
            return -0.5, -0.5, None

        pipeline.fit(self.dataset['X_train'], self.dataset['y_train'])
        y_pred_test = pipeline.predict(self.dataset['X_test'])
        f1_val_test = metrics.f1_score(self.dataset['y_test'], y_pred_test, average='weighted')


        return np.mean(scores), f1_val_test, pipeline.__str__()
        


    def evaluate(self, individual):

        pipeline_modules = self.parse_phenotype(individual)
        try:
            f1_val, f1_val_test, pipeline = exec_timeout(func=self.assemble_pipeline, args=[pipeline_modules], timeout=TIMEOUT)
        except:
            return 9999, {'individual':individual, 'invalid': 1}

        fitness = 1-f1_val

        invalid = 0
        if f1_val < 0:
            invalid = 1

        return fitness,  {'individual': individual, 'f1score_val': f1_val, 'f1score_test': f1_val_test, 'pipeline': pipeline, 'invalid': invalid}

if __name__ == "__main__":
    import sge
    eval_func = AutoML(problem='CE')
    sge.evolutionary_algorithm(evaluation_function=eval_func)
