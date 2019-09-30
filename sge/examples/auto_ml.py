import warnings
from time import gmtime, strftime
from sge.utilities.classifiers import *
from sge.utilities.preprocessing import *
import random
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
    def __init__(self, invalid_fitness=9999999):
        self.dataset = self.load_data()


    def load_data(self, problem = 'CE'):

        data_train = []
        data_test = []

        if problem == 'glass':    
            for x in range(10):
                data_train.append(pd.read_csv('/home/cdv/Documents/fga/sge3-progsys/sge/data/glass/glass-Training%d.csv' % x, header=0, delimiter=","))
                data_test.append(pd.read_csv('/home/cdv/Documents/fga/sge3-progsys/sge/data/glass/glass-Test%d.csv' % x, header=0, delimiter=","))


        elif problem == 'CE':
            for x in range(10):
                data_train.append(pd.read_csv('/home/cdv/Documents/fga/sge3-progsys/sge/data/CE/CE-Training%d.csv' % x, header=0, delimiter=","))
                data_test.append(pd.read_csv('/home/cdv/Documents/fga/sge3-progsys/sge/data/CE/CE-Test%d.csv' % x, header=0, delimiter=","))


        training_df = pd.concat(data_train, axis=0, ignore_index=True)
        test_df = pd.concat(data_test, axis=0, ignore_index=True)

        objectList = list(training_df.select_dtypes(include=['object']).columns)

        if ('class' in objectList and len(objectList)>=1):
            training_df = training_df.apply(LabelEncoder().fit_transform)
            test_df = test_df.apply(LabelEncoder().fit_transform)

        train_data = training_df.ix[:,:-1].values
        train_target = training_df["class"].values

        test_data = test_df.ix[:,:-1].values
        test_target = test_df["class"].values

        X_train, X_validation, y_train, y_validation = train_test_split(train_data, train_target, train_size=0.7, test_size=0.3, stratify=train_target)

        dataset = {'X_train': X_train, 'X_val': X_validation, 'y_train': y_train, 'y_val': y_validation, 'X_test': test_data, 'y_test': test_target}

        return dataset


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
            pipeline.fit(self.dataset['X_train'], self.dataset['y_train'])
        except ValueError:
            return -0.5, -0.5, None
        except MemoryError:
            return -0.5, -0.5, None

        y_pred = pipeline.predict(self.dataset['X_val'])
        f1_val = metrics.f1_score(self.dataset['y_val'], y_pred, average='weighted')


        y_pred_test = pipeline.predict(self.dataset['X_test'])
        f1_val_test = metrics.f1_score(self.dataset['y_test'], y_pred_test, average='weighted')

        return f1_val, f1_val_test, pipeline.__str__()
        


    def evaluate(self, individual):

        pipeline_modules = self.parse_phenotype(individual)
        f1_val, f1_val_test, pipeline = exec_timeout(func=self.assemble_pipeline, args=[pipeline_modules], timeout=TIMEOUT)

        fitness = 1-f1_val

        return fitness,  {'individual': individual, 'f1score_val': f1_val, 'f1score_test': f1_val_test, 'pipeline': pipeline}

if __name__ == "__main__":
    import sge
    eval_func = AutoML()
    sge.evolutionary_algorithm(evaluation_function=eval_func)
