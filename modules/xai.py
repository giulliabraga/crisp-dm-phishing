import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import lime
from lime import lime_tabular
import warnings
import shap


class CompleteXAI():
    def __init__(self, X_train, y_train, X_test, y_test, categorical_features, best_model):
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.categorical_features = categorical_features
        self.best_model = best_model
    
    def get_instances(self):
        X_test, y_test, best_model = self.X_test, self.y_test, self.best_model

        instances = {}

        uncertainties = 1 - best_model.predict_proba(X_test).max(axis=1)

        predictions = best_model.predict(X_test)

        is_prediction_correct = predictions == y_test

        try:
            instances["high_uncertainty_correct_prediction"] = random.choice(
                np.where((uncertainties > 0.40) & is_prediction_correct)[0]
            )
            instances["low_uncertainty_correct_prediction"] = random.choice(
                np.where((uncertainties < 0.15) & is_prediction_correct)[0]
            )
            instances["high_uncertainty_wrong_prediction"] = random.choice(
                np.where((uncertainties > 0.40) & ~is_prediction_correct)[0]
            )
            instances["low_uncertainty_wrong_prediction"] = random.choice(
                np.where((uncertainties < 0.15) & ~is_prediction_correct)[0]
            )
        except IndexError:
            pass  # Simply ignore if any condition has no matches

        return instances, predictions

    
    def init_lime(self):
        # LIME demands you identify which variables are categorical in an array of booleans
        categorical_features_bool = [col in self.categorical_features for col in self.X_test.columns]

        lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_test),
            feature_names=list(self.X_test.columns),
            categorical_features=categorical_features_bool,
            class_names=["0", "1"],
            random_state=42,
        )

        return lime_explainer
    
    def init_shap(self):
        shap_explainer = shap.Explainer(self.best_model.predict_proba, self.X_test)
        shap_values = shap_explainer(self.X_test)

        return shap_explainer, shap_values
    
    def run_local_lime(self):
        lime_explainer = self.init_lime()
        instances, predictions = self.get_instances()

        for key, instance in instances.items():
            print(f'Instância {instance}\n')
            print(f'Variáveis:\n {self.X_test.iloc[instance]}\n')
            print(f'Classe verdadeira {self.y_test.iloc[instance]} e classe predita {predictions[instance]}')
            print(f'Status: \n {key}')

            print('\nIniciando LIME...\n')

            warnings.filterwarnings("ignore")

            exp = lime_explainer.explain_instance(self.X_test.iloc[instance], self.best_model.predict_proba, num_features=10)
            fig_lime = exp.as_pyplot_figure()
            fig_lime.suptitle(f'Instância {instance}', fontsize=16)
            fig_lime.show()

            exp.show_in_notebook(show_table=True)

    def run_local_shap(self):
        shap_explainer, shap_values = self.init_shap()
        instances, predictions = self.get_instances()

        for key, instance in instances.items():
            print(f'Instância {instance}\n')
            print(f'Variáveis:\n {self.X_test.iloc[instance]}\n')
            print(f'Classe verdadeira {self.y_test.iloc[instance]} e classe predita {predictions[instance]}')
            print(f'Status: \n {key}')

            print('Waterfall plot para a classe 1, instância {instance}')
            shap.plots.waterfall(shap_values[instance][:, 1], max_display=20)

    def run_global_shap(self):
        shap_explainer, shap_values = self.init_shap()

        print('Barplot para classe 1')
        shap.plots.bar(shap_values[:,:,1].abs.mean(0))

        print(f'Violin plot para classe 1')
        shap.plots.violin(shap_values[:,:,1], max_display=20, plot_type="layered_violin")