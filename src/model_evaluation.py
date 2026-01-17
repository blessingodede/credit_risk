
import matplotlib.pyplot as plt

from sklearn.metrics import(
    roc_auc_score, roc_curve, classification_report
)

class ModelEvaluation:
    
    def __init__(self, models):
        self.models = models
    
    def evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        plt.figure()
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_proba)
            fpr, tpr, _ = roc_curve(y_test, y_proba)
          
            
            plt.plot(fpr, tpr, label=f"\n{name} (AUC: {auc: .3f})")
            
            results[name] = {
                "AUC": auc,
                "Report": classification_report(y_test, y_pred, output_dict=False)
            }
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()
        
        return results
            
            
            
            
    