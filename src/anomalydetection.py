from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetection:
    def __init__(self, contamination=0.05, random_state=42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state
        )
        self.fitted = False

    def fit(self, receipts_features):
        #fits model on receipt features
        X = self._features_to_array(receipts_features)
        self.model.fit(X)
        self.fitted = True

    def predict_score(self, receipt_feature):
        #predicts anomaly score
        #lower score = higher anomaly risk
        if not self.fitted:
            raise ValueError("Model not fitted yet.")
        
        X = self._features_to_array([receipt_feature])
        score = self.model.decision_function(X)[0]
        return score
    
    def predict_label(self, receipt_feature, threshold=50.0):
        #returns 1 if anomaly, 0 if normal
        score = self.predict_score(receipt_feature)
        return 1 if score < threshold else 0
    
    def _features_to_array(self, receipts_features):
        #converts list of features into numpy array for model
        X = []
        for f in receipts_features:
            row = [
                f.get("total_amount", 0.0),
                f.get("num_lines", 0),
                f.get("num_amounts", 0),
                f.get("avg_item_price", 0.0),
                f.get("contains_tax", 0),
                f.get("receipt_length", 0)
            ]
            X.append(row)
        return np.array(X)
    
#sample usage
if __name__ == "__main__":
    from parsing import parse_receipt

    dummy_texts = [
        """TESCO EXTRA
        Date: 12/05/2024
        Milk 5.00
        Bread 3.50
        GST 2.75
        Total RM 45.60""",
        """FAKE STORE
        Date: 12/05/2024
        ItemX 200.00
        Total RM 200.00"""
    ]

    receipts_features = [parse_receipt(text) for text in dummy_texts]

    #initialize model
    detector = AnomalyDetection(contamination=0.5)
    detector.fit(receipts_features)

    #test prediction
    for i, feat in enumerate(receipts_features):
        score = detector.predict_score(feat)
        label = detector.predict_label(feat)
        print(f"Receipt {i}: score={score:.2f}, label={'Fraud' if label else 'Normal'}")
