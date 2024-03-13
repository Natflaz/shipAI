from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class VotingEnsembleRegressor:
    def __init__(self, weights=None):
        self.models = [
            ('lightgbm', LGBMRegressor(n_estimators=100, random_state=42)),
            ('catboost', CatBoostRegressor(n_estimators=100, random_state=42))
        ]
        self.voting_regressor = VotingRegressor(estimators=self.models, weights=weights)
        self.multi_output_regressor = MultiOutputRegressor(self.voting_regressor)

    def fit(self, X_train, y_train):
        print("Training Voting Ensemble Regressor with weighted models...")
        self.multi_output_regressor.fit(X_train, y_train)

    def predict(self, X_test):
        print("Predicting with Voting Ensemble Regressor with weighted models...")
        return self.multi_output_regressor.predict(X_test)