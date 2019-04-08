import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
from xgboost import XGBClassifier,plot_importance
from sklearn.model_selection import GridSearchCV
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

if __name__ == '__main__':
    origin_data = pd.read_table("data/dsjtzs_txfz_training.txt",sep=" ",names=["index","move_data","target","label"])
    print(origin_data.head())
    arrange_data = pd.read_csv("data/alltrain.csv",index_col=False)
    print(arrange_data.head())
    num_p = 0
    num_n = 0
    for i in range(len(origin_data["label"])):
        if origin_data["label"][i] == 1:
            num_p += 1
        else:
            num_n += 1
    print(num_n)


    # 划分训练集和测试集
    y = np.array(arrange_data["label"])
    X = np.array(arrange_data.drop(["label","id"],axis=1))
    np.random.seed(110)
    np.random.shuffle(X)
    np.random.seed(110)
    np.random.shuffle(y)
    X = scale(X)

    # (trainX,testX,trainY,testY) = train_test_split(X,y,test_size=0.2,random_state=42)
    # 利用xgboost进行训练,利用gridsearchCV对xgboost进行调参
    xgboost_clf = XGBClassifier(n_estimators=70,learning_rate=0.1,max_depth=3)
    # param_xg = {"n_estimators":range(70,100,300)}
    # learning_rate=[0.1,0.01,0.05,0.2,0.15]
    # max_depth = [3,4,5,6,7,8,9]
    # param_xg = dict(max_depth=max_depth)
    # gridsearch = GridSearchCV(estimator=xgboost_clf,param_grid=param_xg,scoring="roc_auc",cv=5)
    # gridsearch.fit(trainX,trainY)
    # print("The best n_estimators is {},and its score is {}".format(gridsearch.best_params_,gridsearch.best_score_))


    eval_set = [(testX, testY)]
    xgboost_clf.fit(trainX, trainY, early_stopping_rounds=10,
                    eval_metric="logloss", eval_set=eval_set, verbose=True)
    y_pred = xgboost_clf.predict(testX)
    print(confusion_matrix(testY,y_pred))
    print(classification_report(testY,y_pred))
    model_stability = cross_val_score(xgboost_clf,trainX,trainY,cv=10,scoring="accuracy")
    mean_score_model = model_stability.mean()
    print(mean_score_model)
    # save model
    pickle.dump(xgboost_clf,open("model/xg_model.model","wb"))
    # ouput the feature importance
    print(xgboost_clf.feature_importances_)
    feature_len = len(xgboost_clf.feature_importances_)
    features = xgboost_clf.feature_importances_
    zip_features = zip(features,range(feature_len))
    most_importance_ft = sorted(zip_features,key=lambda x:x[0],reverse=True)[:30]
    print(most_importance_ft)
    plt.bar(range(len(xgboost_clf.feature_importances_)), xgboost_clf.feature_importances_)
    # savefig必须在show函数之前，这样保存的图片不是空白
    plt.savefig("feature_importance.jpg")
    plt.show()




