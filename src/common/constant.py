import os
class PATH:
    base_dir = os.getcwd() 
    resources = os.path.join(base_dir, 'resources')
    img = os.path.join(resources, 'img')
    model = os.path.join(resources, 'model')
    csv = os.path.join(resources, 'csv')
    config = os.path.join(base_dir, 'config.yaml')
    #model file
    model_tf = os.path.join(model, 'model_tf')
    model_sklearn = os.path.join(model, 'model_sklearn.sav')
    model_stat = os.path.join(model, 'model_stat.pkl')
    #img file
    RF_img = os.path.join(img, 'Random_forest.png')
    SVR_img = os.path.join(img, 'SVR.png')
    #csv file
    eval_df = os.path.join(csv, 'eval_df.csv')
    visualize_df = os.path.join(csv, 'visualize_df.csv')