# Step 4 - Splitting the Dataset into Training and Test Sets. 

from sklearn.model_selection import train_test_split

def split_data(df, target_col="Rates or Price", test_size=0.2, random_state=42):

    # 1. Separate Features (X) and Raw Target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col] # NO LOG TRANSFORMATION HERE! 
    
    # 2. Perform the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test
