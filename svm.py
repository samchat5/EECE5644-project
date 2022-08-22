import pandas as pd
import math
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#################################################### Importing and clening the data ###############################################

# Importing the df and checking if it was imported correctly.
df = pd.read_csv(r"./demographics.csv")
# print(df.head())
print(len(df["income"]))


def remove_nan(df):
    float_nan = float("nan")
    df = df.replace([float_nan], "NaN")
    for col in df.columns:
        for element in range(len(df[col])):
            if df.at[element, col] == "NaN":
                df = df.drop(element)
    return df


def turn_values_to_numeric(df):
    for col in df.columns:
        for element in range(len(df[col])):
            try:
                df.at[element, col] = float(df.at[element, col])
            except:
                continue
    return df


def turn_strings_into_int(df, col_with_large_numbers):
    for col in col_with_large_numbers:
        for element in range(len(df[col])):
            if isinstance(df.at[element, col], str):
                if "," in df.at[element, col]:
                    int_element = float(df.at[element, col].replace(",", ""))
                    df.at[element, col] = int_element

                # removed a row where the income was less than $2654. That seems like an error in the data especially since it contains a '-' as well.
                else:
                    # print(col, element, df.at[element,col])
                    df = df.drop(element)
    return df


# Normalize income, age, population
def normalize_columns(df, columns_to_normalize):
    for col in columns_to_normalize:
        # mean normilization
        df[col] = (df[col] - df[col].mean()) / df[col].std()

        # min/max normilization
        # df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
    return df


# Check if all the data that we want is in float. The only two things that are not are county and state.
# print(df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))


def turn_string_variable_into_binary(df, string_variables):
    list_of_new_columns = []
    list_of_headers = []

    for string_variable in string_variables:
        length_of_col = len(df[string_variable])
        unique_string_variable_list = df[string_variable].unique()
        string_variable_list_from_data = df[string_variable].tolist()

        for var in unique_string_variable_list:
            list_of_headers.append(var)
            var = [0] * length_of_col
            for element in range(len(string_variable_list_from_data)):
                # print(string_variable_list_from_data[element])
                if var == string_variable_list_from_data[element]:
                    var[element] = 1
            list_of_new_columns.append(var)

    return (list_of_headers, list_of_new_columns)


################################# Run data clean up  ########################################
df = remove_nan(df)

# including county didn't improve the results. You can just add the string county to the list bellow.
col_with_string_vars = ["state"]

column_header, state_county_df = turn_string_variable_into_binary(
    df, col_with_string_vars
)
state_county_df = pd.DataFrame(state_county_df)
state_county_df = state_county_df.transpose()
state_county_df.columns = column_header
df = pd.concat([df, state_county_df], axis=1)
# print(len(state_county_df.columns))

df = turn_values_to_numeric(df)
col_with_large_numbers = ["pop", "income"]
df = turn_strings_into_int(df, col_with_large_numbers)
columns_to_normalize = ["pop", "age", "income"]
df = normalize_columns(df, columns_to_normalize)


######################################## Train the algorithm ################################################

input_data = df.drop(columns=["margin", "state", "county"])
outcome_data = df["margin"]
X_train, X_test, y_train, y_test = train_test_split(
    input_data, outcome_data, test_size=0.2, random_state=0
)


# We pick the values of c and epsilon for which we get the best pridction.
epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
c_values = [1, 2, 3, 4, 5, 6]
for c in c_values:
    for epsilon in epsilon_values:
        regr = make_pipeline(StandardScaler(), SVR(C=c, epsilon=epsilon))
        regr.fit(X_train, y_train)
        print(regr.score(X_test, y_test, sample_weight=None), c, epsilon)
