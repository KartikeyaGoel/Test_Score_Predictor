import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import json
    
#Define file paths

file_paths = {
    'Economically_Disadvantaged': 'archive\Demographics\Economically_Disadvantaged.csv',
    'English_Learners': 'archive\Demographics\English_Learners.csv',
    'Foster_Care': 'archive\Demographics\Foster_Care.csv',
    'Gender': 'archive\Demographics\Gender.csv',
    'Homeless': 'archive\Demographics\Homeless.csv',
    'Military_Connected': 'archive\Demographics\Military_Connected.csv',
    'Race': 'archive\Demographics\Race.csv',
    'Students_with_Disabilities': 'archive\Demographics\Students_with_Disabilities.csv',
    'Free_and_Reduced_Lunch': 'archive\Economic Factors\Free_and_Reduced_Lunch.csv',
    'Funding': 'archive\Economic Factors\Funding.csv',
    'Absenteeism': 'archive\Student Behaviors\Absenteeism.csv',
    'Dropout': 'archive\Student Behaviors\Dropout.csv',
    'Graduation': 'archive\Student Behaviors\Graduation.csv',
    'Education_Level': 'archive\Teachers\Education_Level.csv',
    'Experience': 'archive\Teachers\Experience.csv',
    'Licensure': 'archive\Teachers\Licensure.csv',
    'SOL_Pass_Rate': 'archive\Testing\SOL_Pass_Rate.csv',
    'school_by_subject_2022': 'archive\Testing\school-by-subject-2022.xlsx'
}


def load_datasets(file_paths):
    datasets = {}
    for key, path in file_paths.items():
        if path.endswith('.csv'):
            datasets[key] = pd.read_csv(path)
        if path.endswith('.xlsx'):
            datasets[key] = pd.read_excel(path)
    return datasets

def preprocess_datasets(datasets):

    rename_columns = {
        'School': 'School Name',
        'Division': 'Division Name',
        'Sch Name': 'School Name',
        'Div Name': 'Division Name',
    }
    for key, df in datasets.items():
        if '2021-2022 Pass Rate' in df.columns:
            df['2021-2022 Pass Rate'] = pd.to_numeric(df['2021-2022 Pass Rate'], errors='coerce')
            mask = df['Subgroup'].str.contains('All Students', na=False)
            df = df[mask]
            df = df.pivot_table(index=['Sch Name', 'Div Name'], columns='Subject', values='2021-2022 Pass Rate', fill_value=0)
            df = df.reset_index()
            datasets[key] = df

        df.rename(columns=lambda x: rename_columns.get(x, x), inplace=True)
        df.columns = df.columns.str.strip().str.lower()
        
        if key == 'Race':
            print(f"Processing {key} dataset before pivot:")
            print(df.head())
            if 'school name' not in df.columns and 'division name' not in df.columns:
                df.columns = ['division name', 'school name', 'race', 'total count', 'percent', 'sch_div']
            print(f"Processing {key} dataset after renaming columns:")
            print(df.head())
            df = df.pivot_table(index=['school name', 'division name'], columns='race', values='percent', fill_value=0)
            df.reset_index(inplace=True)
            print(f"Processing {key} dataset after pivot:")
            print(df.head())
            datasets[key] = df


        if 'division name' in df.columns:
            df['division name'] = df['division name'].str.lstrip('\'').str.lstrip(' \'').str.strip()
            df['division name'] = df['division name'].str.replace(r'\s* Public Schools\s*$', '', regex=True)

        if 'school name' in df.columns:
            df['school name'] = df['school name'].str.lstrip('\'').str.lstrip(' \'').str.strip()
        
        if 'degree_percent' in df.columns:
            df = df.pivot_table(index=['school name', 'division name'], columns='degree_type', values='degree_percent', fill_value=0)
            df = df.reset_index()
            datasets[key] = df

        if 'percent_of_inexperienced_teachers' in df.columns:
            df['poverty_level'] =LabelEncoder().fit_transform(df['poverty_level'])
            df['title1_code'] =LabelEncoder().fit_transform(df['title1_code'])

        if 'sch_div' in df.columns:
            df.drop('sch_div', axis=1, inplace=True)

        if 'school name' in df.columns and 'division name' in df.columns:
            df['school name'] = df['school name'] + '_' + df['division name']
            df.rename(columns={'school name': 'School_ID'}, inplace=True)
            df.drop('division name', axis=1, inplace=True)

    return datasets



#importing

#Preprocessing

def YNSorter(df, df_index1, df_columns, df_values, choice1, choice2):
    df[df_values] = df[df_values].str.replace(r',', '', regex=True)
    df[df_values] = pd.to_numeric(df[df_values], errors='coerce')
    pivot_df = df.pivot_table(index=df_index1, columns=df_columns, values=df_values, fill_value=0)
    pivot_df['total_students'] = pivot_df.sum(axis=1)
    pivot_df['choice1_percent'] = pivot_df[choice1] / pivot_df['total_students']
    pivot_df['choice2_percent'] = pivot_df[choice2] / pivot_df['total_students']
    pivot_df.rename_axis("School_ID", axis="columns", inplace=True)
    pivot_df.columns = [df_columns[:4] + '_' + str(col) for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    return pivot_df



def merge_data_collections(data_frame1, data_frame2, merge_key):
    data_frame1 = list(data_frame2.values()) + [data_frame1[key] for key in data_frame1 if key not in data_frame2]
    result = data_frame1[0]
    for df in data_frame1[1:]:
        result = result.merge(df, on=merge_key, how='inner')
    return result

result_columns = [
    'School_ID', 'disa_choice1_percent_x', 'engl_choice1_percent', 'fost_choice1_percent', 'gend_choice1_percent',
    'home_choice1_percent', 'mili_choice1_percent', 'American Indian or Alaska Native', 'Asian', 'Black, not of Hispanic origin',
    'Hispanic', 'Native Hawaiian  or Pacific Islander', 'Non-Hispanic, two or more races', 'White, not of Hispanic origin',
    'disa_choice1_percent_y', 'percent eligible', 'end_of_year_average_daily_membership', 'school_level_expenditures_per_pupil_federal',
    'school_level_expenditures_per_pupil_state', 'division_level_expenditures_per_pupil_federal', 'division_level_expenditures_per_pupil_state',
    'total_per_pupil_expenditures', 'total_expenditures', 'number_of_students_missing_10__or_more_of_the_days_enrolled',
    'number_of_students_enrolled_for_half_the_year_or_more', 'chronic_absenteeism_rate', 'cohort_dropout_rate', 'diplomas', 'geds',
    'certificates_of_completion', 'still_enrolled', 'graduation_completion_index', "Bachelor's Degree", 'Doctoral Degree', "Master's Degree",
    'poverty_level_x', 'title1_code', 'percent_of_inexperienced_teachers', 'percent_of_out_of_field_teachers',
    'percent_of_out_of_field_and_inexperienced_teachers', 'provisional_percent', 'english: reading', 'english: writing',
    'history and social sciences', 'mathematics', 'science', 'sol pass rate'
]


#

#Scaling and Reshaping

def scale_data(df, features, target):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df[target].values
    return X, y, scaler

# Split the data into training and testing sets
# Define the RandomForestRegressor model
def build_and_train_model(X_train, y_train):
    regressor = RandomForestRegressor(random_state=0)
    param_grid = {
        'n_estimators': [10, 50, 100, 200, 500],
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'bootstrap': [True, False],
        'max_features': ['sqrt', 'log2', None]
    }
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, random_state=0)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# User input for x value
# .Test 0.274111675	0.137532134	0.002538071	0.47715736	0	0.124365482	0.126903553	11.4213198	20.68527919	22.96954315	0.76142132	10.53299492	33.50253807	0.120558376	25.8	800.97	216	7404	571	3640	11831	9476229

def prepare_input_data(user_input, scaler):
    input_array = np.array([[user_input['disa_choice1_percent_x'],
                                                         user_input['engl_choice1_percent'],
                                                         user_input['fost_choice1_percent'],
                                                         user_input['gend_choice1_percent'],
                                                         user_input['home_choice1_percent'],
                                                         user_input['mili_choice1_percent'],

                                                         user_input['American Indian or Alaska Native'],
                                                         user_input['Asian'],
                                                         user_input['Black, not of Hispanic origin'],
                                                         user_input['Hispanic'],
                                                         user_input['Native Hawaiian  or Pacific Islander'],
                                                         user_input['Non-Hispanic, two or more races'],
                                                         user_input['White, not of Hispanic origin'],

                                                         user_input['disa_choice1_percent_y'],
                                                         user_input['percent eligible'],

                                                         user_input['end_of_year_average_daily_membership'],
                                                         user_input['school_level_expenditures_per_pupil_federal'],
                                                         user_input['school_level_expenditures_per_pupil_state'],
                                                         user_input['division_level_expenditures_per_pupil_federal'],
                                                         user_input['division_level_expenditures_per_pupil_state'],
                                                         user_input['total_per_pupil_expenditures'],
                                                         user_input['total_expenditures'],
                                                         user_input['number_of_students_missing_10__or_more_of_the_days_enrolled'],
                                                         user_input['number_of_students_enrolled_for_half_the_year_or_more'],
                                                         user_input['chronic_absenteeism_rate'],

                                                         user_input['cohort_dropout_rate'],
                                                         user_input['diplomas'],
                                                         user_input['geds'],
                                                         user_input['certificates_of_completion'],
                                                         user_input['still_enrolled'],
                                                         user_input['graduation_completion_index'],
                                                         user_input['Bachelor\'s Degree'],
                                                         user_input['Doctoral Degree'],
                                                         user_input['Master\'s Degree'],
                                                         user_input['poverty_level_x'],
                                                         user_input['title1_code'],
                                                         user_input['percent_of_inexperienced_teachers'],
                                                         user_input['percent_of_out_of_field_teachers'],
                                                         user_input['percent_of_out_of_field_and_inexperienced_teachers'],
                                                         user_input['provisional_percent'],
                                                         user_input['english: reading'],
                                                         user_input['english: writing'],
                                                         user_input['history and social sciences'],
                                                         user_input['mathematics'],
                                                         user_input['science']

                                                         ]]).reshape(1, -1)
    return scaler.transform(input_array)


def toggle_entry(var, entry):
    if var.get():
        entry.pack(side='left')
    else:
        entry.pack_forget()


def on_manual_select():
    global selected_files
    manual_window = tk.Toplevel(root)
    manual_window.title("Manual File Selection")
    manual_window.geometry("1200x800")


        
    # Further processing and model training can go here
    #    messagebox.showinfo("Info", "Datasets loaded and preprocessed.")

    tk.Button(manual_window, text="Apply", command=on_apply).pack()


def on_upload_select():
    global selected_files
    filetypes = [("Text files", ".txt"), ("All files", "*.*")]
    filenames = filedialog.askopenfilenames(title="Select Files", filetypes=filetypes)
    library = {}

    #selected_files = {f"File_{i+1}": path for i, path in enumerate(filenames)}
    with open('C:\\Users\\ellio\\OneDrive\\Desktop\\New folder (2)\\Softwqare Dev\\Sample.txt', 'r', encoding='utf-8') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line at the first colon and get the first part
            parts = line.split(':', 1)
            
            text_before_colon = line.split(':', 1)[0]
            text_after_colon = parts[1].strip()  # Get the part after the colon and strip any leading/trailing whitespace
            # Append the result to the list
            key = parts[0].strip()  # Get the part before the colon and strip any leading/trailing whitespace
            value = parts[1].strip()  # Get the part after the colon and strip any leading/trailing whitespace
            library[key] = value
        print(library)
        file_paths = {}
        for key, path in library.items():
            if key == 'disa_choice1_percent_x':
                file_paths['Economically_Disadvantaged'] = 'archive\Demographics\Economically_Disadvantaged.csv'
            elif key == 'engl_choice1_percent':
                file_paths['English_Learners'] = 'archive\Demographics\English_Learners.csv'
            elif key == 'fost_choice1_percent':
                file_paths['Foster_Care'] = 'archive\Demographics\Foster_Care.csv'
            elif key == 'gend_choice1_percent':
                file_paths['Gender'] = 'archive\Demographics\Gender.csv'
            elif key == 'home_choice1_percent':
                file_paths['Homeless'] = 'archive\Demographics\Homeless.csv'
            elif key == 'mili_choice1_percent':
                file_paths['Military_Connected'] = 'archive\Demographics\Military_Connected.csv'
            elif key == 'American Indian or Alaska Native' or key == 'Asian' or key == 'Black, not of Hispanic origin' or key == 'Hispanic' or key == 'Native Hawaiian or Pacific Islander' or key == 'Non-Hispanic, two or more races' or key == 'White, not of Hispanic origin':
                file_paths['Race'] = 'archive\Demographics\Race.csv'
            elif key == 'disa_choice1_percent_y':
                file_paths['Students_with_Disabilities'] = 'archive\Demographics\Students_with_Disabilities.csv'
            elif key == 'percent eligible':
                file_paths['Free_and_Reduced_Lunch'] = 'archive\Economic Factors\Free_and_Reduced_Lunch.csv'
            elif key == 'end_of_year_average_daily_membership' or key == 'school_level_expenditures_per_pupil_federal' or key == 'school_level_expenditures_per_pupil_state' or key == 'division_level_expenditures_per_pupil_federal' or key == 'division_level_expenditures_per_pupil_state' or key == 'total_per_pupil_expenditures' or key == 'total_expenditures':
                file_paths['Funding'] = 'archive\Economic Factors\Funding.csv'
            elif key == 'number_of_students_missing_10_or_more_of_the_days_enrolled' or key == 'number_of_students_enrolled_for_half_the_year_or_more' or key == 'chronic_absenteeism_rate':
                file_paths['Absenteeism'] = 'archive\Student Behaviors\Absenteeism.csv'
            elif key == 'cohort_dropout_rate':
                file_paths['Dropout'] = 'archive\Student Behaviors\Dropout.csv'
            elif key == 'diplomas' or key == 'geds' or key == 'certificates_of_completion' or key == 'still_enrolled' or key == 'graduation_completion_index':
                file_paths['Graduation'] = 'archive\Student Behaviors\Graduation.csv'
            elif key == 'Bachelors Degree' or key == 'Doctoral Degree' or key == 'Masters Degree':
                file_paths['Education_Level'] = 'archive\Teachers\Education_Level.csv'
            elif key == 'poverty_level_x' or key == 'title1_code' or key == 'percent_of_inexperienced_teachers' or key == 'percent_of_out_of_field_teachers' or key == 'percent_of_out_of_field_and_inexperienced_teachers':
                file_paths['Experience'] = 'archive\Teachers\Experience.csv'
            elif key == 'provisional_percent':
                file_paths['Licensure'] = 'archive\Teachers\Licensure.csv'
            elif key == 'english: reading' or key == 'english: writing' or key == 'history and social sciences' or key == 'mathematics' or key == 'science':
                file_paths['school_by_subject_2022'] = 'archive\Testing\school-by-subject-2022.xlsx'
            elif key == 'sol pass rate':
                file_paths['SOL_Pass_Rate'] = 'archive\Testing\SOL_Pass_Rate.csv'
            
            else:
                print("not included")
        
        datasets = load_datasets(file_paths)
        datasets = preprocess_datasets(datasets)
        print(datasets)
        print(datasets['English_Learners'].columns)

        yn_datasets = {
            'Economically_Disadvantaged': YNSorter(datasets['Economically_Disadvantaged'], 'School_ID', 'disadvantaged', 'total count', 'Y', 'N') if 'Economically_Disadvantaged' in datasets else None,
            'English_Learners': YNSorter(datasets['English_Learners'], 'School_ID', 'english learners', 'total count', 'Y', 'N') if 'English_Learners' in datasets else None,
            'Foster_Care': YNSorter(datasets['Foster_Care'], 'School_ID', 'foster care', 'total count', 'Y', 'N') if 'Foster_Care' in datasets else None,
            'Gender': YNSorter(datasets['Gender'], 'School_ID', 'gender', 'total count', 'Female', 'Male') if 'Gender' in datasets else None,
            'Homeless': YNSorter(datasets['Homeless'], 'School_ID', 'homeless', 'total count', 'Y', 'N') if 'Homeless' in datasets else None,
            'Military_Connected': YNSorter(datasets['Military_Connected'], 'School_ID', 'military', 'total count', 'Y', 'N') if 'Military_Connected' in datasets else None,
            'Students_with_Disabilities': YNSorter(datasets['Students_with_Disabilities'], 'School_ID', 'disabled', 'total count', 'Y', 'N') if 'Students_with_Disabilities' in datasets else None
        }

        school_subjects = datasets['school_by_subject_2022']

        # Assuming the columns are named correctly
        datasets['english_reading'] = school_subjects[['School_ID', 'english: reading']]
        datasets['english_writing'] = school_subjects[['School_ID', 'english: writing']]
        datasets['history_and_social_sciences'] = school_subjects[['School_ID', 'history and social sciences']]
        datasets['mathematics'] = school_subjects[['School_ID', 'mathematics']]
        datasets['science'] = school_subjects[['School_ID', 'science']]

        datasets['english_reading'] = datasets['english_reading'][~(datasets['english_reading'] == 0).any(axis=1)]
        datasets['english_writing'] = datasets['english_writing'][~(datasets['english_writing'] == 0).any(axis=1)]
        datasets['history_and_social_sciences'] = datasets['history_and_social_sciences'][~(datasets['history_and_social_sciences'] == 0).any(axis=1)]
        datasets['mathematics'] = datasets['mathematics'][~(datasets['mathematics'] == 0).any(axis=1)]
        datasets['science'] = datasets['science'][~(datasets['science'] == 0).any(axis=1)]

        datasets['school_by_subject_2022']

        result = merge_data_collections(datasets, yn_datasets, 'School_ID')
   
        result.columns = result.columns.str.strip()
        mainlist = list(library.keys())
        mainlist = mainlist = [item.strip() for item in mainlist]

        print(result)
        print(result.columns)
        print(list(library.keys()))
        result = result[list(library.keys())]
        result.to_csv('Factors_Used.csv', index=False)
        features = result.columns.difference(['School_ID', 'sol pass rate'])
        target = 'sol pass rate'

        X, y, scaler = scale_data(result, features, target)
        print("Length of X_array:", len(X))
        print("Length of Y_array:", len(y))

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        best_regressor = build_and_train_model(X_train, y_train)
        mse = evaluate_model(best_regressor, X_test, y_test)

        x_input_for_prediction = collect_user_input()
        # Predict Percent Passing SOL for the user-inputted x value
        predicted_y = best_regressor.predict(x_input_for_prediction)

        # Extract the predicted value from the array (to avoid deprecation warning)
        predicted_y_scalar = np.squeeze(predicted_y)

        # Display the predicted Percent Passing SOL
        print("Predicted SOL Pass Rate:", predicted_y_scalar)
        
datasets = load_datasets(file_paths)
datasets = preprocess_datasets(datasets)
